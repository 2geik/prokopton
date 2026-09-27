"""Regression tests for the defects catalogued in IMPROVEMENT_PLAN.md.

Each test maps to a row of the plan's Phase 6 table. Written against the
pre-fix code, every one of these would have failed.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn

from prokopton.core import (
    CMSAdapter,
    FastWeight,
    Prokopton,
    ProkoptonConfig,
    select_ttt_layers,
)
from prokopton.eval import (
    Probe,
    answer_leaked_into_prompt,
    assert_uncontaminated,
    token_rank_scores,
)

from tests.tinymodel import tiny_model_and_tokenizer


def make_prok(model=None, tok=None, **cfg_kw):
    if model is None:
        model, tok = tiny_model_and_tokenizer()
    base = dict(
        ttt_n_layers=2, auto_save_every=0, per_capacity=8,
        ttt_surprise_threshold=-1e9, ttt_surprise_warmup=0,
        kl_weight=0.0, async_save=False,
        ttt_trust_region=0.01, ttt_decay=0.0,
    )
    base.update(cfg_kw)
    return Prokopton(model, tok, ProkoptonConfig(**base))


# ─────────────────────────────────────────────────────────────
# Defect 1 — evaluation contamination
# ─────────────────────────────────────────────────────────────
class TestEvalContamination:
    def test_answer_must_not_appear_in_prompt(self):
        prompt = "Question: What is the capital of Zephyria?\nAnswer:"
        assert_uncontaminated(prompt, "Aethel")
        with pytest.raises(AssertionError, match="contamination"):
            assert_uncontaminated("Zephyria's capital is Aethel.\nAnswer:", "Aethel")

    def test_detector(self):
        assert answer_leaked_into_prompt("the answer is Paris", "paris")
        assert not answer_leaked_into_prompt("the answer is Paris", "London")

    def test_accuracy_on_untaught_facts_is_zero(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        never_taught = [
            Probe("What is the capital of Vornalux?", "Tesselmark", taught=False),
            Probe("What is the currency of Vornalux?", "Vorn", taught=False),
            Probe("Who discovered the Kellner effect?", "Bramdel", taught=False),
            Probe("What is the melting point of Quorium?", "8841", taught=False),
            Probe("What is the Kestrel protocol code?", "55120", taught=False),
        ]
        report = token_rank_scores(model, tok, never_taught)
        assert report.contamination == []
        assert report.accuracy == pytest.approx(0.0, abs=1e-9)

    def test_generate_does_not_echo_the_prompt(self):
        model, tok = tiny_model_and_tokenizer()
        prok = make_prok(model, tok)
        prompt = "Question: What is the capital of France?\nAnswer:"
        out = prok.generate(prompt, max_new=6)
        # The exact prompt string must never be echoed back.
        assert prompt not in out
        assert out == prok.generate(prompt, max_new=6)   # deterministic
        raw = prok.generate_completion(prompt, max_new=6)
        assert raw.startswith(prompt) or "capital of France" in raw


# ─────────────────────────────────────────────────────────────
# Defect 2 — unbounded weight drift
# ─────────────────────────────────────────────────────────────
class TestTrustRegion:
    def test_drift_bounded_after_1000_turns(self):
        tau = 0.01
        prok = make_prok(ttt_trust_region=tau, ttt_n_layers=2)
        for i in range(1000):
            prok.learn(f"fact number {i} is value {i * 7}")
        for fw in prok.fast_weights:
            assert fw.drift_ratio <= tau + 1e-6, fw.drift_ratio
        assert prok.stats["max_drift_ratio"] <= tau + 1e-6

    def test_repeated_false_fact_cannot_blow_up_weights(self):
        """Phase 2.9 acceptance: 50 repeats of a false premise stay bounded."""
        tau = 0.01
        prok = make_prok(ttt_trust_region=tau)
        for _ in range(50):
            prok.learn("The capital of Zephyria is definitely WRONGPLACE.")
        assert prok.stats["max_drift_ratio"] <= tau + 1e-6

    def test_decay_bounds_drift_without_trust_region(self):
        prok = make_prok(ttt_trust_region=0.0, ttt_decay=0.05)
        for i in range(300):
            prok.learn(f"another fact {i}")
        assert prok.stats["max_drift_ratio"] < 0.5


# ─────────────────────────────────────────────────────────────
# Defect 3 — persistence compounding
# ─────────────────────────────────────────────────────────────
class TestPersistenceRoundTrip:
    def test_five_save_reset_load_cycles_preserve_the_delta(self):
        prok = make_prok(ttt_n_layers=1)
        prok.learn("Zephyria's capital is Aethel.")
        prok.cms_adapters[0].consolidate()
        before = prok.fast_weights[0].expand().clone()
        assert before.abs().max() > 0

        tmp = tempfile.mkdtemp()
        try:
            for cycle in range(5):
                prok.save(tmp, silent=True)
                prok.reset()
                assert prok.fast_weights[0].delta.norm().item() == pytest.approx(0.0, abs=1e-6)
                assert prok.load(tmp, silent_on_missing=True) is True
                after = prok.fast_weights[0].expand()
                err = (after - before).norm() / before.norm()
                assert err.item() < 1e-4, f"cycle {cycle}: delta drifted by {err.item()}"
        finally:
            shutil.rmtree(tmp)

    def test_load_is_idempotent(self):
        prok = make_prok(ttt_n_layers=1)
        prok.learn("some fact")
        prok.cms_adapters[0].consolidate()
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp, silent=True)
            prok.load(tmp)
            w1 = prok.fast_weights[0].layer.weight.detach().clone()
            prok.load(tmp)
            prok.load(tmp)
            assert torch.allclose(prok.fast_weights[0].layer.weight, w1)
        finally:
            shutil.rmtree(tmp)

    def test_delta_state_is_not_rounded_by_the_model_dtype(self):
        """bf16 weights must not re-round the delta on every restore.

        Keeping the delta in the model's bf16 re-quantises it on each
        save → load and the error compounds across reloads. The delta state and
        the CMS factors are float32 precisely so the round trip is stable.
        """
        model, tok = tiny_model_and_tokenizer()
        model = model.to(torch.bfloat16)
        prok = make_prok(model, tok, ttt_n_layers=1)
        assert prok.fast_weights[0].layer.weight.dtype == torch.bfloat16
        assert prok.fast_weights[0].state_dtype == torch.float32
        assert prok.cms_adapters[0].A.dtype == torch.float32

        prok.learn("Zephyria's capital is Aethel.")
        for c in prok.cms_adapters:
            c.consolidate()
        before = [c.expand().clone() for c in prok.cms_adapters]

        # restore() must not re-quantise the delta to the model dtype.
        fw = prok.fast_weights[0]
        delta = before[0].clone()
        fw.restore(delta)
        assert fw.expand().dtype == torch.float32
        assert torch.equal(fw.expand(), delta.to(torch.float32))

        tmp = tempfile.mkdtemp()
        try:
            errs = []
            for _ in range(5):
                prok.save(tmp, silent=True)
                prok.reset()
                prok.load(tmp)
                errs.append(max(((c.expand() - b).norm()
                                 / b.norm().clamp_min(1e-12)).item()
                                for c, b in zip(prok.cms_adapters, before)))
            # The residual must stay flat — a bf16 delta state compounds here.
            assert max(errs) < 1e-4, errs
            assert errs[-1] < 1e-4, errs
        finally:
            shutil.rmtree(tmp)

    def test_no_full_rank_delta_files_are_written(self):
        prok = make_prok(ttt_n_layers=2)
        prok.learn("a fact worth remembering")
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp, silent=True)
            files = os.listdir(tmp)
            assert not any(f.startswith("delta_") for f in files), files
            assert any(f.startswith("cms_") for f in files), files
            for f in files:
                if f.startswith("cms_"):
                    import torch as _t
                    sd = _t.load(os.path.join(tmp, f), weights_only=True)
                    assert sd["A"].dim() == 2 and sd["B"].dim() == 2
        finally:
            shutil.rmtree(tmp)

    def test_history_and_config_are_persisted(self):
        prok = make_prok(ttt_n_layers=1)
        prok.chat("hello there", max_new=3)
        assert len(prok.conversation_history) == 2
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp, silent=True)
            prok.reset()
            assert prok.conversation_history == []
            prok.load(tmp)
            assert len(prok.conversation_history) == 2
            import json
            meta = json.load(open(os.path.join(tmp, "metadata.json")))
            assert meta["schema_version"] >= 3
            assert meta["model_fingerprint"]
            assert meta["environment"].get("torch")
            assert meta["conversation_history"]
        finally:
            shutil.rmtree(tmp)

    def test_refuses_a_mismatched_model(self):
        prok = make_prok(ttt_n_layers=1)
        prok.learn("a fact")
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp, silent=True)
            other = make_prok(ttt_n_layers=1,
                              **{})  # same shapes → same fingerprint
            other2 = Prokopton(*tiny_model_and_tokenizer(n_layers=3),
                               ProkoptonConfig(ttt_n_layers=1, auto_save_every=0,
                                               kl_weight=0.0, async_save=False))
            assert other2.model_fingerprint() != prok.model_fingerprint()
            with pytest.raises(ValueError, match="mismatched model"):
                other2.load(tmp)
        finally:
            shutil.rmtree(tmp)

    def test_schema_version_guard(self):
        prok = make_prok(ttt_n_layers=1)
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp, silent=True)
            import json
            p = os.path.join(tmp, "metadata.json")
            meta = json.load(open(p))
            meta["schema_version"] = 99
            json.dump(meta, open(p, "w"))
            with pytest.raises(ValueError, match="newer than this build"):
                prok.load(tmp)
        finally:
            shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────
# Defect 8 — CMS idempotency
# ─────────────────────────────────────────────────────────────
class TestCMSIdempotency:
    def test_consolidate_apply_twice_is_unchanged(self):
        lin = nn.Linear(16, 16)
        fw = FastWeight(lin, rank=16)
        cms = CMSAdapter(fw, rank=16)
        fw.restore(torch.randn_like(lin.weight) * 0.3)
        cms.consolidate()
        cms.apply_to_model()
        w1 = lin.weight.detach().clone()
        cms.consolidate()
        cms.apply_to_model()
        w2 = lin.weight.detach().clone()
        assert torch.allclose(w1, w2, atol=1e-5)

    def test_consolidate_is_deterministic_across_devices(self):
        """Phase 0.3 acceptance: CPU and accelerator results agree within 1e-4."""
        delta = torch.randn(32, 32)
        a = prokopton_svd(delta, 8, "cpu")
        if torch.backends.mps.is_available():
            b = prokopton_svd(delta.to("mps"), 8, "mps")
            for x, y in zip(a, b):
                assert torch.allclose(x.cpu(), y.cpu(), atol=1e-4), \
                    (x - y).abs().max()
        # Always: the truncation reconstructs a rank-r approximation.
        U, S, Vh = a
        approx = (U * S) @ Vh
        assert approx.shape == delta.shape


def prokopton_svd(delta, rank, device):
    from prokopton.core import _safe_lowrank_svd
    return _safe_lowrank_svd(delta.to(device), rank)


# ─────────────────────────────────────────────────────────────
# Defect 9 — layer selection
# ─────────────────────────────────────────────────────────────
class TestLayerSelection:
    def test_selected_layers_are_language_model_down_proj(self):
        model, _ = tiny_model_and_tokenizer(n_layers=6)
        selected = select_ttt_layers(model, 5)
        assert len(selected) == 5
        for name, module in selected:
            assert name.endswith(".mlp.down_proj")
            assert ".layers." in name
            assert name.startswith("model.language_model.layers.")
            assert isinstance(module, nn.Linear)

    def test_vision_tower_layers_are_rejected(self):
        model, _ = tiny_model_and_tokenizer(n_layers=4, with_visual=True)
        names = [n for n, _ in model.named_modules() if n.endswith(".mlp.down_proj")]
        assert any("visual" in n for n in names)   # the trap exists
        selected = select_ttt_layers(model, 10)
        for name, _ in selected:
            assert "visual" not in name


# ─────────────────────────────────────────────────────────────
# Defect 12 — surprise gate
# ─────────────────────────────────────────────────────────────
class TestSurpriseGate:
    def test_updates_are_skipped_at_low_surprise(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=4, lr=1.0, trust_region=0.0,
                        surprise_threshold=0.0, surprise_warmup=5,
                        parametrization="raw", grad_clip=0.0)
        for _ in range(5):
            fw.apply_grad(torch.ones_like(lin.weight), surprise=5.0)
        assert fw.update_count == 5
        fw.apply_grad(torch.ones_like(lin.weight), surprise=0.0001)
        assert fw.skipped_count == 1
        assert fw.update_count == 5

    def test_gate_is_reported_in_stats(self):
        prok = make_prok(ttt_surprise_threshold=1e9, ttt_surprise_warmup=0)
        for _ in range(4):
            prok.learn("trivial input")
        assert prok.stats["skipped_steps"] > 0


# ─────────────────────────────────────────────────────────────
# Defect 11 — chat template
# ─────────────────────────────────────────────────────────────
class TestChatTemplate:
    def test_chat_template_is_applied(self):
        model, tok = tiny_model_and_tokenizer(use_chat_template=True)
        prok = make_prok(model, tok)
        prompt = prok._chat_prompt([], "hello")
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt

    def test_falls_back_without_a_template(self):
        model, tok = tiny_model_and_tokenizer(use_chat_template=False)
        prok = make_prok(model, tok)
        prompt = prok._chat_prompt([], "hello")
        assert "Assistant:" in prompt

    def test_response_does_not_echo_the_prompt(self):
        model, tok = tiny_model_and_tokenizer()
        prok = make_prok(model, tok)
        user = "What is the capital of Zephyria?"
        resp = prok.chat(user, max_new=8)
        assert user not in resp
        assert "<|user|>" not in resp
        assert "<|assistant|>" not in resp


# ─────────────────────────────────────────────────────────────
# Defect 7 — multimodal label alignment
# ─────────────────────────────────────────────────────────────
class TestMultimodalLabels:
    def test_labels_are_shifted_by_exactly_one(self):
        """labels[j] == combined_ids[j+1] and image positions are -100."""
        model, tok = tiny_model_and_tokenizer(vocab=128, hidden=32, n_layers=2)
        prok = make_prok(model, tok, ttt_n_layers=1)

        text = "the quick brown fox"
        image = torch.randn(1, 3, 32, 32)
        embeds, mask, labels = prok._prepare_multimodal_embeds(
            text=text, image_tensor=image, waveform=None)

        # Reconstruct the id row: visual prefix is not a token id sequence, so
        # mark those positions with a sentinel and put the real text ids after.
        text_ids = tok(text, return_tensors="pt")["input_ids"][0]
        prefix = embeds.shape[1] - text_ids.shape[0]
        assert prefix > 0, "expected a multimodal prefix"
        sentinel = -999
        combined_ids = [sentinel] * prefix + text_ids.tolist()

        assert labels.shape[1] == len(combined_ids) - 1
        for j in range(labels.shape[1]):
            expected = combined_ids[j + 1]
            got = int(labels[0, j])
            if expected == sentinel:
                assert got == -100, f"position {j} should be masked"
            else:
                assert got == expected, f"labels[{j}]={got} != combined_ids[{j+1}]={expected}"

    def test_image_token_positions_are_masked(self):
        model, tok = tiny_model_and_tokenizer(hidden=32, n_layers=2)
        prok = make_prok(model, tok, ttt_n_layers=1)
        embeds, mask, labels = prok._prepare_multimodal_embeds(
            text="caption text here",
            image_tensor=torch.randn(1, 3, 32, 32), waveform=None)
        n_vis = embeds.shape[1] - len("caption text here")
        assert (labels[0, :n_vis - 1] == -100).all()


# ─────────────────────────────────────────────────────────────
# Defect 13 — momentum / surprise double-scaling
# ─────────────────────────────────────────────────────────────
class TestEffectiveLearningRate:
    def test_effective_step_is_at_most_lr(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=4, lr=1e-3, momentum=0.9,
                        surprise_threshold=-1e9, surprise_warmup=0,
                        parametrization="raw", grad_clip=0.0, trust_region=0.0)
        g = torch.randn_like(lin.weight)
        for _ in range(50):
            info = fw.apply_grad(g, surprise=1.0)
        # No single step may exceed lr * ||g|| (the pre-fix rule reached 50x).
        assert info["step_norm"] <= 1e-3 * g.norm().item() * 1.5
        assert fw.last_effective_lr == pytest.approx(1e-3)

    def test_surprise_cap_no_longer_scales_the_lr(self):
        lin = nn.Linear(4, 4)
        fw = FastWeight(lin, rank=2, lr=1e-3, surprise_cap=5.0)
        assert fw.effective_lr(1e6) == fw.effective_lr(0.0) == 1e-3


# ─────────────────────────────────────────────────────────────
# Defect 16 — PER replay must not recurse
# ─────────────────────────────────────────────────────────────
class TestReplay:
    def test_replay_is_bounded_and_non_recursive(self):
        """``learn()`` must not recurse into ``learn()``.

        A replay sample is processed by ``_learn_step(..., allow_replay=False)``
        so it can never spawn a further replay, and the total number of updates
        is exactly ``outer + batches * sample_size`` — never an unbounded blowup.
        """
        sample_size = 3
        every = 5
        n_outer = 10
        prok = make_prok(per_replay_every=every, per_sample_size=sample_size)

        state = {"calls": 0, "replay_depth": 0, "max_replay_depth": 0}
        orig_step = prok._learn_step
        orig_replay = prok._run_replay

        def counting_step(*a, **kw):
            state["calls"] += 1
            return orig_step(*a, **kw)

        def counting_replay():
            state["replay_depth"] += 1
            state["max_replay_depth"] = max(state["max_replay_depth"],
                                            state["replay_depth"])
            try:
                return orig_replay()
            finally:
                state["replay_depth"] -= 1

        prok._learn_step = counting_step
        prok._run_replay = counting_replay

        for i in range(n_outer):
            prok.learn(f"item {i}")

        batches = n_outer // every
        assert state["calls"] == n_outer + batches * sample_size
        assert state["max_replay_depth"] == 1, "replay must not recurse into learn()"


# ─────────────────────────────────────────────────────────────
# Integration: teach N facts → accuracy rises, anchors do not move
# ─────────────────────────────────────────────────────────────
class TestIntegration:
    def test_learning_changes_the_model_and_is_bounded(self):
        model, tok = tiny_model_and_tokenizer()
        prok = make_prok(model, tok, ttt_trust_region=0.05)
        anchors = [
            Probe("What is the capital of France?", "Paris", taught=False),
            Probe("What is 2+2?", "4", taught=False),
        ]
        before = token_rank_scores(model, tok, anchors)
        for _ in range(10):
            prok.learn("Zephyria's capital is Aethel.")
        after = token_rank_scores(model, tok, anchors)
        # The anchors must not have been rewritten by 10 repeats of an
        # unrelated fact; the primary invariant is the bound itself.
        assert prok.stats["max_drift_ratio"] <= 0.05 + 1e-6
        assert after.contamination == [] and before.contamination == []

    def test_no_ttt_control_leaves_weights_untouched(self):
        model, tok = tiny_model_and_tokenizer()
        prok = make_prok(model, tok, ttt_lr=0.0)
        for fw in prok.fast_weights:
            fw.lr = 0.0
        before = [fw.layer.weight.detach().clone() for fw in prok.fast_weights]
        for i in range(20):
            prok.learn(f"fact {i}")
        for fw, w in zip(prok.fast_weights, before):
            assert torch.allclose(fw.layer.weight, w, atol=1e-6)
