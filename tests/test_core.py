"""Unit tests for Prokopton core module."""

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
    SurpriseBuffer,
    VisualTokenizer,
    AudioTokenizer,
    select_ttt_layers,
)

from tests.tinymodel import tiny_model_and_tokenizer


class TestProkoptonConfig:
    def test_defaults(self):
        cfg = ProkoptonConfig()
        # ttt_lr / parametrization are calibrated together: raw updates at 1e-2
        # are the measured fast-learner on Qwen3-VL-4B (docs/memory-budget.md).
        assert cfg.ttt_lr == 1e-2
        assert cfg.ttt_parametrization == "raw"
        assert cfg.ttt_momentum == 0.9
        assert cfg.ttt_n_layers == 5
        assert cfg.cms_rank == 64
        assert cfg.per_capacity == 128
        assert cfg.ttt_trust_region == 0.01

    def test_custom(self):
        cfg = ProkoptonConfig(ttt_lr=0.01, ttt_n_layers=3, cms_rank=8)
        assert cfg.ttt_lr == 0.01
        assert cfg.ttt_n_layers == 3
        assert cfg.cms_rank == 8

    def test_cms_scale_defaults_to_one(self):
        """Historically alpha/rank == 32/16 == 2.0 silently doubled every delta."""
        assert ProkoptonConfig().cms_scale == 1.0
        assert ProkoptonConfig(cms_rank=8).cms_scale == 1.0
        assert ProkoptonConfig(cms_rank=8, cms_alpha=16).cms_scale == 2.0


class TestFastWeight:
    def test_init(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        assert fw.update_count == 0
        assert torch.allclose(fw.delta, torch.zeros_like(lin.weight))
        assert torch.allclose(fw.original_W, lin.weight)

    def test_reset(self):
        lin = nn.Linear(8, 8)
        original = lin.weight.clone()
        fw = FastWeight(lin)
        with torch.no_grad():
            lin.weight.add_(torch.randn_like(lin.weight) * 0.1)
        fw.reset()
        assert torch.allclose(lin.weight, original)

    def test_delta(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=8)
        assert torch.allclose(fw.delta, torch.zeros_like(lin.weight))
        fw.restore(torch.full_like(lin.weight, 0.5))
        assert not torch.allclose(fw.delta, torch.zeros_like(lin.weight))
        assert torch.allclose(fw.delta, torch.full_like(lin.weight, 0.5), atol=1e-5)

    def test_commit_is_one_way(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=8)
        w0 = fw.W0.clone()
        fw.restore(torch.full_like(lin.weight, 0.3))
        fw.commit()
        assert not torch.allclose(fw.W0, w0)
        assert fw.delta.norm().item() == pytest.approx(0.0, abs=1e-6)

    def test_restore_is_idempotent(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=8)
        target = torch.randn_like(lin.weight) * 0.2
        fw.restore(target)
        w1 = lin.weight.detach().clone()
        fw.restore(target)
        assert torch.allclose(lin.weight, w1)
        fw.restore()
        w2 = lin.weight.detach().clone()
        fw.restore()
        fw.restore()
        assert torch.allclose(lin.weight, w2)

    def test_trust_region_bounds_drift(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=4, lr=1.0, trust_region=0.01,
                        parametrization="raw", surprise_threshold=-1e9,
                        surprise_warmup=0, optimizer="sgd")
        for _ in range(200):
            fw.apply_grad(torch.randn_like(lin.weight), surprise=1.0)
        assert fw.drift_ratio <= 0.01 + 1e-6

    def test_decay_pulls_toward_w0(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=4, lr=1.0, decay=0.5, trust_region=0.0,
                        parametrization="raw", surprise_threshold=-1e9,
                        surprise_warmup=0, optimizer="sgd")
        fw.apply_grad(torch.zeros_like(lin.weight), surprise=1.0)
        with torch.no_grad():
            fw._raw.add_(1.0)
            lin.weight.copy_(fw.W0 + fw._raw)
        before = fw.delta.norm().item()
        fw.apply_grad(torch.zeros_like(lin.weight), surprise=1.0)
        assert fw.delta.norm().item() < before

    def test_surprise_gate_skips_low_surprise(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=4, surprise_threshold=0.0, surprise_warmup=5)
        for _ in range(5):
            assert fw.should_update(10.0) is True
        assert fw.should_update(0.001) is False      # far below the running mean
        assert fw.should_update(1000.0) is True      # genuinely surprising

    def test_momentum_does_not_amplify_lr(self):
        """The historical rule reached 50x ttt_lr in steady state."""
        lin = nn.Linear(4, 4)
        fw = FastWeight(lin, rank=2, lr=0.01, momentum=0.9, trust_region=0.0,
                        grad_clip=0.0,
                        surprise_threshold=-1e9, surprise_warmup=0,
                        optimizer="momentum", parametrization="raw")
        g = torch.ones_like(lin.weight)
        for _ in range(200):
            fw.apply_grad(g, surprise=1.0)
        # Normalised EMA: steady-state step per update is lr * ||g||.
        per_step = fw.delta.norm().item() / 200
        assert per_step == pytest.approx(0.01 * g.norm().item(), rel=0.15)

    def test_effective_lr_is_never_surprise_scaled(self):
        lin = nn.Linear(4, 4)
        fw = FastWeight(lin, rank=2, lr=0.01, surprise_cap=5.0)
        assert fw.effective_lr(1000.0) == 0.01
        assert fw.effective_lr(0.0) == 0.01


class TestCMSAdapter:
    def test_init(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        assert cms.rank == 4
        assert cms.A.shape == (4, 8)
        assert cms.B.shape == (8, 4)

    def test_scale_defaults_to_one(self):
        cms = CMSAdapter(FastWeight(nn.Linear(8, 8)), rank=4)
        assert cms.scale == 1.0

    def test_consolidate_and_expand(self):
        lin = nn.Linear(16, 16)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        fw.restore(torch.randn_like(lin.weight) * 0.5)
        cms.consolidate()
        delta = cms.expand()
        assert delta.shape == (16, 16)

    def test_consolidate_reconstructs_delta(self):
        lin = nn.Linear(16, 16)
        fw = FastWeight(lin, rank=16)
        cms = CMSAdapter(fw, rank=16)
        fw.restore(torch.randn_like(lin.weight) * 0.5)
        cms.consolidate()
        err = (cms.expand() - fw.delta).norm() / fw.delta.norm()
        assert err.item() < 1e-5

    def test_consolidate_truncates_when_rank_is_smaller(self):
        lin = nn.Linear(16, 16)
        fw = FastWeight(lin, rank=16)
        cms = CMSAdapter(fw, rank=2)
        fw.restore(torch.randn_like(lin.weight) * 0.5)
        cms.consolidate()
        assert cms.expand().shape == (16, 16)
        assert cms.A.shape == (2, 16)

    def test_save_load_state(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        sd = cms.state_dict()
        cms2 = CMSAdapter(FastWeight(nn.Linear(8, 8)), rank=4)
        cms2.load_state_dict(sd)
        assert torch.allclose(cms.A, cms2.A)
        assert torch.allclose(cms.B, cms2.B)

    def test_load_state_dict_shape_guard(self):
        cms = CMSAdapter(FastWeight(nn.Linear(8, 8)), rank=4)
        bad = {"A": torch.zeros(2, 8), "B": torch.zeros(8, 2)}
        with pytest.raises(ValueError, match="shape mismatch"):
            cms.load_state_dict(bad)

    def test_apply_to_model_is_restore_not_commit(self):
        """apply_to_model() must set W = W0 + Δ, never W0 += Δ."""
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=8)
        cms = CMSAdapter(fw, rank=8)
        fw.restore(torch.full_like(lin.weight, 0.3))
        cms.consolidate()
        w0 = fw.W0.clone()
        cms.apply_to_model()
        after_first = lin.weight.detach().clone()
        assert torch.allclose(fw.W0, w0)          # W0 untouched
        cms.apply_to_model()
        assert torch.allclose(lin.weight, after_first)   # idempotent

    def test_commit_folds_into_w0(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin, rank=8)
        cms = CMSAdapter(fw, rank=8)
        fw.restore(torch.full_like(lin.weight, 0.3))
        cms.consolidate()
        expected = fw.W0 + cms.expand()
        cms.commit()
        assert torch.allclose(fw.W0, expected, atol=1e-5)
        assert cms.expand().abs().max() < 1e-6


class TestSurpriseBuffer:
    def test_push_and_len(self):
        buf = SurpriseBuffer(10)
        assert len(buf) == 0
        buf.push("a", 0.5)
        buf.push("b", 0.9)
        assert len(buf) == 2

    def test_sample(self):
        buf = SurpriseBuffer(20)
        for i in range(10):
            buf.push(f"item{i}", 0.5 + i * 0.05)
        sample = buf.sample(3)
        assert len(sample) == 3
        assert all(isinstance(s, str) for s in sample)

    def test_capacity_limit(self):
        buf = SurpriseBuffer(5)
        for i in range(10):
            buf.push(f"item{i}", 0.1)
        assert len(buf) == 5

    def test_empty_sample(self):
        buf = SurpriseBuffer(10)
        assert buf.sample(5) == []


class TestVisualTokenizer:
    def test_output_shape(self):
        vt = VisualTokenizer(patch_size=48, embed_dim=128, output_dim=256)
        img = torch.randn(1, 3, 96, 96)
        tokens, info = vt(img)
        assert tokens.ndim == 3
        assert tokens.shape[-1] == 256

    def test_multiple_images(self):
        vt = VisualTokenizer(patch_size=32, embed_dim=64, output_dim=128)
        img = torch.randn(2, 3, 64, 64)
        tokens, info = vt(img)
        assert tokens.shape[0] == 2

    def test_non_divisible(self):
        vt = VisualTokenizer(patch_size=48, embed_dim=128, output_dim=256)
        img = torch.randn(1, 3, 100, 100)
        tokens, info = vt(img)
        assert tokens.ndim == 3

    def test_deprecated(self):
        with pytest.warns(DeprecationWarning):
            VisualTokenizer(patch_size=8, embed_dim=16, output_dim=32)


class TestAudioTokenizer:
    def test_output_shape(self):
        import math
        at = AudioTokenizer(sample_rate=16000, n_mels=40, patch_frames=4,
                            embed_dim=128, output_dim=256)
        audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, 16000))
        tokens, info = at(audio)
        assert len(tokens) == 1
        assert tokens[0].shape[-1] == 256

    def test_short_audio(self):
        import math
        at = AudioTokenizer(sample_rate=16000, n_mels=40, patch_frames=4,
                            embed_dim=128, output_dim=256)
        audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 0.01, 160))
        tokens, info = at(audio)
        assert len(tokens) == 1
        assert tokens[0].shape[-1] == 256


class TestLayerSelection:
    def test_selects_language_model_down_proj(self):
        model, _ = tiny_model_and_tokenizer(n_layers=4)
        selected = select_ttt_layers(model, 3)
        assert len(selected) == 3
        for name, module in selected:
            assert name.startswith("model.language_model.layers.")
            assert name.endswith(".mlp.down_proj")

    def test_never_selects_the_vision_tower(self):
        model, _ = tiny_model_and_tokenizer(n_layers=4, with_visual=True)
        selected = select_ttt_layers(model, 10)
        for name, _ in selected:
            assert "visual" not in name
            assert "vision" not in name
            assert "audio" not in name

    def test_raises_without_language_model_layers(self):
        class NoLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)

        with pytest.raises(ValueError, match="No language-model"):
            select_ttt_layers(NoLM(), 2)


class TestProkoptonCore:
    """Integration tests with a tiny language model."""

    @pytest.fixture
    def dummy_model_and_tokenizer(self):
        return tiny_model_and_tokenizer(vocab=128, hidden=32, intermediate=64,
                                        n_layers=4, with_visual=True)

    def _cfg(self, **kw):
        base = dict(ttt_n_layers=2, auto_save_every=0, per_capacity=8,
                    ttt_surprise_threshold=-1e9, ttt_surprise_warmup=0,
                    kl_weight=0.0, async_save=False)
        base.update(kw)
        return ProkoptonConfig(**base)

    def test_prokopton_create(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg())
        assert len(prok.fast_weights) == 2
        assert len(prok.cms_adapters) == 2
        for name in prok.ttt_layer_names:
            assert name.startswith("model.language_model.layers.")
            assert name.endswith(".mlp.down_proj")

    def test_learn(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg())
        result = prok.learn("hello world test")
        assert "loss" in result
        assert "step" in result
        assert result["step"] == 1

    def test_learn_multiple(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg(ttt_n_layers=1, per_capacity=4))
        for i in range(5):
            r = prok.learn(f"test message {i}")
            assert r["step"] == i + 1

    def test_stats(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg(ttt_n_layers=1, per_capacity=4))
        prok.learn("test")
        s = prok.stats
        assert s["steps"] == 1
        assert s["updates"] >= 1
        assert "max_drift_ratio" in s

    def test_save_load_reset(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg(ttt_n_layers=1, per_capacity=4))
        prok.learn("important fact about Zephyria")

        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp)
            assert os.path.exists(os.path.join(tmp, "metadata.json"))
            prok.reset()
            assert prok.step_counter == 0
            assert prok.load(tmp)
        finally:
            shutil.rmtree(tmp)

    def test_chat(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        prok = Prokopton(model, tok, self._cfg(ttt_n_layers=1, per_capacity=4))
        resp = prok.chat("hello", max_new=5)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_generate_does_not_echo_prompt(self, dummy_model_and_tokenizer):
        model, tok = tiny_model_and_tokenizer()
        prok = Prokopton(model, tok, self._cfg())
        prompt = "Question: What is the capital of France?\nAnswer:"
        out = prok.generate(prompt, max_new=4)
        assert prompt not in out
