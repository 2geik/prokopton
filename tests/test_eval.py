"""Tests for the contamination-proof evaluation layer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from prokopton.eval import (
    CLBenchmark,
    EvalReport,
    FrozenBaseline,
    Probe,
    ProbeResult,
    answer_leaked_into_prompt,
    answer_token_span,
    assert_uncontaminated,
    default_probes,
    generation_accuracy,
    run_ablation,
    run_full_evaluation,
    token_rank_scores,
)
from prokopton.core import Prokopton, ProkoptonConfig

from tests.tinymodel import tiny_model_and_tokenizer


class TestContaminationGuard:
    def test_detects_leakage(self):
        assert answer_leaked_into_prompt("Paris is the answer", "paris")
        assert not answer_leaked_into_prompt("What is the answer?", "paris")

    def test_assert_raises(self):
        with pytest.raises(AssertionError):
            assert_uncontaminated("The capital is Aethel.", "Aethel")

    def test_empty_answer_never_leaks(self):
        assert not answer_leaked_into_prompt("anything", "")


class TestAnswerTokenSpan:
    def _ids(self, *rows):
        return [torch.tensor(r) for r in rows]

    def test_clean_boundary(self):
        full, prompt, ans = self._ids([1, 2, 3, 4, 5], [1, 2, 3], [4, 5])
        assert answer_token_span(full, prompt, ans) == (3, 5)

    def test_boundary_merge_is_detected(self):
        # The tokenizer merged across the boundary: the prompt alone tokenises
        # to [1, 2, 3] but the full string starts [1, 2, 30].
        full, prompt, ans = self._ids([1, 2, 30, 4, 5], [1, 2, 3], [4, 5])
        start, end = answer_token_span(full, prompt, ans)
        assert full[start:end].tolist() == [4, 5]

    def test_search_fallback(self):
        full, prompt, ans = self._ids([9, 9, 4, 5], [1, 2, 3], [4, 5])
        start, end = answer_token_span(full, prompt, ans)
        assert full[start:end].tolist() == [4, 5]

    def test_missing_answer_raises(self):
        full, prompt, ans = self._ids([1, 2, 3], [1, 2], [7, 8])
        with pytest.raises(ValueError):
            answer_token_span(full, prompt, ans)


class TestTokenRankScores:
    def test_report_shape(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        probes = [Probe("What is 2+2?", "4", "A", taught=False),
                  Probe("Who wrote Hamlet?", "Shakespeare", "A", taught=False)]
        report = token_rank_scores(model, tok, probes)
        assert isinstance(report, EvalReport)
        assert len(report.results) == 2
        assert 0.0 <= report.accuracy <= 1.0
        assert report.mean_rank >= 1.0

    def test_contaminated_probe_is_flagged_and_excluded(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        probes = [Probe("The answer is Tesselmark, what is it?", "Tesselmark")]
        report = token_rank_scores(model, tok, probes)
        assert len(report.contamination) == 1
        assert report.accuracy == 0.0   # excluded from the denominator

    def test_delta_versus_frozen_baseline(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        probes = [Probe("What is 2+2?", "4", taught=False)]
        baseline = FrozenBaseline.capture(model, tok, probes)
        report = token_rank_scores(model, tok, probes)
        delta = baseline.delta(report)
        assert delta["accuracy"] == pytest.approx(0.0)
        assert set(delta) == {"accuracy", "first_token_top1", "mean_rank", "mrr"}

    def test_serialisable(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        report = token_rank_scores(model, tok, [Probe("q?", "a")])
        text = report.to_json()
        assert '"accuracy"' in text
        assert '"environment"' in text


class TestBenchmarks:
    def test_default_probes_have_unseen_and_anchors(self):
        probes = default_probes()
        bench = CLBenchmark(probes)
        assert bench.taught and bench.unseen and bench.anchors
        assert len(bench.anchors) >= 10, "3 anchors saturate by chance"

    def test_no_probe_leaks_its_answer(self):
        for probe in default_probes():
            assert_uncontaminated(f"Question: {probe.question}\nAnswer:", probe.answer)

    def test_grouped_tasks(self):
        bench = CLBenchmark()
        assert bench.tasks
        assert all("questions" in t and t["context"] == "" for t in bench.tasks)


class TestFullEvaluation:
    def test_writes_a_committed_artifact(self, tmp_path):
        model, tok = tiny_model_and_tokenizer()
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=8,
                              ttt_surprise_threshold=-1e9, ttt_surprise_warmup=0,
                              kl_weight=0.0, async_save=False)
        prok = Prokopton(model, tok, cfg)
        results = run_full_evaluation(prok, model_name="tiny-test",
                                      output_dir=str(tmp_path), repeats=1, seed=0)
        assert os.path.exists(results["artifact"])
        assert "metrics" in results
        assert "final_delta_vs_frozen" in results
        assert results["environment"].get("torch")
        for key in ("growth_taught", "anchor_drift", "unseen_change", "max_drift_ratio"):
            assert key in results["metrics"]

    def test_trust_region_is_reported(self, tmp_path):
        model, tok = tiny_model_and_tokenizer()
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0,
                              ttt_trust_region=0.01, kl_weight=0.0, async_save=False,
                              ttt_surprise_threshold=-1e9, ttt_surprise_warmup=0)
        prok = Prokopton(model, tok, cfg)
        results = run_full_evaluation(prok, model_name="tiny-trust",
                                      output_dir=str(tmp_path), repeats=1)
        assert results["metrics"]["max_drift_ratio"] <= 0.01 + 1e-6


class TestAblation:
    def test_ttt_contribution_is_computed(self, tmp_path):
        def factory():
            model, tok = tiny_model_and_tokenizer()
            cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=8,
                                  ttt_surprise_threshold=-1e9, ttt_surprise_warmup=0,
                                  kl_weight=0.0, async_save=False)
            return Prokopton(model, tok, cfg)

        out = run_ablation(factory, model_name="tiny-ablation",
                           output_dir=str(tmp_path), repeats=1)
        assert "ttt_contribution" in out
        assert os.path.exists(out["artifact"])
        assert set(out["ttt_contribution"]) == {"growth_taught", "anchor_drift",
                                                "unseen_change"}


class TestGenerationAccuracy:
    def test_prompt_is_sliced_off(self):
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        probes = [Probe("What is the capital of France?", "Paris", taught=False)]
        report = generation_accuracy(model, tok, probes, max_new=8)
        assert report.results[0].answer_in_prompt is False

    def test_generate_text_backend_slices_the_prompt(self):
        from prokopton.backends import generate_text
        model, tok = tiny_model_and_tokenizer()
        model.eval()
        prompt = "Question: What is the capital of France?\nAnswer:"
        out = generate_text(model, tok, prompt, max_new=8)
        assert prompt not in out
        full = generate_text(model, tok, prompt, max_new=8, return_completion=True)
        assert "Question:" in full
