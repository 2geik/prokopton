"""Unit tests for Prokopton backends module."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prokopton.backends import (
    BackendInfo,
    MLXUnsupportedForLearning,
    _resolve_model_path,
    apply_backend_patches,
    backend_summary,
    detect_backend,
    generate_text,
    read_model_config,
    resolve_auto_model_class,
)


class TestDetectBackend:
    def test_auto_detect_is_never_mlx_for_learning(self):
        """detect_backend(require_torch=True) must never hand back MLX."""
        be = detect_backend(require_torch=True)
        assert be.available
        assert be.name in ("rocm", "cuda", "mps", "cpu")
        assert be.supports_ttt

    def test_auto_detect_default_matches_test_expectation(self):
        be = detect_backend()
        assert be.available
        assert be.name in ("rocm", "cuda", "mps", "cpu", "mlx")

    def test_force_rocm(self):
        be = detect_backend(force="rocm")
        assert be.name == "rocm"
        assert be.is_amd
        assert be.needs_warmup_patch

    def test_force_cuda(self):
        be = detect_backend(force="cuda")
        assert be.name == "cuda"
        assert be.is_nvidia

    def test_force_cpu(self):
        be = detect_backend(force="cpu")
        assert be.name == "cpu"
        assert be.device == "cpu"

    def test_force_mps(self):
        be = detect_backend(force="mps")
        assert be.name == "mps"
        assert be.supports_ttt

    def test_force_mlx(self):
        be = detect_backend(force="mlx")
        assert be.name == "mlx"
        assert be.supports_ttt is False

    def test_force_mlx_for_learning_raises(self):
        with pytest.raises(MLXUnsupportedForLearning):
            detect_backend(force="mlx", require_torch=True)


class TestMLXTrap:
    """MLX returns a non-torch model; TTT needs torch.autograd."""

    def test_load_model_refuses_mlx_without_opt_in(self):
        from prokopton.backends import load_model
        be = BackendInfo(name="mlx", device="cpu", description="Apple MLX",
                         available=True, supports_ttt=False)
        with pytest.raises(MLXUnsupportedForLearning, match="not torch.nn.Module"):
            load_model("some/model", be)

    def test_load_model_allows_mlx_for_frozen_chat(self, monkeypatch):
        from prokopton import backends
        be = BackendInfo(name="mlx", device="cpu", description="Apple MLX",
                         available=True, supports_ttt=False)
        calls = []
        monkeypatch.setattr(backends, "_load_mlx",
                            lambda mid: calls.append(mid) or ("m", "t"))
        model, tok = backends.load_model("some/model", be, allow_non_torch=True)
        assert calls == ["some/model"]

    def test_backend_summary_flags_ttt(self):
        be = detect_backend(force="cpu")
        assert backend_summary(be)["supports_ttt"] is True
        be = detect_backend(force="mlx")
        assert backend_summary(be)["supports_ttt"] is False


class TestBackendInfo:
    def test_defaults(self):
        be = BackendInfo(name="test", device="cpu", description="Test")
        assert be.vram_gb == 0.0
        assert not be.available

    def test_torch_dtype_cpu(self):
        be = detect_backend(force="cpu")
        assert "float32" in str(be.torch_dtype)

    def test_torch_dtype_gpu(self):
        be = detect_backend(force="rocm")
        if be.available:
            assert "bfloat16" in str(be.torch_dtype)

    def test_summary(self):
        be = detect_backend(force="cpu")
        s = backend_summary(be)
        assert "name" in s
        assert "device" in s
        assert "dtype" in s


class TestPatches:
    def test_apply_patches_no_crash(self):
        apply_backend_patches(detect_backend(force="cpu"))


class TestResolvePath:
    def test_hf_id(self):
        assert _resolve_model_path("google/gemma-4-E2B") == "google/gemma-4-E2B"

    def test_with_models_prefix(self):
        assert _resolve_model_path("nonexistent-model") == "nonexistent-model"


def _remote_config(model_id):
    try:
        cfg = read_model_config(model_id)
    except Exception as exc:  # pragma: no cover - offline CI
        pytest.skip(f"cannot read remote config for {model_id}: {exc}")
    if not cfg:
        pytest.skip(f"config for {model_id} unavailable offline")
    return cfg


class TestAutoModelDispatch:
    """Phase 0.1 — the loader must pick a multimodal auto class for VL models."""

    def test_qwen3vl_needs_image_text_to_text(self):
        from transformers import AutoModelForImageTextToText
        cfg = _remote_config("Qwen/Qwen3-VL-4B-Instruct")
        assert cfg["model_type"] == "qwen3_vl"
        assert resolve_auto_model_class("Qwen/Qwen3-VL-4B-Instruct") is AutoModelForImageTextToText

    def test_gemma4_uses_multimodal_lm(self):
        from transformers import AutoModelForMultimodalLM
        cfg = _remote_config("google/gemma-4-E2B")
        assert cfg["model_type"] == "gemma4"
        assert resolve_auto_model_class("google/gemma-4-E2B") is AutoModelForMultimodalLM

    def test_text_only_model_still_gets_causal_lm(self, tmp_path):
        import json
        from transformers import AutoModelForCausalLM
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }))
        assert resolve_auto_model_class(str(tmp_path)) is AutoModelForCausalLM

    def test_gemma4_is_natively_multimodal(self):
        """The "E2B is text-only" premise was wrong — it is any-to-any."""
        cfg = _remote_config("google/gemma-4-E2B")
        assert "vision_config" in cfg
        assert "audio_config" in cfg
        assert "image_token_id" in cfg
        assert "audio_token_id" in cfg
