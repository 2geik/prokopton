"""Unit tests for Prokopton backends module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prokopton.backends import (
    detect_backend,
    BackendInfo,
    backend_summary,
    apply_backend_patches,
    _resolve_model_path,
    _is_apple_silicon,
)


class TestDetectBackend:
    def test_auto_detect(self):
        be = detect_backend()
        assert be.available
        assert be.name in ("rocm", "cuda", "mps", "cpu")

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

    def test_force_mlx(self):
        be = detect_backend(force="mlx")
        assert be.name == "mlx"


class TestBackendInfo:
    def test_defaults(self):
        be = BackendInfo(name="test", device="cpu", description="Test")
        assert be.vram_gb == 0.0
        assert not be.available

    def test_torch_dtype_cpu(self):
        be = detect_backend(force="cpu")
        dtype_str = str(be.torch_dtype)
        assert "float32" in dtype_str

    def test_torch_dtype_gpu(self):
        be = detect_backend(force="rocm")
        if be.available:
            dtype_str = str(be.torch_dtype)
            assert "bfloat16" in dtype_str

    def test_summary(self):
        be = detect_backend(force="cpu")
        s = backend_summary(be)
        assert "name" in s
        assert "device" in s
        assert "dtype" in s


class TestPatches:
    def test_apply_patches_no_crash(self):
        be = detect_backend(force="cpu")
        apply_backend_patches(be)  # should not crash


class TestResolvePath:
    def test_hf_id(self):
        p = _resolve_model_path("google/gemma-4-E2B")
        assert p == "google/gemma-4-E2B"

    def test_with_models_prefix(self):
        # Should check models/ folder, fall back to original
        p = _resolve_model_path("nonexistent-model")
        assert p == "nonexistent-model"
