"""
Prokopton Backend — Platform-agnostic GPU/CPU model loading.

Auto-detects the best available backend:
    ROCm (AMD) > CUDA (NVIDIA) > MPS (macOS) > MLX (Apple Silicon) > CPU

Usage:
    from prokopton.backends import detect_backend, load_model

    backend = detect_backend()
    model, tokenizer = load_model("google/gemma-4-E2B", backend)
"""

import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
from pathlib import Path


@dataclass
class BackendInfo:
    """Detected hardware backend information."""
    name: str           # "rocm", "cuda", "mps", "mlx", "cpu"
    device: str         # torch device string: "cuda", "mps", "cpu"
    description: str    # human-readable: "AMD ROCm", "Apple MLX", etc.
    vram_gb: float = 0.0
    gpu_name: str = ""
    available: bool = False
    is_apple_silicon: bool = False
    is_amd: bool = False
    is_nvidia: bool = False
    needs_warmup_patch: bool = False  # ROCm monkey-patch needed

    @property
    def torch_dtype(self):
        """Best dtype for this backend."""
        import torch
        if self.name == "cpu":
            return torch.float32
        return torch.bfloat16


def detect_backend(force: Optional[str] = None) -> BackendInfo:
    """
    Detect the best available GPU/CPU backend.

    Args:
        force: Override detection ("cuda", "cpu", "mps", "mlx", "rocm")

    Returns:
        BackendInfo with device details
    """
    import torch

    if force:
        force = force.lower()
        return _build_forced(force)

    # 1. ROCm (presents as CUDA with HIP)
    if torch.cuda.is_available():
        try:
            is_hip = torch.version.hip is not None
        except Exception:
            is_hip = False

        if is_hip:
            return _build_rocm(torch)

        # 2. NVIDIA CUDA
        return _build_cuda(torch)

    # 3. Apple Silicon
    if platform.system() == "Darwin" and _is_apple_silicon():
        # Try MLX first (best performance on Apple Silicon)
        mlx_info = _build_mlx()
        if mlx_info.available:
            return mlx_info

        # Fall back to MPS
        if torch.backends.mps.is_available():
            return _build_mps(torch)

    # 4. CPU fallback
    return _build_cpu(torch)


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    if platform.system() != "Darwin":
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        cpu = result.stdout.strip().lower()
        return "apple" in cpu and any(x in cpu for x in ["m1", "m2", "m3", "m4"])
    except Exception:
        return platform.machine() == "arm64"


def _build_rocm(torch) -> BackendInfo:
    info = BackendInfo(
        name="rocm",
        device="cuda",
        description="AMD ROCm",
        available=True,
        is_amd=True,
        needs_warmup_patch=True,
    )
    try:
        info.gpu_name = torch.cuda.get_device_name(0)
        info.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        info.gpu_name = "AMD GPU"
    return info


def _build_cuda(torch) -> BackendInfo:
    info = BackendInfo(
        name="cuda",
        device="cuda",
        description="NVIDIA CUDA",
        available=True,
        is_nvidia=True,
    )
    try:
        info.gpu_name = torch.cuda.get_device_name(0)
        info.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        info.gpu_name = "NVIDIA GPU"
    return info


def _build_mps(torch) -> BackendInfo:
    info = BackendInfo(
        name="mps",
        device="mps",
        description="Apple MPS (Metal Performance Shaders)",
        available=True,
        is_apple_silicon=True,
    )
    info.gpu_name = "Apple Silicon (MPS)"
    try:
        info.vram_gb = _get_macos_memory_gb()
    except Exception:
        pass
    return info


def _build_mlx() -> BackendInfo:
    info = BackendInfo(
        name="mlx",
        device="cpu",  # MLX doesn't use torch device
        description="Apple MLX",
        is_apple_silicon=True,
    )
    try:
        import mlx.core as mx
        info.available = True
        info.gpu_name = "Apple Silicon (MLX)"
        info.vram_gb = _get_macos_memory_gb()
    except ImportError:
        info.available = False
    return info


def _build_cpu(torch) -> BackendInfo:
    return BackendInfo(
        name="cpu",
        device="cpu",
        description="CPU (no GPU detected)",
        available=True,
        gpu_name="CPU",
    )


def _build_forced(force: str) -> BackendInfo:
    import torch
    mapping = {
        "rocm": _build_rocm,
        "cuda": _build_cuda,
        "mps": _build_mps,
        "mlx": _build_mlx,
        "cpu": _build_cpu,
    }
    if force in mapping:
        builder = mapping[force]
        if force == "mlx":
            return builder()
        return builder(torch)
    # Unknown, fallback
    return _build_cpu(torch)


def _get_macos_memory_gb() -> float:
    """Get total system memory on macOS (unified memory)."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / 1024**3
    except Exception:
        return 0.0


# ============================================================
# Platform-specific patches
# ============================================================

def apply_backend_patches(backend: BackendInfo):
    """Apply platform-specific patches before model loading."""
    if backend.needs_warmup_patch:
        # ROCm: monkey-patch caching_allocator_warmup to avoid OOM
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            import transformers.modeling_utils as tmu
            if hasattr(tmu, "caching_allocator_warmup"):
                tmu.caching_allocator_warmup = lambda *a, **kw: None
        except ImportError:
            pass


# ============================================================
# Unified model loader
# ============================================================

def load_model(
    model_id_or_path: str,
    backend: Optional[BackendInfo] = None,
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with optimal backend.

    Args:
        model_id_or_path: HF model ID or local path
        backend: Detected backend (auto-detected if None)
        **kwargs: passed to from_pretrained (dtype, device_map, etc.)

    Returns:
        (model, tokenizer) tuple
    """
    if backend is None:
        backend = detect_backend()

    # MLX path
    if backend.name == "mlx" and backend.available:
        return _load_mlx(model_id_or_path)

    # PyTorch path (ROCm, CUDA, MPS, CPU)
    return _load_pytorch(model_id_or_path, backend, **kwargs)


def _load_pytorch(model_id_or_path: str, backend: BackendInfo, **kwargs):
    """Load model via PyTorch + transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    apply_backend_patches(backend)

    # Resolve path
    source = _resolve_model_path(model_id_or_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model kwargs
    model_kwargs = {
        "torch_dtype": kwargs.pop("torch_dtype", backend.torch_dtype),
        **kwargs,
    }

    # Device map
    if backend.device == "cuda" and "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
    elif backend.device == "mps":
        model_kwargs.pop("device_map", None)

    # Offload unsupported kwargs for CPU
    if backend.device == "cpu":
        model_kwargs.pop("device_map", None)
        if model_kwargs.get("torch_dtype") == torch.bfloat16:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)

    # Move to device for MPS/CPU
    if backend.device in ("mps", "cpu"):
        try:
            model = model.to(backend.device)
        except Exception:
            pass  # MPS sometimes fails on specific ops, leave on CPU

    return model, tokenizer


def _load_mlx(model_id_or_path: str):
    """Load model via MLX (Apple Silicon)."""
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        raise ImportError(
            "MLX not installed. Install with: pip install mlx-lm\n"
            "Or use --backend mps for PyTorch MPS backend."
        )

    source = _resolve_model_path(model_id_or_path)
    model, tokenizer = mlx_load(source)
    return model, tokenizer


def mlx_generate(model, tokenizer, prompt: str, max_tokens: int = 256,
                 temp: float = 0.7) -> str:
    """
    Generate text using an MLX model.

    Args:
        model: MLX model (from mlx_lm.load)
        tokenizer: MLX tokenizer
        prompt: Input text
        max_tokens: Maximum tokens to generate
        temp: Sampling temperature

    Returns:
        Generated text
    """
    try:
        from mlx_lm import generate as mlx_gen
    except ImportError:
        raise ImportError("MLX not installed. pip install mlx-lm")

    response = mlx_gen(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    return response


def generate_text(model, tokenizer, prompt: str, max_new: int = 128,
                  backend: Optional[BackendInfo] = None,
                  stream: bool = False):
    """
    Unified text generation across backends.

    Args:
        model: Model (PyTorch or MLX)
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new: Max new tokens
        backend: Backend info (auto-detect if None)
        stream: If True, yields tokens one at a time

    Returns:
        Generated text string, or generator if stream=True
    """
    if backend is None:
        backend = detect_backend()

    if backend.name == "mlx" and backend.available:
        text = mlx_generate(model, tokenizer, prompt, max_new)
        if stream:
            return (t for t in [text])  # MLX doesn't stream easily
        return text

    # PyTorch path
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if stream:
        return _stream_generate(model, tokenizer, inputs, max_new)
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _stream_generate(model, tokenizer, inputs, max_new: int):
    """Generate text token by token (generator)."""
    import torch
    with torch.no_grad():
        generated = inputs["input_ids"]
        for _ in range(max_new):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=-1)
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text


def _resolve_model_path(model_id_or_path: str) -> str:
    """Resolve model ID or local path."""
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        return str(path)
    # Check models/ folder
    local_path = Path("models") / model_id_or_path
    if local_path.exists() and local_path.is_dir():
        return str(local_path)
    return model_id_or_path


def get_vram_usage(backend: BackendInfo) -> float:
    """Get current VRAM usage in GB."""
    import torch
    if backend.device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


# ============================================================
# Backend info for display
# ============================================================

def backend_summary(backend: BackendInfo) -> Dict[str, Any]:
    """Return a display-friendly backend summary."""
    return {
        "name": backend.name.upper(),
        "device": backend.device,
        "description": backend.description,
        "gpu": backend.gpu_name,
        "vram_gb": round(backend.vram_gb, 1),
        "dtype": str(backend.torch_dtype).split(".")[-1],
    }


def print_backend_info(backend: BackendInfo):
    """Print backend detection result to console."""
    info = backend_summary(backend)
    print(f"🖥️  Backend: {info['description']}")
    print(f"   GPU: {info['gpu']}")
    if info["vram_gb"] > 0:
        print(f"   VRAM: {info['vram_gb']} GB")
    print(f"   Dtype: {info['dtype']}")
