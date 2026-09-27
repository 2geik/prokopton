"""
Prokopton Backend — Platform-agnostic GPU/CPU model loading.

Auto-detects the best available backend:
    ROCm (AMD) > CUDA (NVIDIA) > MPS (macOS) > MLX (Apple Silicon) > CPU

MLX is a *frozen-inference* backend only: its models are not ``torch.nn.Module``
instances, so test-time training (``torch.autograd.grad`` over module weights)
cannot touch them. :func:`detect_backend` therefore never returns MLX when a
learning context is requested, and :func:`load_model` refuses MLX unless
``allow_non_torch=True``.

Usage:
    from prokopton.backends import detect_backend, load_model

    backend = detect_backend(require_torch=True)
    model, tokenizer = load_model("Qwen/Qwen3-VL-4B-Instruct", backend)
"""

import json
import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
from pathlib import Path


class MLXUnsupportedForLearning(RuntimeError):
    """Raised when an MLX model is requested in a test-time-training context."""


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
    supports_ttt: bool = True         # False for MLX: no torch autograd path

    @property
    def torch_dtype(self):
        """Best dtype for this backend."""
        import torch
        if self.name == "cpu":
            return torch.float32
        return torch.bfloat16


def detect_backend(force: Optional[str] = None, require_torch: bool = False) -> BackendInfo:
    """
    Detect the best available GPU/CPU backend.

    Args:
        force: Override detection ("cuda", "cpu", "mps", "mlx", "rocm")
        require_torch: When True, never return the MLX backend. Use this for any
            learning/TTT context — an MLX model has no ``torch.autograd`` path.

    Returns:
        BackendInfo with device details
    """
    import torch

    if force:
        force = force.lower()
        info = _build_forced(force)
        if require_torch and info.name == "mlx":
            raise MLXUnsupportedForLearning(
                "MLX was requested but cannot support test-time training "
                "(the returned model is not a torch.nn.Module, so "
                "torch.autograd.grad cannot update its weights). "
                "Use --backend mps, or pass allow_non_torch=True for frozen chat only.")
        return info

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
        # MLX first only when frozen inference is acceptable.
        if not require_torch:
            mlx_info = _build_mlx()
            if mlx_info.available:
                return mlx_info

        # MPS — the only Apple-Silicon path that supports TTT.
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
        supports_ttt=False,
    )
    try:
        import mlx.core as mx  # noqa: F401
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

# architecture suffix → transformers auto class name
_ARCH_TO_AUTO = (
    ("ForImageTextToText", "AutoModelForImageTextToText"),
    ("ForMultimodalLM", "AutoModelForMultimodalLM"),
    ("ForConditionalGeneration", "AutoModelForImageTextToText"),
    ("ForCausalLM", "AutoModelForCausalLM"),
    ("ForSeq2SeqLM", "AutoModelForSeq2SeqLM"),
)

# multimodal `model_type` values that need a multimodal auto class
_MULTIMODAL_MODEL_TYPES = {
    "qwen3_vl", "qwen2_vl", "qwen2_5_vl", "gemma4", "llava", "llava_next",
    "llava_onevision", "internvl", "paligemma", "chameleon", "aria",
    "aya_vision", "blip", "blip-2", "fuyu", "kosmos-2", "mllama", "qwen_vl",
}


def read_model_config(model_id_or_path: str) -> Dict[str, Any]:
    """Read a model's ``config.json`` without loading weights."""
    source = _resolve_model_path(model_id_or_path)
    cfg_path = Path(source) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    # Remote id: ask the hub for the config only.
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_id_or_path, filename="config.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_auto_model_class(model_id_or_path: str):
    """Pick the right ``AutoModelFor*`` class for a model.

    Dispatch order:

    1. ``transformersInfo.auto_model`` on the model card, when present.
    2. The ``architectures`` suffix (``*ForConditionalGeneration`` on a
       multimodal config → ``AutoModelForImageTextToText``; Gemma4 declares
       ``AutoModelForMultimodalLM``).
    3. ``model_type`` membership in the known multimodal set.
    4. ``AutoModelForCausalLM`` as the text-only fallback.
    """
    import transformers

    cfg = read_model_config(model_id_or_path)

    info = cfg.get("transformersInfo") or {}
    if isinstance(info, dict):
        declared = info.get("auto_model")
        cls = getattr(transformers, declared, None) if declared else None
        if cls is not None:
            return cls

    model_type = (cfg.get("model_type") or "").lower()
    is_multimodal = model_type in _MULTIMODAL_MODEL_TYPES or any(
        k in cfg for k in ("vision_config", "audio_config", "visual_config", "speech_config"))
    # Any-to-any models (Gemma4 Unified) declare `AutoModelForMultimodalLM`;
    # vision-language models declare `AutoModelForImageTextToText`. Both auto
    # classes resolve to the same concrete architecture, so the split is a
    # declaration-level distinction, not a behavioural one.
    is_any_to_any = any(k in cfg for k in ("audio_config", "speech_config"))

    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    for suffix, auto_name in _ARCH_TO_AUTO:
        if arch.endswith(suffix):
            if suffix == "ForConditionalGeneration" and not is_multimodal:
                continue
            if suffix == "ForConditionalGeneration" and is_any_to_any:
                cls = getattr(transformers, "AutoModelForMultimodalLM", None)
                if cls is not None:
                    return cls
            cls = getattr(transformers, auto_name, None)
            if cls is not None:
                return cls

    if is_multimodal:
        ordered = ("AutoModelForMultimodalLM", "AutoModelForImageTextToText") \
            if is_any_to_any else ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")
        for auto_name in ordered:
            cls = getattr(transformers, auto_name, None)
            if cls is not None:
                return cls

    return transformers.AutoModelForCausalLM


def load_model(
    model_id_or_path: str,
    backend: Optional[BackendInfo] = None,
    allow_non_torch: bool = False,
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with optimal backend.

    Args:
        model_id_or_path: HF model ID or local path
        backend: Detected backend (auto-detected if None)
        allow_non_torch: Permit the MLX path. MLX models are not
            ``torch.nn.Module`` s and cannot be used with TTT; leave this False
            for any learning context.
        **kwargs: passed to from_pretrained (dtype, device_map, etc.)

    Returns:
        (model, tokenizer) tuple
    """
    if backend is None:
        backend = detect_backend(require_torch=not allow_non_torch)

    # MLX path — frozen inference only.
    if backend.name == "mlx" and backend.available:
        if not allow_non_torch:
            raise MLXUnsupportedForLearning(
                "Refusing to load an MLX model for a learning context. MLX "
                "models are not torch.nn.Module instances, so test-time "
                "training cannot update their weights. Use the MPS backend "
                "(--backend mps), or pass allow_non_torch=True for --no-ttt "
                "frozen chat.")
        return _load_mlx(model_id_or_path)

    # PyTorch path (ROCm, CUDA, MPS, CPU)
    return _load_pytorch(model_id_or_path, backend, **kwargs)


def _load_pytorch(model_id_or_path: str, backend: BackendInfo, **kwargs):
    """Load model via PyTorch + transformers."""
    from transformers import AutoTokenizer
    import torch

    apply_backend_patches(backend)

    source = _resolve_model_path(model_id_or_path)

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": kwargs.pop("torch_dtype", backend.torch_dtype),
        **kwargs,
    }

    if backend.device == "cuda" and "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
    elif backend.device == "mps":
        model_kwargs.pop("device_map", None)

    if backend.device == "cpu":
        model_kwargs.pop("device_map", None)
        if model_kwargs.get("torch_dtype") == torch.bfloat16:
            model_kwargs["torch_dtype"] = torch.float32

    auto_cls = resolve_auto_model_class(source)
    model = auto_cls.from_pretrained(source, **model_kwargs)

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
                  stream: bool = False,
                  return_completion: bool = False):
    """
    Unified text generation across backends.

    By default returns **only the generated tokens** (the prompt is sliced off),
    so callers can assert on the completion without prompt-echo contamination.

    Args:
        model: Model (PyTorch or MLX)
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new: Max new tokens
        backend: Backend info (auto-detect if None)
        stream: If True, yields tokens one at a time
        return_completion: If True, return prompt+completion (legacy behaviour)

    Returns:
        Generated text string, or generator if stream=True
    """
    if backend is None:
        backend = detect_backend()

    if backend.name == "mlx" and backend.available:
        text = mlx_generate(model, tokenizer, prompt, max_new)
        if stream:
            return (t for t in [text])
        return text if return_completion else text

    # PyTorch path
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if stream:
        return _stream_generate(model, tokenizer, inputs, max_new)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    if return_completion:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


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
    """Get current device memory usage in GB."""
    import torch
    if backend.device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    if backend.device == "mps" and getattr(torch, "mps", None) is not None:
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except Exception:
            return 0.0
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
        "supports_ttt": backend.supports_ttt,
    }


def print_backend_info(backend: BackendInfo):
    """Print backend detection result to console."""
    info = backend_summary(backend)
    print(f"🖥️  Backend: {info['description']}")
    print(f"   GPU: {info['gpu']}")
    if info["vram_gb"] > 0:
        print(f"   VRAM: {info['vram_gb']} GB")
    print(f"   Dtype: {info['dtype']}")
    if not info["supports_ttt"]:
        print("   ⚠ TTT unsupported on this backend (frozen inference only)")
