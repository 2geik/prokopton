"""
Prokopton Models — Platform-agnostic model loading.

Uses prokopton.backends for auto-detection.
"""
from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.backends import (
    detect_backend,
    load_model,
    apply_backend_patches,
    get_vram_usage,
    MLXUnsupportedForLearning,
)


# Default target: Qwen3-VL-4B-Instruct. Verified loadable with
# transformers >= 4.57 on MPS in bf16. The 2B is retained for CI / low-memory
# and the 8B as the stretch option on 24 GB.
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def load_prokopton(
    model_name: str = DEFAULT_MODEL,
    lr: float = 1e-3,
    n_layers: int = 5,
    backend: str = None,
    allow_non_torch: bool = False,
    **config_overrides,
) -> Prokopton:
    """
    Load a Prokopton-wrapped model with optimal backend.

    Args:
        model_name: HuggingFace model ID or local path
        lr: TTT learning rate
        n_layers: Number of TTT layers
        backend: Force backend ("rocm", "cuda", "mps", "mlx", "cpu")
        allow_non_torch: Permit the MLX backend. MLX models cannot be trained
            with TTT; this is only meaningful with ``--no-ttt`` frozen chat.
        **config_overrides: forwarded to :class:`ProkoptonConfig`

    Returns:
        Prokopton instance ready for chat/learn
    """
    be = detect_backend(force=backend, require_torch=not allow_non_torch)

    print(f"🖥️  Backend: {be.description}")
    print(f"   GPU: {be.gpu_name}")

    apply_backend_patches(be)

    print(f"Loading {model_name}...")
    model, tokenizer = load_model(model_name, be, allow_non_torch=allow_non_torch)

    vram = get_vram_usage(be) or be.vram_gb
    print(f"   VRAM: {vram:.1f} GB")

    config = ProkoptonConfig(ttt_lr=lr, ttt_n_layers=n_layers, **config_overrides)
    prok = Prokopton(model, tokenizer, config)

    print(f"   TTT layers: {len(prok.fast_weights)} -> {prok.ttt_layer_names}")
    return prok


# Model registry — every entry verified against the Hugging Face registry and
# the locally installed transformers version.
AVAILABLE_MODELS = {
    "qwen3vl-2b": {
        "name": "Qwen/Qwen3-VL-2B-Instruct",
        "params": "2.2B",
        "vram_bf16": "4.5 GB",
        "multimodal": True,
        "auto_model": "AutoModelForImageTextToText",
        "description": "Small native VLM. Fast CI / low-memory target.",
    },
    "qwen3vl-4b": {
        "name": DEFAULT_MODEL,
        "params": "4.0B",
        "vram_bf16": "8.0 GB",
        "multimodal": True,
        "auto_model": "AutoModelForImageTextToText",
        "description": "Primary target. Native vision tower; comfortable on 24 GB.",
    },
    "qwen3vl-8b": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "params": "8.0B",
        "vram_bf16": "16 GB",
        "multimodal": True,
        "auto_model": "AutoModelForImageTextToText",
        "description": "Stretch target. Workable on 24 GB, tighter headroom.",
    },
    "gemma4-e2b": {
        "name": "google/gemma-4-E2B",
        "params": "5.1B",
        "vram_bf16": "9.5 GB",
        "multimodal": True,
        "auto_model": "AutoModelForMultimodalLM",
        "description": (
            "Natively multimodal (vision + audio). Requires transformers>=5.5. "
            "Held as a separate target for its unique native-audio path."
        ),
    },
}

__all__ = ["DEFAULT_MODEL", "AVAILABLE_MODELS", "load_prokopton",
           "MLXUnsupportedForLearning"]
