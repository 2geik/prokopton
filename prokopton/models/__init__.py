"""
Prokopton Models — Platform-agnostic model loading.

Uses prokopton.backends for auto-detection.
"""
from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.backends import detect_backend, load_model, apply_backend_patches, get_vram_usage


def load_prokopton(
    model_name: str = "google/gemma-4-E2B",
    lr: float = 1e-3,
    n_layers: int = 5,
    backend: str = None,
) -> Prokopton:
    """
    Load a Prokopton-wrapped model with optimal backend.

    Args:
        model_name: HuggingFace model ID or local path
        lr: TTT learning rate
        n_layers: Number of TTT layers
        backend: Force backend ("rocm", "cuda", "mps", "mlx", "cpu")

    Returns:
        Prokopton instance ready for chat/learn
    """
    be = detect_backend(force=backend)

    print(f"🖥️  Backend: {be.description}")
    print(f"   GPU: {be.gpu_name}")

    apply_backend_patches(be)

    print(f"Loading {model_name}...")
    model, tokenizer = load_model(model_name, be)

    vram = get_vram_usage(be) or be.vram_gb
    print(f"   VRAM: {vram:.1f} GB")

    config = ProkoptonConfig(ttt_lr=lr, ttt_n_layers=n_layers)
    prok = Prokopton(model, tokenizer, config)

    print(f"   TTT layers: {len(prok.fast_weights)}")
    return prok


# Model registry
AVAILABLE_MODELS = {
    "e2b": {
        "name": "google/gemma-4-E2B",
        "params": "5.1B",
        "vram_bf16": "9.5 GB",
        "description": "Lightweight, ideal for fast iteration",
    },
    "e4b": {
        "name": "google/gemma-4-E4B",
        "params": "7.9B",
        "vram_bf16": "14.2 GB",
        "description": "Mid-size, tight on 16 GB VRAM",
    },
    "12b": {
        "name": "google/gemma-4-12B",
        "params": "12B",
        "vram_bf16": "24+ GB",
        "description": "Most powerful, needs 24+ GB or quantization",
    },
}
