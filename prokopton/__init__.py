"""Prokopton — continual-learning, non-forgetting, multimodal LLM (Nested Learning based).

Konuştukça ağırlıkları güncellenen, deneyim biriktikçe büyüyen, eskiyi unutmayan LLM.
"""

__version__ = "0.2.0"

from prokopton.core import (
    Prokopton,
    ProkoptonConfig,
    FastWeight,
    CMSAdapter,
    SurpriseBuffer,
    VisualTokenizer,
    AudioTokenizer,
)
from prokopton.eval import CLBenchmark, run_full_evaluation
from prokopton.models import load_prokopton, AVAILABLE_MODELS
from prokopton.backends import (
    detect_backend,
    load_model,
    apply_backend_patches,
    get_vram_usage,
    backend_summary,
    BackendInfo,
    generate_text,
    mlx_generate,
)
from prokopton.config import ProkoptonCLIConfig, load_config, save_config

__all__ = [
    # Core
    "Prokopton",
    "ProkoptonConfig",
    "FastWeight",
    "CMSAdapter",
    "SurpriseBuffer",
    "VisualTokenizer",
    "AudioTokenizer",
    # Eval
    "CLBenchmark",
    "run_full_evaluation",
    # Models
    "load_prokopton",
    "AVAILABLE_MODELS",
    # Backends
    "detect_backend",
    "load_model",
    "apply_backend_patches",
    "get_vram_usage",
    "backend_summary",
    "BackendInfo",
    "generate_text",
    "mlx_generate",
    # Config
    "ProkoptonCLIConfig",
    "load_config",
    "save_config",
]
