"""Prokopton — continual-learning, non-forgetting, multimodal LLM (Nested Learning based).

Konuştukça ağırlıkları güncellenen, deneyim biriktikçe büyüyen, eskiyi unutmayan LLM.
"""

__version__ = "0.0.1"

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

__all__ = [
    "Prokopton",
    "ProkoptonConfig",
    "FastWeight",
    "CMSAdapter",
    "SurpriseBuffer",
    "VisualTokenizer",
    "AudioTokenizer",
    "CLBenchmark",
    "run_full_evaluation",
    "load_prokopton",
    "AVAILABLE_MODELS",
]
