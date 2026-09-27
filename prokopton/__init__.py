"""Prokopton — continual-learning, non-forgetting, multimodal LLM.

Talk to it and its weights update; give it experience and it accumulates;
persist it and the knowledge survives a restart — without unbounded drift and
without compounding the saved delta on every load.
"""

__version__ = "0.5.0"

from prokopton.core import (
    MEMORY_SCHEMA_VERSION,
    Prokopton,
    ProkoptonConfig,
    FastWeight,
    CMSAdapter,
    SurpriseBuffer,
    VisualTokenizer,
    AudioTokenizer,
    select_ttt_layers,
)
from prokopton.eval import (
    CLBenchmark,
    Probe,
    EvalReport,
    FrozenBaseline,
    ProbeResult,
    token_rank_scores,
    generation_accuracy,
    evaluate_with_control,
    run_full_evaluation,
    run_ablation,
    answer_leaked_into_prompt,
    assert_uncontaminated,
)
from prokopton.models import load_prokopton, AVAILABLE_MODELS, DEFAULT_MODEL
from prokopton.backends import (
    detect_backend,
    load_model,
    apply_backend_patches,
    get_vram_usage,
    backend_summary,
    BackendInfo,
    generate_text,
    mlx_generate,
    resolve_auto_model_class,
    MLXUnsupportedForLearning,
)
from prokopton.config import ProkoptonCLIConfig, load_config, save_config

__all__ = [
    # Core
    "MEMORY_SCHEMA_VERSION",
    "Prokopton",
    "ProkoptonConfig",
    "FastWeight",
    "CMSAdapter",
    "SurpriseBuffer",
    "VisualTokenizer",
    "AudioTokenizer",
    "select_ttt_layers",
    # Eval
    "CLBenchmark",
    "Probe",
    "ProbeResult",
    "EvalReport",
    "FrozenBaseline",
    "token_rank_scores",
    "generation_accuracy",
    "evaluate_with_control",
    "run_full_evaluation",
    "run_ablation",
    "answer_leaked_into_prompt",
    "assert_uncontaminated",
    # Models
    "load_prokopton",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    # Backends
    "detect_backend",
    "load_model",
    "apply_backend_patches",
    "get_vram_usage",
    "backend_summary",
    "BackendInfo",
    "generate_text",
    "mlx_generate",
    "resolve_auto_model_class",
    "MLXUnsupportedForLearning",
    # Config
    "ProkoptonCLIConfig",
    "load_config",
    "save_config",
]
