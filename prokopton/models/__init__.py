"""
Prokopton Models — Gemma 4 graft entegrasyonu.

Gemma 4 E2B/E4B/12B modellerini Prokopton ile sarmalayan fabrika fonksiyonları.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prokopton.core import Prokopton, ProkoptonConfig


def load_prokopton(model_name: str = "google/gemma-4-E2B", 
                   lr: float = 1e-3,
                   n_layers: int = 5,
                   device: str = None) -> Prokopton:
    """
    Prokopton modelini yükle.
    
    Args:
        model_name: HuggingFace model adı
        lr: TTT öğrenme hızı
        n_layers: TTT uygulanacak MLP katman sayısı
        device: "cuda" veya "cpu"
    
    Returns:
        Prokopton instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    config = ProkoptonConfig(
        ttt_lr=lr,
        ttt_n_layers=n_layers,
    )
    
    prokopton = Prokopton(model, tokenizer, config)
    
    vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
    print(f"  VRAM: {vram:.1f} GB  TTT layers: {len(prokopton.fast_weights)}")
    
    return prokopton


# Model registry
AVAILABLE_MODELS = {
    "e2b": {
        "name": "google/gemma-4-E2B",
        "params": "5.1B",
        "vram_bf16": "9.5 GB",
        "description": "En hafif, hızlı iterasyon için ideal",
    },
    "e4b": {
        "name": "google/gemma-4-E4B", 
        "params": "7.9B",
        "vram_bf16": "14.2 GB",
        "description": "Orta seviye, 16 GB'ta sınırda",
    },
    "12b": {
        "name": "google/gemma-4-12B",
        "params": "12B",
        "vram_bf16": "24+ GB",
        "description": "En güçlü, 16 GB'a sığmaz (quantization gerekir)",
    },
}
