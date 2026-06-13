"""
M7 — Multimodal Entegrasyon Deneyi

Gemma 4 E2B + Görsel Tokenizer + Ses Tokenizer + TTT.
Tek pipeline'da metin, görsel ve ses işleme.

Kullanım:
  .venv/bin/python experiments/m7_multimodal.py
"""
import torch, math, time, json
from pathlib import Path
from prokopton.core import (
    Prokopton, ProkoptonConfig, 
    VisualTokenizer, AudioTokenizer
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_multimodal_pipeline():
    """Multimodal pipeline entegrasyon testi."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Prokopton M7 — Multimodal Pipeline Test")
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print("=" * 60)
    
    # ====== Görsel Tokenizer Testi ======
    print("\n--- Görsel Tokenizer ---")
    vision = VisualTokenizer(patch_size=48, embed_dim=1024, output_dim=2560)
    
    # Farklı çözünürlüklerde test
    for size in [112, 224, 448]:
        img = torch.randn(1, 3, size, size)
        tokens, info = vision(img)
        print(f"  {size}×{size} → {info['num_tokens']} token (grid: {info['grid']})")
    
    # ====== Ses Tokenizer Testi ======
    print("\n--- Ses Tokenizer ---")
    audio = AudioTokenizer(sample_rate=16000, n_mels=80, patch_frames=4, 
                           embed_dim=1024, output_dim=2560)
    
    for dur in [1.0, 3.0, 10.0]:
        t = torch.linspace(0, dur, int(16000 * dur))
        waveform = torch.sin(2 * math.pi * 440 * t)
        tokens, info = audio(waveform)
        print(f"  {dur:.1f}s → {info[0]['num_tokens']} token ({info[0]['duration_ms']:.0f}ms)")
    
    # ====== Gemma 4 Entegrasyon ======
    print("\n--- Gemma 4 E2B Entegrasyon ---")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    config = ProkoptonConfig(ttt_n_layers=3, ttt_lr=5e-4)
    prokopton = Prokopton(model, tokenizer, config)
    
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM: {vram:.1f} GB")
    print(f"  TTT layers: {len(prokopton.fast_weights)}")
    print(f"  Vision tokenizer: ✓")
    print(f"  Audio tokenizer: ✓")
    
    # ====== Metin + TTT Testi ======
    print("\n--- Metin + TTT Öğrenme ---")
    
    # Bir olgu öğret
    fact = "Zephyria's capital is Aethel. The president is Elara Voss."
    info = prokopton.learn(fact)
    print(f"  Öğrenme: loss={info['loss']:.4f} surprise={info['surprise']:.4f}")
    
    # Birkaç kez daha
    for i in range(3):
        info = prokopton.learn(fact)
    
    # Test
    response = prokopton.generate(
        "Question: What is the capital of Zephyria?\nAnswer:", max_new=32
    )
    print(f"  Test: {response.split('Answer:')[-1].strip()[:80]}")
    
    s = prokopton.stats
    print(f"\n  Stats: {s['steps']} step, {s['updates']} updates, ΔW={s['weight_change']:.4f}")
    
    # ====== Multimodal Embedding Boyut Uyumu ======
    print("\n--- Embedding Boyut Kontrolü ---")
    img_tokens, _ = vision(torch.randn(1, 3, 224, 224))
    wf = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, 16000))
    aud_tokens, _ = audio(wf)
    
    # Gemma 4 hidden size ile uyum kontrolü
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 2560
    print(f"  Gemma 4 hidden_size: {hidden_size}")
    print(f"  Görsel token dim: {img_tokens.shape[-1]}")
    print(f"  Ses token dim: {aud_tokens[0].shape[-1]}")
    
    if img_tokens.shape[-1] == hidden_size and aud_tokens[0].shape[-1] == hidden_size:
        print("  ✓ Tüm embedding boyutları uyumlu")
    else:
        print("  ⚠ Boyut uyuşmazlığı — output_dim ayarlanmalı")
    
    # ====== Pipeline Özeti ======
    print("\n" + "=" * 60)
    print("MULTIMODAL PIPELINE ÖZETİ")
    print("=" * 60)
    print(f"""
  Metin:     Tokenizer → Gemma 4 embedding
  Görsel:    PatchEmbed2D → 2D-RoPE → Lineer → Gemma 4 embedding
  Ses:       MelPatch → Lineer + PE → Lineer → Gemma 4 embedding
  
  TTT:       {len(prokopton.fast_weights)} katman fast-weight
  CMS:       {len(prokopton.cms_adapters)} adaptör
  PER:       {config.per_capacity} kapasiteli buffer
  
  Hepsi aynı embedding uzayında → ortak sürpriz metriği → 
  görsel/ses de fast-weight güncellemesini tetikler.
""")
    
    return prokopton


if __name__ == "__main__":
    test_multimodal_pipeline()
