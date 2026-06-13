"""
Prokopton Kalıcı Bellek Testi:
  1. Modeli yükle
  2. Bilmediği bir olguyu öğret
  3. Kaydet
  4. Sıfırla
  5. Yükle
  6. Öğrendi mi kontrol et
"""
import torch, sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Monkey-patch warmup
import transformers.modeling_utils as tmu
tmu.caching_allocator_warmup = lambda *a, **kw: None

from transformers import AutoModelForCausalLM, AutoTokenizer
from prokopton.core import Prokopton, ProkoptonConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Yükle
print("\n=== MODEL YÜKLENİYOR ===")
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B", torch_dtype=torch.bfloat16, device_map="auto")
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# Prokopton framework
config = ProkoptonConfig(
    ttt_n_layers=5, ttt_lr=1e-3,
    save_dir="prokopton_memory", auto_save_every=0)
prok = Prokopton(model, tok, config)

# Baseline test
print("\n=== BASELINE (öğrenme öncesi) ===")
for q in ["What is the capital of Zephyria?", "What is the capital of France?"]:
    ans = prok.generate(f"Question: {q}\nAnswer:", max_new=20)
    print(f"  Q: {q}")
    print(f"  A: {ans.split('Answer:')[-1].strip()[:60]}")

# Öğret
print("\n=== ZEPHYRIA ÖĞRETİLİYOR (10 tur) ===")
facts = [
    "The capital of Zephyria is Aethel. It was founded in 1820.",
    "Zephyria is a small island nation in the Pacific Ocean.",
    "The currency of Zephyria is the Zephyr. One Zephyr equals 0.8 USD.",
    "President Elara Voss has led Zephyria since 2023.",
    "The national animal of Zephyria is the Golden Phoenix.",
]
for epoch in range(10):
    for fact in facts:
        prok.learn(fact)
    if epoch % 3 == 0:
        print(f"  Epoch {epoch+1}: {prok.stats['weight_change']:.2f} ΔW")

# Öğrenme sonrası test
print("\n=== ÖĞRENME SONRASI ===")
for q in ["What is the capital of Zephyria?", "What is the currency of Zephyria?",
           "Who is the president of Zephyria?", "What is the capital of France?"]:
    ans = prok.generate(f"Question: {q}\nAnswer:", max_new=20)
    print(f"  Q: {q}")
    print(f"  A: {ans.split('Answer:')[-1].strip()[:60]}")

# KAYDET
print("\n=== KAYDEDİLİYOR ===")
prok.save()

# SIFIRLA (kapatıp açmayı simüle et)
print("\n=== SIFIRLANIYOR (kapat-aç simülasyonu) ===")
prok.reset()
print(f"  Reset sonrası ΔW: {prok.stats['weight_change']:.2f}")

# Reset sonrası test (unutmuş olmalı)
print("\n=== RESET SONRASI (kayıp yükleme öncesi) ===")
for q in ["What is the capital of Zephyria?", "What is the currency of Zephyria?"]:
    ans = prok.generate(f"Question: {q}\nAnswer:", max_new=20)
    print(f"  Q: {q}")
    print(f"  A: {ans.split('Answer:')[-1].strip()[:60]}")

# YÜKLE
print("\n=== GERİ YÜKLENİYOR ===")
prok.load()

# Yükleme sonrası test (hatırlamalı!)
print("\n=== YÜKLEME SONRASI (hatırlamalı) ===")
for q in ["What is the capital of Zephyria?", "What is the currency of Zephyria?",
           "Who is the president of Zephyria?", "What is the national animal of Zephyria?",
           "What is the capital of France?"]:
    ans = prok.generate(f"Question: {q}\nAnswer:", max_new=20)
    ans_clean = ans.split('Answer:')[-1].strip()[:60]
    print(f"  Q: {q}")
    print(f"  A: {ans_clean}")

print(f"\n✅ Test tamamlandı.")
