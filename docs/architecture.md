# Prokopton — Mimari Notları (v3 — Tamamlanmış Adaptasyon)

ROCm 7.2.4 + AMD Radeon RX 6800 + PyTorch 2.12.0 üzerinde çalışan,
Apple M4 Pro → AMD ROCm adaptasyonu tamamlanmış sürüm.

---

## 1. Sistem

| Bileşen | Değer |
|---|---|
| İşletim Sistemi | Ubuntu 24.04 |
| GPU | AMD Radeon RX 6800 (gfx1030) |
| VRAM | 16 GB |
| ROCm | 7.2.4 |
| HIP | 7.2.53211 |
| PyTorch | 2.12.0+rocm7.2 |
| Python | 3.11.15 |
| Paket Yöneticisi | uv |

## 2. Adaptasyon Değişiklikleri

| Orijinal (M4 Pro) | Yeni (ROCm) |
|---|---|
| MLX + mlx-lm | PyTorch ROCm + transformers |
| PyTorch-MPS | PyTorch ROCm (CUDA API) |
| Apple GPU 24 GB unified | AMD RX 6800 16 GB discrete |
| `torch.backends.mps` | `torch.cuda` |
| Gemma 4 MLX yükleme | Gemma 4 HuggingFace yükleme |

## 3. Proje Yapısı

```
prokopton/
  core/__init__.py       ← Prokopton, FastWeight, CMS, PER, tokenizer'lar
  models/__init__.py     ← Model fabrika + registry (E2B, E4B, 12B)
  eval/__init__.py       ← CLBenchmark, değerlendirme metrikleri
  repl.py                ← Etkileşimli sohbet + öğrenme döngüsü
  
experiments/
  m1_associative_recall.py      ← M1(a): Titans hafıza kanıtı
  m1b_inplace_ttt.py            ← M1(b): In-Place TTT mini-kanıtı
  m2_rocm_profile.py            ← M2: ROCm profilleme
  m3_gemma4_graft.py            ← M3: Gemma 4 E4B graft
  m3plus_intensive_ttt.py       ← M3+: Yoğun TTT öğrenme deneyi
  m4_visual.py                  ← M4: Tuna-2 görsel tokenizer
  m5_audio.py                   ← M5: Mel-LLM ses tokenizer
  m6_evaluation.py              ← M6: Değerlendirme benchmark
  m7_multimodal.py              ← M7: Multimodal entegrasyon
  
  runs/                         ← Deney çıktıları (JSON/log)
```

## 4. Tüm Aşamalar — Tamamlandı ✓

| Aşama | Sonuç |
|---|---|
| M0 | ROCm 7.2.4 + PyTorch 2.12.0 + RX 6800 |
| M1(a) | Titans associative recall GPU'da çalışıyor |
| M1(b) | In-Place TTT: loss %45 düştü (3.94→2.15) |
| M2 | 19ms/chunk, 6k tok/s, 0.97 GB VRAM |
| M3 | Gemma 4 E4B + TTT + CMS + SDFT + PER |
| M4 | Tuna-2 2D-RoPE encoder-free (25-196 tok) |
| M5 | Mel-LLM Mel patch (25 tok/s) |
| M6 | Forgetting≈0, anchor korundu |
| M7 | Multimodal pipeline entegre |

## 5. Bilinen Sınırlamalar

- **VRAM:** 16 GB ile E4B bf16 sınırda (~14.2 GB). E2B önerilir (~9.5 GB).
- **bitsandbytes:** ROCm'da 8-bit/4-bit quantization çalışmıyor (segfault). bf16 kullan.
- **TTT lr:** Çok yüksek lr (>1e-2) Gemma 4'te output çökmesine yol açıyor. 1e-3—1e-4 güvenli.
- **Token hızı:** E2B'de ~100 tok/s üretim, TTT ile ~50-100 chunk/dk öğrenme.

## 6. Sonraki Adımlar

- Daha uzun TTT eğitimi (binlerce chunk)
- Çok-oturumlu öğrenme (model kapanıp açılsa da bilgi korunsun)
- Gerçek görsel/ses girdisi ile multimodal test
- Gemma 4 12B için quantization çözümü
