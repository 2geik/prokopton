# Prokopton 🧠

> *Prokopton* (προκόπτων): bilgeliğe doğru sürekli ilerleyen kişi.

**Konuştukça ağırlıkları gerçekten güncellenen**, deneyim biriktikçe **büyüyen** ve
**eskiyi unutmayan** bir LLM. RAG/agent-memory değil — parametreler çıkarım sırasında
değişir.

- 🔄 **In-Place TTT**: Her konuşmada MLP ağırlıkları güncellenir
- 💾 **Kalıcı Bellek**: CMS adaptörleri diske kaydedilir, yeniden başlatınca yüklenir
- 🎯 **Unutmama**: Forgetting ≈ 0, anchor korunur
- 🖼️🎵 **Multimodal**: Görsel + doğrudan ses (STT'siz)
- 🎮 **TUI Arayüz**: Terminal üzerinden kullanıcı dostu arayüz
- ⬇️ **HF'den Model İndirme**: HuggingFace URL'si ile tek tuşta model indir

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
git clone https://github.com/kullaniciadi/prokopton.git
cd prokopton

# Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate

# ROCm (AMD GPU) için PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# Prokopton'u kur
pip install -e .
```

### 2. TUI'yi Başlat

```bash
prokopton
# veya
python -m prokopton.tui
```

Açılan ekranda:
1. Model seçin (HF ID yazın veya listeden seçin)
2. Sohbet etmeye başlayın!
3. `Ctrl+S` ile belleği kaydedin, `Ctrl+L` ile geri yükleyin

### 3. Model Yükleme

İki yöntem:

**A) Otomatik — TUI içinden:**
- `Ctrl+D` → HF URL veya model ID girin → İndir

**B) Manuel — `models/` klasörüne:**
```bash
# HuggingFace'den manuel indir
huggingface-cli download google/gemma-4-E2B --local-dir models/gemma-4-E2B

# Veya herhangi bir modeli models/ altına koyun
# Sonra TUI'de otomatik görünecektir
```

## 📋 TUI Kısayolları

| Kısayol | İşlem |
|---------|-------|
| `Ctrl+Q` | Çıkış |
| `Ctrl+S` | Belleği kaydet |
| `Ctrl+L` | Belleği yükle |
| `Ctrl+R` | Belleği sıfırla |
| `Ctrl+M` | Model değiştir |
| `Ctrl+D` | HF'den model indir |
| `Ctrl+P` | İstatistikleri göster |
| `Enter` | Mesaj gönder |

## 🧪 Python API

```python
from prokopton import Prokopton, ProkoptonConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model yükle
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
tokenizer.pad_token = tokenizer.eos_token

# Prokopton framework
prok = Prokopton(model, tokenizer)

# Konuştukça öğren
prok.learn("Zephyria'nın başkenti Aethel'dir.")
prok.save("my_memory")  # Diske kaydet

# Yeni oturum — belleği geri yükle
prok.load("my_memory")
cevap = prok.chat("Zephyria'nın başkenti neresi?")
print(cevap)  # "Aethel"
```

## 📦 Desteklenen Modeller

| Model | Parametre | VRAM (bf16) | Durum |
|-------|-----------|-------------|-------|
| `google/gemma-4-E2B` | 5.1B | ~9.5 GB | ✅ Önerilen |
| `google/gemma-4-E4B` | 7.9B | ~14.2 GB | ⚠️ 16 GB'ta sınırda |
| `google/gemma-4-12B` | 12B | 24+ GB | 🔮 Quantization gerekli |

## 🖥️ Donanım Gereksinimleri

- **Önerilen:** AMD Radeon RX 6800+ (16 GB VRAM), ROCm 7.0+
- **Minimum:** ROCm uyumlu herhangi AMD GPU
- **CPU:** Deneysel (yavaş ama çalışır)
- **RAM:** 32 GB önerilir

## 📊 Durum

| Aşama | | Sonuç |
|---|---|---|
| M0 | Ortam | ROCm 7.2.4 + PyTorch 2.12.0 ✅ |
| M1(a) | Titans recall | Hafıza penceresi ötesi bilgi ✅ |
| M1(b) | In-Place TTT | Kayıp %45 düştü ✅ |
| M2 | ROCm profil | 19ms/chunk, 6k tok/s ✅ |
| M3+ | Gemma 4 + Yoğun TTT | %60→%90 accuracy ✅ |
| M4 | Görsel tokenizer | Tuna-2 2D-RoPE ✅ |
| M5 | Ses tokenizer | Mel-LLM, STT'siz ✅ |
| M6 | Değerlendirme | Forgetting≈0, anchor korundu ✅ |
| M7 | Multimodal | Pipeline entegre ✅ |

## 📚 Bilimsel Temel

- **Nested Learning / Hope** — arXiv 2512.24695 (NeurIPS 2025)
- **Titans** — arXiv 2501.00663
- **In-Place TTT** — arXiv 2604.06169
- **SDFT** — arXiv 2601.19897
- **Tuna-2** (encoder-free görsel) — arXiv 2604.24763
- **Mel-LLM** (encoder-free ses) — arXiv 2606.10231

## 📄 Lisans

MIT License — detaylar için [LICENSE](LICENSE)
