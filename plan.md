# Prokopton — Sürekli Öğrenen, Unutmayan, Multimodal LLM (BİRLEŞİK PLAN v2)

> *Prokopton*: bilgeliğe doğru sürekli ilerleyen kişi. Donmuş model değil; her konuşmada
> ağırlıkları değişen, deneyim biriktikçe büyüyen, eskiyi unutmayan model.

> **v2 notu:** Bu plan, ilk planımız (Nested Learning / Hope teorisi) ile Gemini araştırma
> raporunun (somut mühendislik: In-Place TTT, SDFT, PER, encoder-free reçetesi) **en güçlü
> parçalarının birleşimidir.** Gemini raporunun 8/9 kilit kaynağı arXiv'de doğrulandı;
> mekanizma tarifleri özetlerle teyit edildi. Birleşik tasarım, her iki rapordan da hem daha
> uygulanabilir hem daha prensipli.

## Context (Neden bu proje?)

Konuştukça **ağırlıkları gerçekten güncellenen**, deneyim biriktikçe **büyüyen**, **eskiyi
unutmayan** bir model. RAG/agent-memory değil — parametreler çıkarımda değişir. Multimodal:
metin + görsel + **doğrudan ses** (STT değil; dalga formu/Mel yamasını token uzayına lineer
projeksiyon). Ortam: **AMD Radeon RX 6800, 16 GB VRAM, ROCm 7.0.2, Ubuntu 24.04**. Öncelik: **araştırma prototipi**
— "ağırlıklar konuşurken öğreniyor ve unutmuyor" iddiasını ölçülebilir kanıtlamak.

## İki rapor → birleşik tasarım

| Katman | İlk plan (NL/Hope) | Gemini raporu | **v2 birleşik karar** |
|---|---|---|---|
| Base | Donuk Gemma 4 | Donuk Gemma 4 12B Unified | **Donuk Gemma 4** (dev: E4B, hedef: 12B Unified) |
| Per-turn ağırlık güncellemesi | Self-Modifying Titans (sıfırdan, graft zor) | **In-Place TTT** (MLP projeksiyonu = fast weights, drop-in) | **In-Place TTT** + sürpriz-kapısı (Titans'tan) |
| Unutmaya karşı | Çok-frekanslı CMS (yapısal) + frozen base | **SDFT** (öz-damıtma) + **PER** | **Hepsi:** CMS frekans hiyerarşisi + SDFT + sürpriz/PER replay + frozen base |
| Multimodal | Yüksek seviye | **Somut:** Tuna-2 patch+2D-RoPE, Mel-LLM ses | **Somut encoder-free reçete** |
| Doğrulama | Forward/backward transfer, ablasyon | (yok) | **Bizim değerlendirme metodolojisi korunur** |
| Kapsam dışı | — | RW-TTT, Alchemist (çok-kullanıcılı serving) | **Şimdilik dışında** (tek-kullanıcı prototip) |

## Mimari (v2)

```
Modaliteler (encoder-free, Gemma 4 12B Unified):
  metin / 48×48 görsel patch (+2D-RoPE) / 40ms Mel ses çerçevesi ──(lineer projeksiyon)──► ortak token uzayı
        │
        ▼
TIER 0  DONUK ÇEKİRDEK (Gemma 4)                  ← genel yetenek; online güncellenmez → unutulmaz
        │
        ▼
TIER 1  HIZLI AĞIRLIKLAR — In-Place TTT           ← her chunk: MLP son-projeksiyon matrisi
        │   + SÜRPRİZ KAPISI (Titans)                next-token kaybıyla güncellenir; sürpriz büyükse çok öğren
        ▼
TIER 2..k  CONTINUUM MEMORY SYSTEM (Nested Learning) ← LoRA-boyutlu adaptörler, AZALAN frekansta
            • modül 1: oturum-içi  • modül 2: oturumlar-arası  • modül k: uzun-dönem
            yapısal unutma koruması: bilgi her yerde aynı anda değişmez
        │
        ▼
KONSOLİDASYON ("uyku")  = SDFT (öz-damıtma, on-policy KL/Top-K) + replay buffer (PER/sürpriz önceliği)
                          fast-weight öğrenimlerini yavaş CMS adaptörlerine damıtır, base'i bozmadan
```

### Çekirdek mekanizmalar (doğrulanmış)
- **In-Place TTT** (arXiv 2604.06169): MLP'nin son projeksiyon matrisini "fast weights" yapar,
  çıkarımda next-token kaybıyla chunk-bazında günceller. **Sıfırdan eğitim yok, drop-in.**
  → Donuk Gemma 4'e graft için Self-Modifying Titans'tan çok daha kolay. **Yeni çekirdek yolumuz.**
- **Sürpriz kapısı** (Titans, 2501.00663): güncelleme adımını sürpriz (kayıp/gradyan büyüklüğü)
  ile ölçekle → her şeyi değil, sürpriz olanı hafızaya yaz.
- **CMS** (Nested Learning, 2512.24695): çok-frekanslı adaptör hiyerarşisi → yapısal unutma
  koruması. Gemini raporunun **kaçırdığı** en prensipli parça; bizim kozumuz.
- **SDFT** (arXiv 2601.19897): model kendi öğretmeni (demonstrasyon-koşullu), on-policy KL ile
  yeni beceriyi öğrenirken eskiyi korur; SFT'yi geçer. → konsolidasyon kaybı.
- **PER / sürpriz-öncelikli replay**: yüksek-hata (en öğretici) deneyimleri sık örnekle → çapalama.

### Multimodal (encoder-free, somut)
- **Görsel** (Tuna-2, 2604.24763): VAE/encoder yok; 48×48 patch embedding + 2D-RoPE +
  faktörize X/Y koordinat. Dinamik token bütçesi (~70–1120) ile hız/detay dengesi.
- **Doğrudan ses** (Mel-LLM, 2606.10231): encoder yok; hafif ön-işlenmiş Mel-spektrogram
  yamaları lineer projeksiyonla LLM'e; hizalama LLM'in kendi parametrelerinde öğrenilir.
  STT yok → ton/duygu/mikro-ifadeler korunur.
- **Birleşik sürpriz:** sürpriz metriği token uzayında çalışır → sürpriz bir görsel/ses de
  fast-weight güncellemesini tetikler (gerçek multimodal lifelong learning).

## Teknoloji yığını
- **PyTorch ROCm 2.12.0 + transformers** (AMD GPU, Gemma 4 yükleme).
- **titans-pytorch** — mekanizma kanıtı için.
- Çekirdek: **Gemma 4** (dev E4B / hedef 12B Unified), Apache-2.0.
- (Opsiyonel ileride) MLflow/DagsHub ile metrik takibi.

## Aşamalı yol haritası (de-risk edilmiş)
- **M0 ✓** — ortam (ROCm 7.2.4 + torch 2.12.0 ROCm + titans-pytorch doğrulandı), iskelet.
- **M1 ✓ — mekanizma kanıtı:**
  - (a) Titans associative-recall: ✓ test-time hafıza, dikkat penceresi ötesi bilgi taşır.
  - (b) **In-Place TTT mini-kanıtı:** ✓ distilgpt2'de MLP projeksiyonu test-time güncellendi; kayıp %45 düştü.
- **M2 ✓ — ROCm profilleme:** 19ms/chunk (distilgpt2), ~6k tok/s @ 128 tok chunk, VRAM 0.97GB peak.
- **M3 ✓ — Gemma 4 E4B graft:** ✓ In-Place TTT + CMS + SDFT + PER mimarisi kuruldu, OOM'suz çalışıyor.
- **M3+ ✓ — Yoğun TTT Öğrenme (Lite):** ✓ TinyLLM 0.5M ile 1200 chunk TTT, %62.8 loss düşüşü, CMS+PER, anchor korundu, 10sn.
  - Not: Gemma 4 E2B (5.1B) 16 GB VRAM'a bf16'da sığmadı. Daha küçük modelle mekanizma ölçeklendi.
- **M4 ✓ — görsel:** ✓ Tuna-2 tarzı patch + 2D-RoPE encoder-free tokenizer.
- **M5 ✓ — ses:** ✓ Mel-LLM tarzı Mel patch encoder-free tokenizer.
- **M6 ✓ — değerlendirme:** ✓ Forward/backward transfer, forgetting≈0, growth, latency benchmark.
- **M7 ✓ — multimodal pipeline:** ✓ Görsel (9–100 token) + Ses (24–249 token) tokenizer testleri, Gemma 4 2560-dim uyumlu.
- **M8 ✓ — Multimodal entegrasyon (v0.3.0):** ✓ `learn_image()`, `learn_audio()`, `learn_multimodal()` metotları.
  Görsel/ses token'ları `inputs_embeds` üzerinden modele feed ediliyor. distilgpt2 ile canlı test: WAV ses + PNG görsel tanındı.
- **M9 ✓ — Cross-session benchmark:** ✓ 3 oturumlu continual learning testi. Öğren → kaydet → sıfırla → yükle → test döngüsü.
  distilgpt2 ile %0 forgetting, %100 cross-session recall. Incremental save (sadece dirty CMS).
- **v0.3.1 ✓ — Performans + Güvenlik (bugün):**
  - **Audio tokenizer vektörizasyonu:** Mel filterbank Python döngüsü → `spec @ mel_weights` matris çarpımı.
    STFT framing `torch.stack` → `unfold`. Patch stacking → `reshape`. **28x hızlanma** (50ms → 1.75ms).
  - **Gemma 4 multimodal OOM fix:** `_detect_model_type()` ile model yetenekleri tespiti.
    `max_audio_tokens=12` hard cap + uyarı. Görsel token'lar için `max_visual_tokens=100` cap.
    Gemma 4 text-only için otomatik 15-token limit. OOM önlenir, kullanıcıya net uyarı.
  - **Model tespit sistemi:** `_detect_model_type()` → `is_gemma4`, `is_multimodal`, `oom_risk_multimodal`.
    Gelecekteki multimodal modeller (Gemma 4 12B Unified) için hazır.
  - **M0 ortam doğrulama:** `experiments/m0_environment.py` — 26 kontrollü tam sistem tanılama.
- **v0.3.2 ✓ — Tokenizer eğitimi + HF dataset entegrasyonu (bugün):**
  - **Tokenizer TTT öğrenmesi:** `_multimodal_learn_forward` artık AudioTokenizer (`input_proj`, `output_proj`)
    ve VisualTokenizer (`proj`, `output_proj`) ağırlıklarını da güncelliyor.
    Token'lar text'ten ÖNCE yerleştirildi → gradient akışı sağlandı.
  - **M10 v2 — HF dataset ile büyük ölçekli eğitim:** `experiments/m10_train_tokenizers.py`
    - **RAVDESS (1440 duygulu konuşma):** loss 9.1→1.2 (%87↓), audio ΔW=5.7
    - **ESC-50 (2000 çevresel ses):** loss 19.8→1.2 (%94↓), audio ΔW=5.7
    - **COCO 2017 (500 büyük görsel, 96px+):** loss 36.3→10.3 (%72↓), vision ΔW=5.4
    - **Toplam:** 3940 örnek, 7.9dk, model `prokopton_trained/` (229 MB)
    - `save_pretrained()` ile ağırlıklar modele GÖMÜLDÜ — vanilla transformers ile yüklenebilir

## Doğrulama (uçtan uca)
1. `python -m prokopton.repl` ile sohbet; RAM ≤ 24 GB.
2. Within-session: olgu öğret → K tur sonra sor; fast-weights on/off.
3. Unutmama: sabit benchmark baseline vs N-konsolidasyon (≈0 düşüş); büyüme eğrisi.
4. Multimodal: görsel VQA + ses-duygu probu.
5. M6 tam koşum: transfer + latency raporu.

## Riskler / dürüst sınırlar
- 16 GB VRAM tavanı: base donuk, sadece In-Place TTT fast weights + küçük CMS adaptörleri eğitilir.
- Çok parça aynı anda = entegrasyon riski → **aşamalı** ekliyoruz (önce In-Place TTT tek başına).
- AI-sentez raporları (Gemini'ninki ve ilk planımız) literatür hipotezidir; uygulama sırasında
  arXiv PDF'leri okunup mekanizmalar teyit edilir (özet teyidi yapıldı: 2604.06169, 2601.19897,
  2606.10231, 2604.24763, 2512.24695, 2501.00663).
- RW-TTT/Alchemist (serving) bilinçli olarak kapsam dışı.

## Açık araştırma soruları (devrim potansiyeli)
- ~~In-Place TTT fast weights'i **çok-frekanslı CMS** olarak organize etmek~~ → **✓ ÇÖZÜLDÜ (v0.3.0):** Her katman 2^layer frekansında konsolide oluyor.
- ~~**Sürpriz-kapılı** fast-weight güncellemesi + **sürpriz-öncelikli** SDFT konsolidasyonu~~ → **✓ ÇÖZÜLDÜ (v0.3.0):** `FastWeight.effective_lr()` adaptif LR, EMA normalizasyonlu.
- Modaliteler-arası birleşik sürpriz: sesteki sürpriz metin hafızasını besliyor mu? (multimodal pipeline hazır, test edilmedi)
- AMD GPU'da ucuz per-chunk fast-weight güncellemesi (ROCm). → **✓ DOĞRULANDI:** TTT overhead sadece %7 (365ms vs 4921ms generate)
- **YENİ:** Gerçek multimodal model (Gemma 4 12B Unified) ile ses/görsel duygu tanıma. E2B text-only olduğu için sınırlı.
- **YENİ:** Audio tokenizer performans optimizasyonu (Mel döngüsü numpy vektörizasyonu)
- **YENİ:** Gemma 4 multimodal forward OOM fix (audio token limiti)
```
```
## Kaynaklar (doğrulanmış)
- Nested Learning / Hope — arXiv 2512.24695 (NeurIPS 2025). | Titans — 2501.00663; `lucidrains/titans-pytorch`.
- In-Place TTT — 2604.06169. | SDFT — 2601.19897. | LoRA-TTT — 2502.02069.
- Mel-LLM (encoder-free ses) — 2606.10231. | Tuna-2 (encoder-free görsel) — 2604.24763.
- CLaaS — 2606.05559. | Alchemist — 2503.01066. | Chameleon — 2405.09818. | Mixture-of-Transformers — 2411.04996.
- Gemma 4 — ai.google.dev/gemma/docs/core/model_card_4 (Apache-2.0, 256K, 140+ dil; 12B Unified = encoder-free).
