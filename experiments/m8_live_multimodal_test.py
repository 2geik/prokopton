"""
M8 Live Test — Gerçek ses ve görsel dosyalarıyla multimodal pipeline testi.

distilgpt2 modelini yükler, gerçek WAV ses + PNG görsel oluşturur,
learn_audio / learn_image / learn_multimodal metotlarını test eder.
"""

import wave, struct, math, time
from pathlib import Path
import numpy as np
from PIL import Image as PILImage


def create_test_audio(path: str, freq: float = 440.0, duration: float = 1.0,
                       sample_rate: int = 16000):
    """Create a WAV file with a sine wave + amplitude envelope (voice-like)."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Voice-like: fundamental + 2 harmonics + fade in/out
    signal = (
        0.6 * np.sin(2 * math.pi * freq * t) +
        0.25 * np.sin(2 * math.pi * freq * 2 * t) +
        0.15 * np.sin(2 * math.pi * freq * 3 * t)
    )

    # ADSR envelope (attack, sustain, release)
    attack = int(0.05 * n_samples)
    release = int(0.1 * n_samples)
    env = np.ones(n_samples)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    signal = signal * env * 0.8  # prevent clipping

    # Write WAV
    samples = (signal * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return path


def create_test_image(path: str, size: int = 224):
    """Create a simple colorful test image (red circle on blue background)."""
    from PIL import ImageDraw

    img = PILImage.new('RGB', (size, size), color=(30, 60, 120))  # dark blue bg
    draw = ImageDraw.Draw(img)

    # Red circle
    margin = 40
    draw.ellipse([margin, margin, size - margin, size - margin],
                 fill=(200, 40, 40), outline=(255, 80, 80), width=3)

    # Yellow square in center
    sq = size // 4
    draw.rectangle([size // 2 - sq, size // 2 - sq,
                    size // 2 + sq, size // 2 + sq],
                   fill=(240, 200, 40), outline=(255, 220, 80), width=2)

    img.save(path, 'PNG')
    return path


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prokopton.core import Prokopton, ProkoptonConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}  (ROCm: {torch.version.hip if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    # ── Create test assets ──
    audio_path = create_test_audio("/tmp/prokopton_test_audio.wav",
                                    freq=523.25, duration=1.5)  # C5 note, 1.5s
    print(f"🎵 Test audio created: {audio_path} (1.5s, 523Hz C5, 3 harmonics + envelope)")

    image_path = create_test_image("/tmp/prokopton_test_image.png", size=224)
    print(f"🖼️  Test image created: {image_path} (224×224, red circle + yellow square)")

    # ── Load model ──
    print(f"\n📥 Loading distilgpt2...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    print(f"   Model: {type(model).__name__}, hidden_size={model.config.hidden_size}")
    print(f"   Vocab: {model.config.vocab_size}")

    # ── Setup Prokopton ──
    cfg = ProkoptonConfig(
        ttt_n_layers=3,
        ttt_lr=1e-3,
        cms_rank=4,
        auto_save_every=0,  # no auto-save during test
    )
    prok = Prokopton(model, tokenizer, cfg)
    print(f"   TTT layers: {len(prok.fast_weights)}")
    for i, cms in enumerate(prok.cms_adapters):
        print(f"   CMS[{i}]: freq={cms.frequency}, rank={cms.rank}")
    print(f"   Vision tokenizer: output_dim={prok.vision_tokenizer.output_proj.out_features}")
    print(f"   Audio tokenizer: output_dim={prok.audio_tokenizer.output_proj.out_features}")

    # ════════════════════════════════════════════════════════════
    # TEST 1: learn_audio
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("TEST 1: learn_audio() — gerçek WAV dosyası")
    print("=" * 65)

    # Read WAV file using built-in wave module
    with wave.open(audio_path, 'r') as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n_frames)
        waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

    print(f"   Waveform: {waveform.shape}, sr={sr}, min={waveform.min():.3f}, max={waveform.max():.3f}")
    waveform_tensor = torch.from_numpy(waveform).float()

    t0 = time.time()
    result = prok.learn_audio(waveform_tensor)
    t_audio = time.time() - t0

    print(f"   ✅ learn_audio() completed in {t_audio*1000:.0f}ms")
    print(f"   Loss: {result['loss']:.4f}  Surprise: {result['surprise']:.4f}  Step: {result['step']}")
    print(f"   Stats: updates={prok.stats['updates']}, ΔW={prok.stats['weight_change']:.4f}")

    # ════════════════════════════════════════════════════════════
    # TEST 2: learn_image
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("TEST 2: learn_image() — gerçek PNG görsel")
    print("=" * 65)

    from PIL import Image as PILImage
    pil_img = PILImage.open(image_path).convert('RGB')
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    # To tensor: [H, W, C] → [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    print(f"   Image tensor: {img_tensor.shape}, range=[{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    t0 = time.time()
    result = prok.learn_image(img_tensor)
    t_image = time.time() - t0

    print(f"   ✅ learn_image() completed in {t_image*1000:.0f}ms")
    print(f"   Loss: {result['loss']:.4f}  Surprise: {result['surprise']:.4f}  Step: {result['step']}")
    print(f"   Stats: updates={prok.stats['updates']}, ΔW={prok.stats['weight_change']:.4f}")

    # ════════════════════════════════════════════════════════════
    # TEST 3: learn_multimodal — hepsi bir arada
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("TEST 3: learn_multimodal() — metin + görsel + ses aynı anda")
    print("=" * 65)

    t0 = time.time()
    result = prok.learn_multimodal({
        "text": "A red circle and yellow square on blue background, with a C5 musical note playing.",
        "image": img_tensor,
        "audio": waveform_tensor,
    })
    t_multi = time.time() - t0

    print(f"   ✅ learn_multimodal() completed in {t_multi*1000:.0f}ms")
    print(f"   Loss: {result['loss']:.4f}  Surprise: {result['surprise']:.4f}  Step: {result['step']}")
    print(f"   Stats: updates={prok.stats['updates']}, ΔW={prok.stats['weight_change']:.4f}")

    # ════════════════════════════════════════════════════════════
    # TEST 4: Save & verify
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("TEST 4: Incremental save & load")
    print("=" * 65)

    prok.save("/tmp/prokopton_mm_test", incremental=True)
    prok.reset()
    print(f"   After reset: step={prok.step_counter}, ΔW={prok.stats['weight_change']:.4f}")

    loaded = prok.load("/tmp/prokopton_mm_test")
    print(f"   Loaded: {loaded}, step={prok.step_counter}")

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("🏁 ALL MULTIMODAL TESTS COMPLETED")
    print("=" * 65)
    print(f"   learn_audio:       ✅ {t_audio*1000:.0f}ms")
    print(f"   learn_image:       ✅ {t_image*1000:.0f}ms")
    print(f"   learn_multimodal:  ✅ {t_multi*1000:.0f}ms")
    print(f"   Total steps:       {prok.step_counter}")
    print(f"   Total ΔW:          {prok.stats['weight_change']:.4f}")
    print(f"   Per-layer eff LRs: ", end="")
    for i in range(len(prok.fast_weights)):
        key = f"layer_{i}_eff_lr"
        if key in prok.stats:
            print(f"[L{i}:{prok.stats[key]}] ", end="")
    print()

    # Cleanup
    import os
    os.remove(audio_path)
    os.remove(image_path)
    print(f"\n   🧹 Cleaned up temp files")


if __name__ == "__main__":
    main()
