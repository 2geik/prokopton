"""
M10 — Tokenizer Training with HuggingFace Datasets
===================================================
ESC-50 (çevresel sesler) + Fashion-MNIST (caption'lı görseller) ile
AudioTokenizer ve VisualTokenizer projeksiyon katmanlarını eğit.

Her örnekte model: "Bu bir X sesi/görseli" text'ini tahmin etmeye çalışır.
Audio/visual token'lar text'ten ÖNCE geldiği için, tokenizer projeksiyonları
gradient alır ve anlamlı embedding'ler öğrenir.

Kullanım:
  .venv/bin/python experiments/m10_train_tokenizers.py
  .venv/bin/python experiments/m10_train_tokenizers.py --model distilgpt2 --audio-samples 200 --image-samples 500
  .venv/bin/python experiments/m10_train_tokenizers.py --save-trained  # ağırlıkları kaydet
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import datasets

from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.backends import detect_backend, load_model, apply_backend_patches


# ═══════════════════════════════════════════════════════════════
# Audio preprocessing
# ═══════════════════════════════════════════════════════════════

def preprocess_audio(audio_dict_or_bytes, target_sr=16000, max_duration=1.0):
    """Resample audio to target sample rate, trim/pad to max_duration, return tensor.

    Handles both decoded audio dicts and raw bytes (fallback for torchcodec issues).
    """
    try:
        import librosa
        HAS_LIBROSA = True
    except ImportError:
        HAS_LIBROSA = False

    # Case 1: Already decoded audio dict from datasets
    if isinstance(audio_dict_or_bytes, dict) and "array" in audio_dict_or_bytes:
        arr = audio_dict_or_bytes["array"].astype("float32")
        sr = audio_dict_or_bytes["sampling_rate"]
    # Case 2: Raw bytes — decode manually with soundfile
    elif isinstance(audio_dict_or_bytes, dict) and "bytes" in audio_dict_or_bytes:
        import io
        import soundfile as sf
        raw = audio_dict_or_bytes["bytes"]
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # stereo → mono
    elif isinstance(audio_dict_or_bytes, bytes):
        import io
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(audio_dict_or_bytes), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
    else:
        raise ValueError(f"Beklenmeyen audio formatı: {type(audio_dict_or_bytes)}")

    # Resample to target_sr
    if sr != target_sr:
        if HAS_LIBROSA:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        else:
            # SciPy fallback
            from scipy.signal import resample
            new_len = int(len(arr) * target_sr / sr)
            arr = resample(arr, new_len)

    # Trim/pad
    max_samples = int(max_duration * target_sr)
    if len(arr) > max_samples:
        arr = arr[:max_samples]
    elif len(arr) < max_samples:
        arr = torch.nn.functional.pad(
            torch.tensor(arr), (0, max_samples - len(arr))).numpy()

    return torch.from_numpy(arr.copy()).float()


# ═══════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════

def preprocess_image(pil_image, min_size=48):
    """Resize PIL image to at least min_size, convert to normalized tensor.
    No torchvision needed — uses PIL + torch directly."""
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize so shortest side = min_size (preserve aspect ratio)
    w, h = pil_image.size
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h))

    # PIL → [0,255] uint8 → [0,1] float32 → [C, H, W]
    arr = torch.from_numpy(
        np.array(pil_image, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)  # HWC → CHW
    return arr


# ═══════════════════════════════════════════════════════════════
# ESC-50 LABELS (50 sınıf → insan-okunur isimler)
# ═══════════════════════════════════════════════════════════════

ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea waves", "crackling fire", "crickets", "chirping birds",
    "water drops", "wind", "pouring water", "toilet flush", "thunderstorm",
    "crying baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing teeth", "snoring", "drinking sipping",
    "door knock", "mouse click", "keyboard typing", "door creak", "can opening",
    "washing machine", "vacuum cleaner", "clock alarm", "clock tick", "glass breaking",
    "helicopter", "chainsaw", "siren", "car horn", "engine",
    "train", "church bells", "airplane", "fireworks", "hand saw",
]

FASHION_MNIST_LABELS = [
    "t-shirt or top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]


# ═══════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M10 — Tokenizer Training with HF Datasets")
    parser.add_argument("--model", default="distilgpt2", help="Base model")
    parser.add_argument("--audio-samples", type=int, default=200,
                        help="Number of audio samples to train on (max 2000)")
    parser.add_argument("--image-samples", type=int, default=500,
                        help="Number of image samples to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Passes over the data")
    parser.add_argument("--lr", type=float, default=1e-3, help="TTT learning rate")
    parser.add_argument("--save-trained", action="store_true",
                        help="Save trained tokenizer weights")
    parser.add_argument("--output-dir", default="trained_tokenizers",
                        help="Output directory for saved weights")
    args = parser.parse_args()

    n_audio = min(args.audio_samples, 2000)
    n_image = args.image_samples

    # ── Backend ──
    backend = detect_backend()
    apply_backend_patches(backend)
    device = backend.device
    print(f"🔧 {backend.description} | {device}")
    print(f"{'='*65}")

    # ── Load model ──
    print(f"\n📥 Loading {args.model}...")
    model, tokenizer = load_model(args.model, backend)

    config = ProkoptonConfig(
        ttt_lr=args.lr,
        ttt_n_layers=3,          # daha az katman = daha hızlı
        max_audio_tokens=8,      # ESC-50 sesleri 5sn → token'ları kıs
        max_visual_tokens=25,    # 28×28 → 1 patch → 1 token yeterli
        save_dir="m10_memory",
        auto_save_every=0,
    )
    prok = Prokopton(model, tokenizer, config)

    print(f"   TTT layers: {len(prok.fast_weights)}")
    for i, cms in enumerate(prok.cms_adapters):
        print(f"   CMS[{i}]: freq={cms.frequency}, rank={cms.rank}")

    # ── Snapshot tokenizer weights ──
    at = prok.audio_tokenizer
    vt = prok.vision_tokenizer
    snap = {
        "audio_out": at.output_proj.weight.clone(),
        "audio_in": at.input_proj.weight.clone(),
        "vision_out": vt.output_proj.weight.clone(),
        "vision_in": vt.proj.weight.clone(),
    }

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Audio Tokenizer Training (ESC-50)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHASE 1: Audio Tokenizer — ESC-50 ({n_audio} örnek)")
    print(f"{'─'*65}")

    try:
        # Load ESC-50 with raw audio bytes (torchcodec unavailable on ROCm)
        esc50 = load_dataset("ashraq/esc50", split="train")
        esc50 = esc50.shuffle(seed=42).select(range(n_audio))
        # Disable audio decoding — we'll decode manually with soundfile
        esc50 = esc50.cast_column("audio", datasets.Audio(decode=False))
    except Exception as e:
        print(f"   ⚠ ESC-50 yüklenemedi: {e}")
        print(f"   İnternet bağlantını kontrol et veya manuel indir:")
        print(f"   https://huggingface.co/datasets/ashraq/esc50")
        esc50 = None

    audio_losses = []
    audio_samples_done = 0

    if esc50 is not None:
        # Group by category for better label diversity
        for epoch in range(args.epochs):
            t0 = time.time()
            for idx, sample in enumerate(esc50):
                try:
                    wav = preprocess_audio(sample["audio"], target_sr=16000, max_duration=1.0)
                    # ESC-50 'category' is already a human-readable string (e.g., "dog", "rain")
                    label = sample["category"]
                    text = f"This is the sound of {label}."

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = prok.learn_multimodal({"text": text, "audio": wav})

                    audio_losses.append(result["loss"])
                    audio_samples_done += 1

                    if (idx + 1) % 50 == 0:
                        avg_loss = sum(audio_losses[-50:]) / min(50, len(audio_losses))
                        print(f"   [{idx+1:4d}/{n_audio}] "
                              f"loss={result['loss']:.4f} avg50={avg_loss:.4f} "
                              f"label={label[:25]}")

                except Exception as e:
                    print(f"   ⚠ Sample {idx} hatası: {e}")
                    continue

            elapsed = time.time() - t0
            print(f"   Epoch {epoch+1}: {audio_samples_done} örnek, {elapsed:.1f}s, "
                  f"son loss={audio_losses[-1]:.4f}" if audio_losses else "")

    # Check audio tokenizer progress
    adw_out = (at.output_proj.weight - snap["audio_out"]).norm().item()
    adw_in = (at.input_proj.weight - snap["audio_in"]).norm().item()
    print(f"\n   Audio tokenizer ΔW: out={adw_out:.4f}, in={adw_in:.4f}")
    if adw_out > 0.01:
        print(f"   ✅ Audio tokenizer öğreniyor!")
    else:
        print(f"   ⚠ Audio tokenizer değişmedi (ESC-50 yüklenemedi mi?)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Visual Tokenizer Training (Fashion-MNIST)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHASE 2: Visual Tokenizer — Fashion-MNIST ({n_image} örnek)")
    print(f"{'─'*65}")

    try:
        # Use standard Fashion-MNIST (PIL images + class labels)
        # The enriched version has URI strings instead of images
        fmnist = load_dataset("zalando-datasets/fashion_mnist", split="train")
        fmnist = fmnist.shuffle(seed=42).select(range(n_image))
        fmnist_has_captions = False
    except Exception as e:
        print(f"   ⚠ Fashion-MNIST yüklenemedi: {e}")
        fmnist = None

    image_losses = []
    image_samples_done = 0

    if fmnist is not None:
        for epoch in range(args.epochs):
            t0 = time.time()
            for idx, sample in enumerate(fmnist):
                try:
                    img = preprocess_image(sample["image"], min_size=48)
                    label_idx = sample["label"]

                    if fmnist_has_captions and "caption" in sample:
                        text = sample["caption"]
                    else:
                        label_name = FASHION_MNIST_LABELS[label_idx]
                        text = f"This is an image of a {label_name}."

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = prok.learn_multimodal({"text": text, "image": img})

                    image_losses.append(result["loss"])
                    image_samples_done += 1

                    if (idx + 1) % 100 == 0:
                        avg_loss = sum(image_losses[-100:]) / min(100, len(image_losses))
                        print(f"   [{idx+1:5d}/{n_image}] "
                              f"loss={result['loss']:.4f} avg100={avg_loss:.4f}")

                except Exception as e:
                    print(f"   ⚠ Sample {idx} hatası: {e}")
                    continue

            elapsed = time.time() - t0
            print(f"   Epoch {epoch+1}: {image_samples_done} örnek, {elapsed:.1f}s, "
                  f"son loss={image_losses[-1]:.4f}" if image_losses else "")

    # Check visual tokenizer progress
    vdw_out = (vt.output_proj.weight - snap["vision_out"]).norm().item()
    vdw_in = (vt.proj.weight - snap["vision_in"]).norm().item()
    print(f"\n   Visual tokenizer ΔW: out={vdw_out:.4f}, in={vdw_in:.4f}")
    if vdw_out > 0.01:
        print(f"   ✅ Visual tokenizer öğreniyor!")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"M10 — TOKENIZER EĞİTİM ÖZETİ")
    print(f"{'='*65}")

    print(f"\n   📊 Audio tokenizer (ESC-50):")
    print(f"      Örnek:     {audio_samples_done}")
    print(f"      output_proj ΔW: {adw_out:.4f}")
    print(f"      input_proj  ΔW: {adw_in:.4f}")
    if audio_losses:
        print(f"      İlk loss:  {audio_losses[0]:.4f}")
        print(f"      Son loss:  {audio_losses[-1]:.4f}")
        print(f"      Δ loss:    {audio_losses[0] - audio_losses[-1]:+.4f}")

    print(f"\n   📊 Visual tokenizer (Fashion-MNIST):")
    print(f"      Örnek:     {image_samples_done}")
    print(f"      output_proj ΔW: {vdw_out:.4f}")
    print(f"      input_proj  ΔW: {vdw_in:.4f}")
    if image_losses:
        print(f"      İlk loss:  {image_losses[0]:.4f}")
        print(f"      Son loss:  {image_losses[-1]:.4f}")
        print(f"      Δ loss:    {image_losses[0] - image_losses[-1]:+.4f}")

    # Verdict
    verdicts = []
    if adw_out > 0.1:
        verdicts.append("✅ Audio tokenizer anlamlı öğrendi")
    elif adw_out > 0.01:
        verdicts.append("⚠ Audio tokenizer az öğrendi (daha fazla örnek dene)")
    else:
        verdicts.append("❌ Audio tokenizer öğrenmedi")

    if vdw_out > 0.1:
        verdicts.append("✅ Visual tokenizer anlamlı öğrendi")
    elif vdw_out > 0.01:
        verdicts.append("⚠ Visual tokenizer az öğrendi")
    else:
        verdicts.append("❌ Visual tokenizer öğrenmedi")

    steps = prok.stats
    print(f"\n   Toplam adım:  {steps['steps']}")
    print(f"   TTT updates:  {steps['updates']}")
    print(f"   MLP ΔW:       {steps['weight_change']:.4f}")

    print(f"\n   Verdict: {' | '.join(verdicts)}")

    # ── Save trained tokenizer weights ──
    if args.save_trained:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "tokenizers.pt"
        torch.save({
            "audio_input_proj": at.input_proj.state_dict(),
            "audio_output_proj": at.output_proj.state_dict(),
            "vision_proj": vt.proj.state_dict(),
            "vision_output_proj": vt.output_proj.state_dict(),
            "audio_stats": {"ΔW_out": adw_out, "ΔW_in": adw_in, "samples": audio_samples_done},
            "vision_stats": {"ΔW_out": vdw_out, "ΔW_in": vdw_in, "samples": image_samples_done},
            "config": {
                "audio_n_mels": config.audio_n_mels,
                "audio_patch_frames": config.audio_patch_frames,
                "vision_patch_size": config.vision_patch_size,
            },
        }, save_path)
        print(f"\n   💾 Trained tokenizers saved: {save_path}")

    # Temizlik
    prok.reset()
    del prok, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    return verdicts


if __name__ == "__main__":
    main()
