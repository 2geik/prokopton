"""
M10 v2 — Büyük Ölçekli Tokenizer Eğitimi + Model Ağırlıklarına Gömme
=====================================================================
Konuşma (duygu + tonlama) + Çevresel ses + Büyük görseller.
Eğitim sonunda save_pretrained() ile model ağırlıklarına gömülür.

Dataset'ler:
  - RAVDESS (1440 konuşma, 8 duygu: neutral, calm, happy, sad, angry, fearful, disgust, surprised)
  - ESC-50 (2000 çevresel ses, 50 sınıf)
  - COCO 2017 Captions (büyük görseller, caption'lı, 118k train)
  - Flickr30k (30k yüksek kaliteli caption'lı görsel, lmms-lab/flickr30k)

RX 6800 16GB için optimize edildi: tüm sesler torchcodec-free (soundfile ile çözülür).

Kullanım:
  .venv/bin/python experiments/m10_train_tokenizers.py
  .venv/bin/python experiments/m10_train_tokenizers.py --model gpt2
"""

import io
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
import soundfile as sf

from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.backends import detect_backend, load_model, apply_backend_patches


# ═══════════════════════════════════════════════════════════════
# Audio preprocessing
# ═══════════════════════════════════════════════════════════════

def decode_audio_bytes(raw_audio, target_sr=16000):
    """Decode raw audio bytes with soundfile → numpy float32 mono.

    Handles both decoded dicts and raw bytes.
    Returns (array_float32, sample_rate).
    """
    if isinstance(raw_audio, dict):
        if "array" in raw_audio:
            return raw_audio["array"].astype("float32"), raw_audio.get("sampling_rate", target_sr)
        if "bytes" in raw_audio:
            raw = raw_audio["bytes"]
        else:
            raise ValueError(f"Unknown audio dict keys: {raw_audio.keys()}")
    elif isinstance(raw_audio, bytes):
        raw = raw_audio
    else:
        raise ValueError(f"Unknown audio type: {type(raw_audio)}")

    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def preprocess_audio(raw_audio, target_sr=16000, max_duration=2.5):
    """Decode, resample, trim/pad audio to uniform length. Returns tensor."""
    arr, sr = decode_audio_bytes(raw_audio, target_sr)

    # Resample
    if sr != target_sr:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample
            new_len = int(len(arr) * target_sr / sr)
            arr = resample(arr, new_len)

    # Trim/pad
    max_samples = int(max_duration * target_sr)
    if len(arr) > max_samples:
        arr = arr[:max_samples]
    elif len(arr) < max_samples:
        arr = np.pad(arr, (0, max_samples - len(arr)))

    return torch.from_numpy(arr.copy()).float()


# ═══════════════════════════════════════════════════════════════
# Image preprocessing
# ═══════════════════════════════════════════════════════════════

def preprocess_image(pil_image, min_size=96):
    """Resize PIL image to at least min_size, return [C, H, W] float tensor."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    w, h = pil_image.size
    if min(w, h) < min_size:
        scale = min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h))
    arr = torch.from_numpy(
        np.array(pil_image, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)
    return arr


# ═══════════════════════════════════════════════════════════════
# Label helpers
# ═══════════════════════════════════════════════════════════════

RAVDESS_EMOTIONS = [
    "neutral", "calm", "happy", "sad", "angry",
    "fearful", "disgust", "surprised"
]
# Reverse map for string labels
RAVDESS_LABEL_MAP = {e: i for i, e in enumerate(RAVDESS_EMOTIONS)}

# ═══════════════════════════════════════════════════════════════
# Main training
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M10 v2 — Large-Scale Tokenizer Training")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--audio-duration", type=float, default=2.5,
                        help="Max audio duration in seconds (longer audio = more context)")
    parser.add_argument("--image-min-size", type=int, default=128,
                        help="Minimum image dimension (bigger = more detail)")
    parser.add_argument("--ravdess-samples", type=int, default=1440,
                        help="RAVDESS speech samples (max 1440)")
    parser.add_argument("--esc50-samples", type=int, default=2000,
                        help="ESC-50 environmental sounds (max 2000)")
    parser.add_argument("--coco-samples", type=int, default=5000,
                        help="COCO image samples (max 118k)")
    parser.add_argument("--flickr30k-samples", type=int, default=3000,
                        help="Flickr30k image samples (max 30k)")
    parser.add_argument("--output-model", default="prokopton_trained",
                        help="Directory to save trained model via save_pretrained()")
    args = parser.parse_args()

    # ── Backend ──
    backend = detect_backend()
    apply_backend_patches(backend)
    device = backend.device
    dur_str = f"{args.audio_duration}s ses, {args.image_min_size}px görsel"
    print(f"🔧 {backend.description} | {device} | {dur_str}")
    print(f"{'='*68}")

    # ── Load model ──
    print(f"\n📥 Loading {args.model}...")
    model, tokenizer = load_model(args.model, backend)

    # Longer audio → more tokens → need higher cap
    max_audio_tok = min(int(args.audio_duration * 25), 24)
    config = ProkoptonConfig(
        ttt_lr=args.lr,
        ttt_n_layers=3,
        max_audio_tokens=max_audio_tok,
        max_visual_tokens=49,
        save_dir="m10_v2_memory",
        auto_save_every=0,
    )
    prok = Prokopton(model, tokenizer, config)

    at = prok.audio_tokenizer
    vt = prok.vision_tokenizer
    snap = {
        "audio_out": at.output_proj.weight.clone(),
        "audio_in": at.input_proj.weight.clone(),
        "vision_out": vt.output_proj.weight.clone(),
        "vision_in": vt.proj.weight.clone(),
    }

    total_start = time.time()
    total_steps = 0

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: RAVDESS — Emotional Speech (duygu + tonlama)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print(f"PHASE 1: Duygulu Konuşma — RAVDESS ({args.ravdess_samples} örnek)")
    print(f"{'─'*68}")

    n_ravdess = 0
    ravdess_losses = []

    try:
        ds = load_dataset(
            "DynamicSuperb/EmotionalSpeechAudioClassification_RAVDESS-EmotionalSound",
            split="test")  # only split available
        ds = ds.shuffle(seed=42).select(range(min(args.ravdess_samples, len(ds))))
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
    except Exception as e:
        print(f"   ⚠ RAVDESS yüklenemedi: {e}")
        ds = None

    if ds is not None:
        t0 = time.time()
        for idx, sample in enumerate(ds):
            try:
                wav = preprocess_audio(sample["audio"], max_duration=args.audio_duration)
                # label can be string ("angry") or int — handle both
                raw_label = sample["label"]
                if isinstance(raw_label, str):
                    emotion = raw_label  # already human-readable
                else:
                    emotion = RAVDESS_EMOTIONS[int(raw_label)] if int(raw_label) < 8 else f"emotion_{raw_label}"
                text = (
                    f"A person is speaking with a {emotion} tone of voice. "
                    f"The emotional expression sounds {emotion}."
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = prok.learn_multimodal({"text": text, "audio": wav})

                ravdess_losses.append(result["loss"])
                total_steps += 1
                n_ravdess += 1

                if (idx + 1) % 200 == 0:
                    avg = sum(ravdess_losses[-200:]) / 200
                    print(f"   [{idx+1:5d}/{len(ds)}] loss={result['loss']:.4f} avg200={avg:.4f} "
                          f"emotion={emotion}")

            except Exception as e:
                continue

        elapsed = time.time() - t0
        print(f"   ✅ RAVDESS: {n_ravdess} örnek, {elapsed:.1f}s "
              f"({n_ravdess/elapsed:.1f} örnek/sn)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: ESC-50 — Environmental Sounds
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print(f"PHASE 2: Çevresel Sesler — ESC-50 ({args.esc50_samples} örnek)")
    print(f"{'─'*68}")

    n_esc50 = 0
    esc50_losses = []

    try:
        esc50 = load_dataset("ashraq/esc50", split="train")
        esc50 = esc50.shuffle(seed=42).select(range(min(args.esc50_samples, 2000)))
        esc50 = esc50.cast_column("audio", datasets.Audio(decode=False))
    except Exception as e:
        print(f"   ⚠ ESC-50 yüklenemedi: {e}")
        esc50 = None

    if esc50 is not None:
        t0 = time.time()
        for idx, sample in enumerate(esc50):
            try:
                wav = preprocess_audio(sample["audio"], max_duration=args.audio_duration)
                label = sample["category"]
                text = f"The sound of {label} can be heard in this audio recording."

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = prok.learn_multimodal({"text": text, "audio": wav})

                esc50_losses.append(result["loss"])
                total_steps += 1
                n_esc50 += 1

                if (idx + 1) % 500 == 0:
                    avg = sum(esc50_losses[-500:]) / 500
                    print(f"   [{idx+1:5d}/{len(esc50)}] loss={result['loss']:.4f} avg500={avg:.4f}")

            except Exception as e:
                continue

        elapsed = time.time() - t0
        print(f"   ✅ ESC-50: {n_esc50} örnek, {elapsed:.1f}s "
              f"({n_esc50/elapsed:.1f} örnek/sn)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: COCO 2017 — Large Images with Captions
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print(f"PHASE 3: Büyük Görseller — COCO 2017 ({args.coco_samples} örnek)")
    print(f"{'─'*68}")

    n_coco = 0
    coco_losses = []

    try:
        # COCO 2017 val split — PIL images + caption answers (5000 samples)
        coco = load_dataset("lmms-lab/COCO-Caption2017", split="val")
        coco = coco.shuffle(seed=42).select(range(args.coco_samples))
    except Exception as e:
        print(f"   ⚠ COCO yüklenemedi: {e}")
        coco = None

    if coco is not None:
        t0 = time.time()
        for idx, sample in enumerate(coco):
            try:
                img = preprocess_image(sample["image"], min_size=args.image_min_size)
                # COCO-Caption2017 has 'answer' (list of captions), not 'sentences'
                captions = sample.get("answer", sample.get("sentences", sample.get("captions", [])))
                if isinstance(captions, list) and len(captions) > 0:
                    caption = str(captions[0])
                else:
                    caption = "An image showing various objects and scenes."

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = prok.learn_multimodal({"text": caption, "image": img})

                coco_losses.append(result["loss"])
                total_steps += 1
                n_coco += 1

                if (idx + 1) % 1000 == 0:
                    avg = sum(coco_losses[-1000:]) / 1000
                    print(f"   [{idx+1:5d}/{len(coco)}] loss={result['loss']:.4f} avg1000={avg:.4f}")

            except Exception as e:
                continue

        elapsed = time.time() - t0
        print(f"   ✅ COCO: {n_coco} örnek, {elapsed:.1f}s "
              f"({n_coco/elapsed:.1f} örnek/sn)")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Flickr30k — High-Quality Captioned Images
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print(f"PHASE 4: Yüksek Kaliteli Görseller — Flickr30k ({args.flickr30k_samples} örnek)")
    print(f"{'─'*68}")

    n_flickr = 0
    flickr_losses = []

    try:
        flickr = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
        flickr = flickr.shuffle(seed=42).take(args.flickr30k_samples)
    except Exception as e:
        print(f"   ⚠ Flickr30k yüklenemedi: {e}")
        flickr = None

    if flickr is not None:
        t0 = time.time()
        for idx, sample in enumerate(flickr):
            try:
                img = preprocess_image(sample["image"], min_size=args.image_min_size)
                captions = sample.get("caption", [])
                if isinstance(captions, list) and len(captions) > 0:
                    caption = str(captions[0])
                else:
                    caption = "A high-quality photograph showing people, objects, or scenes."

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = prok.learn_multimodal({"text": caption, "image": img})

                flickr_losses.append(result["loss"])
                total_steps += 1
                n_flickr += 1

                if (idx + 1) % 1000 == 0:
                    avg = sum(flickr_losses[-1000:]) / min(1000, len(flickr_losses))
                    print(f"   [{idx+1:5d}/{args.flickr30k_samples}] loss={result['loss']:.4f} avg={avg:.4f}")

            except Exception as e:
                continue

        elapsed = time.time() - t0
        print(f"   ✅ Flickr30k: {n_flickr} örnek, {elapsed:.1f}s "
              f"({n_flickr/elapsed:.1f} örnek/sn)")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY + SAVE
    # ═══════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    adw_out = (at.output_proj.weight - snap["audio_out"]).norm().item()
    adw_in = (at.input_proj.weight - snap["audio_in"]).norm().item()
    vdw_out = (vt.output_proj.weight - snap["vision_out"]).norm().item()
    vdw_in = (vt.proj.weight - snap["vision_in"]).norm().item()

    print(f"\n{'='*68}")
    print(f"M10 v2 — EĞİTİM ÖZETİ")
    print(f"{'='*68}")

    print(f"\n   📊 RAVDESS (duygulu konuşma):")
    print(f"      Örnek:     {n_ravdess}")
    print(f"      Duygular:  {len(RAVDESS_EMOTIONS)} (neutral→surprised)")
    if ravdess_losses:
        print(f"      İlk loss:  {ravdess_losses[0]:.4f}")
        print(f"      Son loss:  {ravdess_losses[-1]:.4f}")
        print(f"      Δ loss:    {ravdess_losses[0] - ravdess_losses[-1]:+.4f}")

    print(f"\n   📊 ESC-50 (çevresel ses):")
    print(f"      Örnek:     {n_esc50}")
    if esc50_losses:
        print(f"      İlk loss:  {esc50_losses[0]:.4f}")
        print(f"      Son loss:  {esc50_losses[-1]:.4f}")
        print(f"      Δ loss:    {esc50_losses[0] - esc50_losses[-1]:+.4f}")

    print(f"\n   📊 COCO 2017 (büyük görseller):")
    print(f"      Örnek:     {n_coco}")
    print(f"      Görsel boy:≥{args.image_min_size}px")
    if coco_losses:
        print(f"      İlk loss:  {coco_losses[0]:.4f}")
        print(f"      Son loss:  {coco_losses[-1]:.4f}")
        print(f"      Δ loss:    {coco_losses[0] - coco_losses[-1]:+.4f}")

    print(f"\n   📊 Flickr30k (yüksek kaliteli caption'lı):")
    print(f"      Örnek:     {n_flickr}")
    print(f"      Görsel boy:≥{args.image_min_size}px")
    if flickr_losses:
        print(f"      İlk loss:  {flickr_losses[0]:.4f}")
        print(f"      Son loss:  {flickr_losses[-1]:.4f}")
        print(f"      Δ loss:    {flickr_losses[0] - flickr_losses[-1]:+.4f}")

    print(f"\n   🔧 Ağırlık değişimleri:")
    print(f"      Audio out:  ΔW = {adw_out:.4f}")
    print(f"      Audio in:   ΔW = {adw_in:.4f}")
    print(f"      Vision out: ΔW = {vdw_out:.4f}")
    print(f"      Vision in:  ΔW = {vdw_in:.4f}")

    total_samples = n_ravdess + n_esc50 + n_coco + n_flickr
    print(f"\n   📈 Toplam: {total_samples} örnek, {total_steps} adım, "
          f"{total_elapsed:.0f}s ({total_elapsed/60:.1f}dk)")

    # Verdict
    verdicts = []
    if adw_out > 0.5:
        verdicts.append("✅ Audio tokenizer anlamlı öğrendi (konuşma + ses)")
    elif adw_out > 0.05:
        verdicts.append("⚠ Audio az öğrendi")
    else:
        verdicts.append("❌ Audio öğrenmedi")
    if vdw_out > 0.5:
        verdicts.append("✅ Visual tokenizer anlamlı öğrendi (COCO + Flickr30k)")
    elif vdw_out > 0.05:
        verdicts.append("⚠ Visual az öğrendi")
    else:
        verdicts.append("❌ Visual öğrenmedi")
    print(f"\n   Verdict: {' | '.join(verdicts)}")

    # ── SAVE: Model ağırlıklarına GÖM ──
    print(f"\n{'─'*68}")
    print(f"💾 save_pretrained() — öğrenilen bilgi modele gömülüyor...")
    out_path = Path(args.output_model)
    prok.save_pretrained(str(out_path))
    print(f"{'─'*68}")

    # Temizlik
    prok.reset()
    del prok, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✅ Eğitim tamamlandı. Model: {out_path}/")
    print(f"   AutoModelForCausalLM.from_pretrained('{out_path}') ile yüklenebilir.")


if __name__ == "__main__":
    main()
