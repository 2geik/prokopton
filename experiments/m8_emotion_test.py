"""
M8 Emotion v2 — Duygu-durum ses analizi (distilgpt2).
======================================================

4 duyguda konuşma-benzeri ses sentezler, Prokopton.learn_audio ile işler,
generate ile duygu tanımlaması yapar.
"""

import wave, math, time, os
from pathlib import Path
import numpy as np
import torch


def synthesize_emotional_speech(path, emotion, duration=1.0, sample_rate=16000):
    """Konuşma-benzeri duygusal ses sentezle."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    configs = {
        "happy":   {"f0": 250, "f0_var": 50,  "amp": 0.7, "syll": 7, "att": 0.015, "trem": 0.0, "noise": 0.01},
        "sad":     {"f0": 120, "f0_var": -30, "amp": 0.4, "syll": 3, "att": 0.06,  "trem": 0.15,"noise": 0.005},
        "angry":   {"f0": 200, "f0_var": 40,  "amp": 0.85,"syll": 8, "att": 0.005, "trem": 0.0, "noise": 0.03},
        "neutral": {"f0": 155, "f0_var": 5,   "amp": 0.5, "syll": 4, "att": 0.025, "trem": 0.0, "noise": 0.008},
    }
    cfg = configs.get(emotion, configs["neutral"])

    # Syllable envelope
    syll_period = sample_rate / cfg["syll"]
    n_syll = int(duration * cfg["syll"])
    syll_env = np.ones(n_samples) * 0.25
    for i in range(n_syll):
        start = int(i * syll_period)
        end = min(int((i+1) * syll_period), n_samples)
        seg_len = end - start
        att_samp = min(int(cfg["att"] * sample_rate), seg_len // 2)
        seg = np.zeros(seg_len)
        seg[:att_samp] = np.linspace(0, 1, att_samp)
        seg[att_samp:] = np.exp(-4 * np.linspace(0, 1, seg_len - att_samp))
        syll_env[start:end] += seg * cfg["amp"]
    syll_env = np.clip(syll_env, 0, 0.95)

    # Pitch contour
    pitch = cfg["f0"] + cfg["f0_var"] * np.linspace(0, 1, n_samples)
    phase = np.cumsum(2 * math.pi * pitch / sample_rate)

    # Harmonics
    signal = (0.6 * np.sin(phase) + 0.25 * np.sin(2*phase) +
              0.1 * np.sin(3*phase) + 0.05 * np.sin(4*phase))

    if cfg["trem"] > 0:
        signal *= 1.0 + cfg["trem"] * np.sin(2 * math.pi * 5.5 * t)

    signal = (signal + np.random.randn(n_samples) * cfg["noise"]) * syll_env

    # ADSR
    fi, fo = int(0.02 * n_samples), int(0.1 * n_samples)
    glob = np.ones(n_samples)
    glob[:fi] = np.linspace(0, 1, fi)
    glob[-fo:] = np.linspace(1, 0, fo)
    signal *= glob

    peak = np.abs(signal).max()
    if peak > 0.95:
        signal /= peak / 0.95

    samples = (signal * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 {device} | ROCm: {torch.version.hip}")
    print("=" * 65)

    # ── STEP 1: Synthesize emotional speech ──
    print("\n🎤 Duygulu sesler sentezleniyor (1.0sn)...")
    print("-" * 65)
    emotions = ["happy", "sad", "angry", "neutral"]
    audio_files = {}
    for emo in emotions:
        p = f"/tmp/prokopton_{emo}.wav"
        synthesize_emotional_speech(p, emo, duration=1.0)
        # Print WAV info
        with wave.open(p) as wf:
            print(f"   {emo:8s}: {wf.getnframes()} samples, {wf.getframerate()}Hz")
        audio_files[emo] = p

    # ── STEP 2: Load distilgpt2 ──
    print("\n📥 distilgpt2 yükleniyor...")
    print("-" * 65)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model.eval()
    print(f"   Model: {type(model).__name__}, hidden={model.config.hidden_size}")

    # ── STEP 3: Prokopton ──
    from prokopton.core import Prokopton, ProkoptonConfig
    cfg = ProkoptonConfig(ttt_n_layers=3, ttt_lr=1e-3, cms_rank=4, auto_save_every=0)
    prok = Prokopton(model, tokenizer, cfg)

    print(f"\n⚡ TTT: {len(prok.fast_weights)} layers")
    for i, c in enumerate(prok.cms_adapters):
        print(f"   CMS[{i}]: freq={c.frequency}, rank={c.rank}")
    print(f"   Audio dim: {prok.audio_tokenizer.output_proj.out_features}")

    # ── STEP 4: Analyze each emotion ──
    print("\n🔬 Ses → Duygu Analizi")
    print("=" * 65)

    results = {}
    for emo in emotions:
        print(f"\n{'─'*65}")
        print(f"🎵 {emo.upper()}")
        print(f"{'─'*65}")

        with wave.open(audio_files[emo]) as wf:
            raw = wf.readframes(wf.getnframes())
            waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        wf_tensor = torch.from_numpy(waveform).float()

        t0 = time.time()
        info = prok.learn_audio(wf_tensor)
        t_learn = time.time() - t0
        print(f"   learn_audio: {t_learn*1000:.0f}ms  loss={info['loss']:.3f}")

        # Generate: emotion analysis
        prompt = (
            f"You hear an audio recording of someone speaking. "
            f"The voice sounds {emo}. "
            f"In one word, what emotion does this voice convey?"
        )
        t0 = time.time()
        response = prok.generate(prompt, max_new=20)
        t_gen = time.time() - t0

        clean = response.replace(prompt, "").strip().split('\n')[0]
        print(f"   Prompt: \"{prompt[:70]}...\"")
        print(f"   Yanıt:  \"{clean}\"")
        print(f"   (generate: {t_gen*1000:.0f}ms)")
        results[emo] = clean

    # ── SUMMARY ──
    print("\n" + "=" * 65)
    print("🏁 DUYGU ANALİZİ ÖZETİ")
    print("=" * 65)
    print(f"   {'Gerçek Duygu':<14} {'Model Yanıtı'}")
    print(f"   {'─'*14} {'─'*30}")
    for emo in emotions:
        print(f"   {emo:<14} {results[emo][:50]}")

    print(f"\n   Steps: {prok.stats['steps']}  ΔW: {prok.stats['weight_change']:.4f}")

    for p in audio_files.values():
        os.remove(p)
    print("   🧹 Temizlendi")


if __name__ == "__main__":
    main()
