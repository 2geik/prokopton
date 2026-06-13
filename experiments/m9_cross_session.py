"""
M9 — Cross-Session Continual Learning Benchmark
================================================
Multi-restart benchmark: modeli 3 ayrı "oturumda" test et.
Her oturumda öğren → kaydet → sıfırla → yükle döngüsüyle
forgetting curve çıkar.

Session 1:  5 bilgi öğret, test et, kaydet, sıfırla
Session 2:  Belleği yükle, eski bilgileri test et, 5 YENİ bilgi öğret, kaydet, sıfırla
Session 3:  Belleği yükle, TÜM bilgileri test et (10 soru), forgetting hesapla

Kullanım:
  .venv/bin/python experiments/m9_cross_session.py
  .venv/bin/python experiments/m9_cross_session.py --model distilgpt2
"""

import json
import time
import sys
import argparse
from pathlib import Path

import torch

# Prokopton imports
from prokopton.core import Prokopton, ProkoptonConfig
from prokopton.backends import detect_backend, load_model, apply_backend_patches, get_vram_usage

# ============================================================
# Cross-Session Benchmark Data
# ============================================================

SESSION_1_FACTS = {
    "name": "S1: İsim-Yaş Eşleşmeleri",
    "context": (
        "Alice is 28 years old. Bob is 35 years old. "
        "Carol is 42 years old. Dave is 19 years old. "
        "Eve is 31 years old."
    ),
    "questions": [
        ("How old is Alice?", "28"),
        ("How old is Bob?", "35"),
        ("How old is Carol?", "42"),
        ("How old is Dave?", "19"),
        ("How old is Eve?", "31"),
    ],
}

SESSION_2_FACTS = {
    "name": "S2: Şehir-Kod Eşleşmeleri",
    "context": (
        "Istanbul's area code is 212. Ankara's area code is 312. "
        "Izmir's area code is 232. Antalya's area code is 242. "
        "Bursa's area code is 224."
    ),
    "questions": [
        ("What is Istanbul's area code?", "212"),
        ("What is Ankara's area code?", "312"),
        ("What is Izmir's area code?", "232"),
        ("What is Antalya's area code?", "242"),
        ("What is Bursa's area code?", "224"),
    ],
}

# Sabit benchmark soruları — unutma kontrolü için
ANCHOR_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("What color is the sky?", "blue"),
]


# ============================================================
# Helpers
# ============================================================

def evaluate_accuracy(generate_fn, questions, context=""):
    """Bir soru listesinde doğruluk oranını ölç."""
    correct = 0
    total = len(questions)
    for question, expected in questions:
        if context:
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        answer = generate_fn(prompt)
        if expected.lower() in answer.lower():
            correct += 1
    return correct / total if total > 0 else 0.0


def compute_forgetting(baseline_acc, current_acc):
    """Forgetting = baseline - current (clamped at 0)."""
    return max(0.0, baseline_acc - current_acc)


# ============================================================
# Main Benchmark
# ============================================================

def run_cross_session_benchmark(model_name="distilgpt2", lr=1e-3, save_dir="m9_memory"):
    """
    Cross-session continual learning benchmark.

    Args:
        model_name: HuggingFace model ID (distilgpt2, gpt2, etc.)
        lr: TTT learning rate
        save_dir: Directory for persistent memory
    """
    print(f"M9 Cross-Session Benchmark — {model_name}")
    print("=" * 65)

    # Backend detection
    backend = detect_backend()
    apply_backend_patches(backend)
    device = backend.device
    print(f"Backend: {backend.description} | Device: {device}")
    if backend.vram_gb > 0:
        print(f"VRAM: {backend.vram_gb:.1f} GB")

    # Session 1 model
    print(f"\nLoading model for Session 1: {model_name}...")
    model, tokenizer = load_model(model_name, backend)

    config = ProkoptonConfig(
        ttt_lr=lr,
        ttt_n_layers=5,
        save_dir=save_dir,
        auto_save_every=0,  # manual save
        per_capacity=64,
    )

    results = {
        "model": model_name,
        "backend": backend.name,
        "device": device,
        "ttt_lr": lr,
        "sessions": [],
        "metrics": {},
    }

    # Helper to create a Prokopton instance
    def make_prokopton(m, tok, cfg):
        return Prokopton(m, tok, cfg)

    # ================================================================
    # SESSION 1: Öğret 5 bilgi → test → kaydet → sıfırla
    # ================================================================
    print("\n" + "─" * 65)
    print("SESSION 1: İsim-Yaş Eşleşmelerini Öğren")
    print("─" * 65)

    prok1 = make_prokopton(model, tokenizer, config)

    # Baseline (öğrenmeden önce)
    baseline_s1 = evaluate_accuracy(
        lambda p: prok1.generate(p, max_new=32),
        SESSION_1_FACTS["questions"],
        SESSION_1_FACTS["context"],
    )
    anchor_baseline = evaluate_accuracy(
        lambda p: prok1.generate(p, max_new=32),
        ANCHOR_QUESTIONS,
    )
    print(f"  Baseline (S1 facts): {baseline_s1:.0%}")
    print(f"  Anchor baseline:     {anchor_baseline:.0%}")

    # Öğret
    print(f"  Teaching S1 facts (5x repetition)...")
    s1_losses = []
    t0 = time.time()
    for rep in range(5):
        info = prok1.learn(SESSION_1_FACTS["context"])
        s1_losses.append(info["loss"])
    s1_learn_time = time.time() - t0

    # Test
    s1_accuracy = evaluate_accuracy(
        lambda p: prok1.generate(p, max_new=32),
        SESSION_1_FACTS["questions"],
        SESSION_1_FACTS["context"],
    )
    anchor_s1 = evaluate_accuracy(
        lambda p: prok1.generate(p, max_new=32),
        ANCHOR_QUESTIONS,
    )
    s1_stats = prok1.stats

    print(f"  After learning (S1): {s1_accuracy:.0%}")
    print(f"  Anchor after S1:     {anchor_s1:.0%}")
    print(f"  Loss: {s1_losses[-1]:.4f} (avg: {sum(s1_losses)/len(s1_losses):.4f})")
    print(f"  Learn time: {s1_learn_time:.2f}s")

    # Kaydet
    prok1.save()
    s1_vram = get_vram_usage(backend)

    session1 = {
        "session": 1,
        "name": SESSION_1_FACTS["name"],
        "baseline_accuracy": baseline_s1,
        "post_accuracy": s1_accuracy,
        "anchor": anchor_s1,
        "anchor_baseline": anchor_baseline,
        "step": prok1.step_counter,
        "losses": s1_losses,
        "learn_time_s": s1_learn_time,
        "vram_gb": s1_vram,
        "weight_changes": [fw.weight_change for fw in prok1.fast_weights],
        "total_surprise": sum(fw.total_surprise for fw in prok1.fast_weights),
    }
    results["sessions"].append(session1)

    # Sıfırla (RAM temizliği)
    prok1.reset()
    print(f"  💾 Saved. Memory reset.")

    # ================================================================
    # SESSION 2: Yükle → eskiyi test et → 5 YENİ bilgi öğret → kaydet → sıfırla
    # ================================================================
    print("\n" + "─" * 65)
    print("SESSION 2: Belleği Yükle, Eski Bilgileri Test Et, Yeni Bilgiler Öğren")
    print("─" * 65)

    # YENİ bir model yükle (farklı oturum simülasyonu)
    print(f"  Loading fresh model for Session 2...")
    model2, tokenizer2 = load_model(model_name, backend)
    prok2 = make_prokopton(model2, tokenizer2, config)

    # Önceki belleği yükle
    loaded = prok2.load()
    print(f"  Memory loaded: {loaded}")

    # Eski (S1) bilgileri test et
    s2_s1_recall = evaluate_accuracy(
        lambda p: prok2.generate(p, max_new=32),
        SESSION_1_FACTS["questions"],
        SESSION_1_FACTS["context"],
    )
    forgetting_s1 = compute_forgetting(s1_accuracy, s2_s1_recall)
    print(f"  S1 recall (after load): {s2_s1_recall:.0%}")
    print(f"  Forgetting (S1):        {forgetting_s1:.0%}")

    # Yeni (S2) bilgileri baseline test
    baseline_s2 = evaluate_accuracy(
        lambda p: prok2.generate(p, max_new=32),
        SESSION_2_FACTS["questions"],
        SESSION_2_FACTS["context"],
    )
    print(f"  Baseline (S2 facts):    {baseline_s2:.0%}")

    # Öğret
    print(f"  Teaching S2 facts (5x repetition)...")
    s2_losses = []
    t0 = time.time()
    for rep in range(5):
        info = prok2.learn(SESSION_2_FACTS["context"])
        s2_losses.append(info["loss"])
    s2_learn_time = time.time() - t0

    # Tüm bilgileri test et (S1 + S2)
    s2_accuracy = evaluate_accuracy(
        lambda p: prok2.generate(p, max_new=32),
        SESSION_2_FACTS["questions"],
        SESSION_2_FACTS["context"],
    )
    s2_s1_after_new = evaluate_accuracy(
        lambda p: prok2.generate(p, max_new=32),
        SESSION_1_FACTS["questions"],
        SESSION_1_FACTS["context"],
    )
    anchor_s2 = evaluate_accuracy(
        lambda p: prok2.generate(p, max_new=32),
        ANCHOR_QUESTIONS,
    )

    # Forward transfer: S2 öğrenildikten sonra S1 ne kadar hatırlanıyor?
    ft_s1 = s2_s1_after_new - s2_s1_recall  # positive = improved, negative = interference

    print(f"  After learning (S2):   {s2_accuracy:.0%}")
    print(f"  S1 after S2 learned:   {s2_s1_after_new:.0%}")
    print(f"  Forward transfer (S1): {ft_s1:+.0%}")
    print(f"  Anchor after S2:       {anchor_s2:.0%}")

    s2_stats = prok2.stats
    s2_vram = get_vram_usage(backend)

    # Kaydet
    prok2.save()

    session2 = {
        "session": 2,
        "name": SESSION_2_FACTS["name"],
        "s1_recall_after_load": s2_s1_recall,
        "forgetting_s1": forgetting_s1,
        "baseline_accuracy_s2": baseline_s2,
        "post_accuracy_s2": s2_accuracy,
        "s1_after_new_learning": s2_s1_after_new,
        "forward_transfer_s1": ft_s1,
        "anchor": anchor_s2,
        "anchor_baseline": anchor_baseline,
        "step": prok2.step_counter,
        "losses": s2_losses,
        "learn_time_s": s2_learn_time,
        "vram_gb": s2_vram,
        "weight_changes": [fw.weight_change for fw in prok2.fast_weights],
        "total_surprise": sum(fw.total_surprise for fw in prok2.fast_weights),
    }
    results["sessions"].append(session2)

    # Sıfırla
    prok2.reset()
    print(f"  💾 Saved. Memory reset.")

    # ================================================================
    # SESSION 3: Yükle → TÜM bilgileri test et (10 soru)
    # ================================================================
    print("\n" + "─" * 65)
    print("SESSION 3: Belleği Yükle, Tüm Bilgileri Test Et (10 Soru)")
    print("─" * 65)

    print(f"  Loading fresh model for Session 3...")
    model3, tokenizer3 = load_model(model_name, backend)
    prok3 = make_prokopton(model3, tokenizer3, config)

    loaded = prok3.load()
    print(f"  Memory loaded: {loaded}")

    # Tüm sorular (S1 + S2 = 10 soru)
    all_questions = SESSION_1_FACTS["questions"] + SESSION_2_FACTS["questions"]
    all_context = SESSION_1_FACTS["context"] + "\n" + SESSION_2_FACTS["context"]

    # Tüm sorularda doğruluk
    s3_s1_accuracy = evaluate_accuracy(
        lambda p: prok3.generate(p, max_new=32),
        SESSION_1_FACTS["questions"],
        SESSION_1_FACTS["context"],
    )
    s3_s2_accuracy = evaluate_accuracy(
        lambda p: prok3.generate(p, max_new=32),
        SESSION_2_FACTS["questions"],
        SESSION_2_FACTS["context"],
    )
    s3_overall = (s3_s1_accuracy * 5 + s3_s2_accuracy * 5) / 10

    anchor_s3 = evaluate_accuracy(
        lambda p: prok3.generate(p, max_new=32),
        ANCHOR_QUESTIONS,
    )

    # Forgetting hesapla
    forgetting_s1_final = compute_forgetting(s1_accuracy, s3_s1_accuracy)
    forgetting_s2_final = compute_forgetting(s2_accuracy, s3_s2_accuracy)

    print(f"  S1 recall (final):     {s3_s1_accuracy:.0%}")
    print(f"  S2 recall (final):     {s3_s2_accuracy:.0%}")
    print(f"  Overall (10Q):         {s3_overall:.0%}")
    print(f"  Forgetting S1:         {forgetting_s1_final:.0%}")
    print(f"  Forgetting S2:         {forgetting_s2_final:.0%}")
    print(f"  Anchor final:          {anchor_s3:.0%}")

    s3_vram = get_vram_usage(backend)
    s3_stats = prok3.stats

    session3 = {
        "session": 3,
        "name": "Final Recall (S1 + S2)",
        "s1_accuracy": s3_s1_accuracy,
        "s2_accuracy": s3_s2_accuracy,
        "overall_accuracy": s3_overall,
        "forgetting_s1": forgetting_s1_final,
        "forgetting_s2": forgetting_s2_final,
        "anchor": anchor_s3,
        "step": prok3.step_counter,
        "vram_gb": s3_vram,
        "weight_changes": [fw.weight_change for fw in prok3.fast_weights],
        "total_surprise": sum(fw.total_surprise for fw in prok3.fast_weights),
    }
    results["sessions"].append(session3)

    # ================================================================
    # METRICS & SUMMARY
    # ================================================================
    avg_forgetting = (forgetting_s1_final + forgetting_s2_final) / 2
    anchor_drift = anchor_baseline - anchor_s3
    learning_curve = [s1_accuracy, s2_accuracy, s3_overall]

    metrics = {
        "avg_forgetting": avg_forgetting,
        "anchor_drift": anchor_drift,
        "final_s1_accuracy": s3_s1_accuracy,
        "final_s2_accuracy": s3_s2_accuracy,
        "final_overall_accuracy": s3_overall,
        "learning_curve": learning_curve,
        "forgetting_curve": [
            forgetting_s1,           # S1→S2 arası forgetting
            forgetting_s1_final,     # S1→S3 arası forgetting
            forgetting_s2_final,     # S2→S3 arası forgetting
        ],
        "total_learn_time_s": s1_learn_time + s2_learn_time,
        "peak_vram_gb": max(
            s.get("vram_gb", 0) for s in results["sessions"]
        ),
        "anchor_baseline": anchor_baseline,
        "anchor_final": anchor_s3,
    }
    results["metrics"] = metrics

    # ================================================================
    # PRINT SUMMARY
    # ================================================================
    print("\n" + "=" * 65)
    print("M9 CROSS-SESSION BENCHMARK ÖZETİ")
    print("=" * 65)
    print(f"\n  Model:          {model_name}")
    print(f"  Backend:        {backend.description}")
    print(f"  TTT LR:         {lr}")
    print(f"\n  Learning Curve:")
    print(f"    S1 (5 öğe):   {s1_accuracy:.0%}")
    print(f"    S2 (5 öğe):   {s2_accuracy:.0%}")
    print(f"    S3 (10 öğe):  {s3_overall:.0%}")
    print(f"\n  Forgetting:")
    print(f"    S1 after S2:   {forgetting_s1:.0%}")
    print(f"    S1 final:      {forgetting_s1_final:.0%}")
    print(f"    S2 final:      {forgetting_s2_final:.0%}")
    print(f"    Average:       {avg_forgetting:.0%}")
    print(f"\n  Anchor:")
    print(f"    Baseline:      {anchor_baseline:.0%}")
    print(f"    Final:         {anchor_s3:.0%}")
    print(f"    Drift:         {anchor_drift:+.0%}")
    print(f"\n  Weight Changes:")
    for i, wc in enumerate(session1.get("weight_changes", [])):
        print(f"    Layer {i}: ΔW = {wc:.6f}")
    print(f"\n  Total learn time: {s1_learn_time + s2_learn_time:.2f}s")

    # Verdict
    verdicts = []
    if avg_forgetting < 0.2:
        verdicts.append("✓ Low forgetting")
    else:
        verdicts.append("✗ High forgetting")
    if abs(anchor_drift) < 0.15:
        verdicts.append("✓ Anchor preserved")
    else:
        verdicts.append("✗ Anchor degraded")
    if learning_curve[-1] >= learning_curve[0]:
        verdicts.append("✓ Knowledge accumulated")
    else:
        verdicts.append("✗ Knowledge decayed")
    print(f"\n  Verdict: {' | '.join(verdicts)}")
    print("=" * 65)

    # ================================================================
    # SAVE JSON
    # ================================================================
    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m9_cross_session.json"

    # Convert any non-serializable values
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(out_path, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"\n  💾 Results saved: {out_path}")

    # Clean up
    prok3.reset()
    del prok1, prok2, prok3, model, model2, model3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M9 — Cross-Session Continual Learning Benchmark",
    )
    parser.add_argument(
        "--model", "-m",
        default="distilgpt2",
        help="Model name (default: distilgpt2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="TTT learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--save-dir",
        default="m9_memory",
        help="Memory save directory (default: m9_memory)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    args = parser.parse_args()

    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    run_cross_session_benchmark(
        model_name=args.model,
        lr=args.lr,
        save_dir=args.save_dir,
    )
