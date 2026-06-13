"""
Prokopton Eval — Continual-learning değerlendirme ve benchmark.

Metrikler:
  - Forward transfer
  - Backward transfer
  - Forgetting
  - Growth
  - Latency
  - NIAH (Needle in a Haystack)
"""
import time, json, torch
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any


class CLBenchmark:
    """Continual-learning benchmark."""
    
    def __init__(self):
        self.tasks = [
            {
                "name": "T1: Facts",
                "context": (
                    "Zephyria's capital is Aethel. The currency is Zephyr. "
                    "President Elara Voss leads since 2023. Population is 2.3 million."
                ),
                "questions": [
                    ("What is the capital of Zephyria?", "Aethel"),
                    ("What is the currency?", "Zephyr"),
                    ("Who is the president?", "Elara"),
                ]
            },
            {
                "name": "T2: Codes",
                "context": (
                    "Secret code: 7391. Backup code: 4820. Master key: 1563. "
                    "Meeting room: Delta-7. Access code: 9999."
                ),
                "questions": [
                    ("What is the secret code?", "7391"),
                    ("What is the backup code?", "4820"),
                    ("What is the access code?", "9999"),
                ]
            },
            {
                "name": "T3: Places",
                "context": (
                    "Alpha base is in Istanbul. Beta base is in Ankara. "
                    "Gamma base is in Izmir. Delta base is in Antalya."
                ),
                "questions": [
                    ("Where is Alpha base?", "Istanbul"),
                    ("Where is Gamma base?", "Izmir"),
                    ("Where is Delta base?", "Antalya"),
                ]
            },
        ]
        
        self.anchor_questions = [
            ("What is the capital of France?", "Paris"),
            ("What is 2+2?", "4"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ]
    
    def evaluate_accuracy(self, generate_fn: Callable, task: Dict) -> float:
        correct = 0
        for question, expected in task["questions"]:
            prompt = f"{task['context']}\n\nQuestion: {question}\nAnswer:"
            answer = generate_fn(prompt)
            if expected.lower() in answer.lower():
                correct += 1
        return correct / len(task["questions"]) if task["questions"] else 0.0
    
    def evaluate_anchor(self, generate_fn: Callable) -> float:
        correct = 0
        for question, expected in self.anchor_questions:
            answer = generate_fn(f"Question: {question}\nAnswer:")
            if expected.lower() in answer.lower():
                correct += 1
        return correct / len(self.anchor_questions)


def run_full_evaluation(prokopton, model_name: str = "unknown", output_dir: str = "experiments/runs"):
    """
    Tam continual-learning değerlendirme koşumu.
    
    1. Baseline tüm görevler
    2. Sequential learning (T1 → T2 → T3)
    3. Her adımda geriye dönük test
    4. Forgetting, growth, transfer hesapla
    """
    bench = CLBenchmark()
    
    def generate(prompt):
        return prokopton.generate(prompt, max_new=64)
    
    results = {
        "model": model_name,
        "baseline": {},
        "sequence": [],
        "anchor_baseline": 0.0,
        "anchor_final": 0.0,
    }
    
    # ====== Baseline ======
    print("Baseline (donuk model)...")
    for task in bench.tasks:
        acc = bench.evaluate_accuracy(generate, task)
        results["baseline"][task["name"]] = acc
        print(f"  {task['name']}: {acc:.0%}")
    
    anchor_base = bench.evaluate_anchor(generate)
    results["anchor_baseline"] = anchor_base
    
    # ====== Sequential ======
    print("\nSequential learning...")
    t0 = time.time()
    
    for step, task in enumerate(bench.tasks):
        # Öğret
        for _ in range(5):  # 5 tekrar
            prokopton.learn(task["context"])
        
        t_learn = time.time() - t0
        
        # Tüm görevleri test et
        scores = {}
        for t in bench.tasks:
            scores[t["name"]] = bench.evaluate_accuracy(generate, t)
        
        anchor = bench.evaluate_anchor(generate)
        
        step_result = {
            "step": step + 1,
            "task": task["name"],
            "scores": scores,
            "anchor": anchor,
            "time_s": t_learn,
            "ttt_stats": prokopton.stats,
        }
        results["sequence"].append(step_result)
        
        print(f"  Step {step+1} ({task['name']}): " + 
              " | ".join(f"{n}: {s:.0%}" for n, s in scores.items()) +
              f" | anchor: {anchor:.0%}")
    
    results["anchor_final"] = results["sequence"][-1]["anchor"]
    
    # ====== Metrikler ======
    metrics = {}
    
    # Forgetting
    baseline = results["baseline"]
    final_scores = results["sequence"][-1]["scores"]
    forgetting_total = sum(
        max(0, baseline[t["name"]] - final_scores.get(t["name"], 0))
        for t in bench.tasks[:-1]
    )
    metrics["forgetting"] = forgetting_total / max(1, len(bench.tasks) - 1)
    
    # Anchor drift
    metrics["anchor_drift"] = anchor_base - results["anchor_final"]
    
    # Growth
    total_baseline = sum(baseline.values())
    total_final = sum(final_scores.values())
    metrics["growth"] = total_final - total_baseline
    
    # Forward transfer
    ft_values = []
    for i in range(1, len(results["sequence"])):
        prev_task = bench.tasks[i-1]["name"]
        prev_score = results["sequence"][i]["scores"].get(prev_task, 0)
        ft_values.append(prev_score - baseline[prev_task])
    metrics["forward_transfer"] = sum(ft_values) / len(ft_values) if ft_values else 0.0
    
    metrics["total_time_s"] = time.time() - t0
    
    results["metrics"] = metrics
    
    # ====== ÖZET ======
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    verdicts = []
    if metrics["forgetting"] < 0.15:
        verdicts.append("✓ Low forgetting")
    if abs(metrics["anchor_drift"]) < 0.1:
        verdicts.append("✓ Anchor preserved")
    if metrics["growth"] > 0:
        verdicts.append("✓ Growth observed")
    print(f"  Verdict: {' | '.join(verdicts)}")
    print("=" * 50)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    
    return results
