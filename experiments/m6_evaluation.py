"""
M6 — Bilimsel Değerlendirme: Transfer, Forgetting, Growth, Latency

Metrikler:
  - Forward transfer: Yeni görev eski görevi bozuyor mu?
  - Backward transfer: Eski görev yeni görevden sonra hala hatırlanıyor mu?
  - Forgetting ≈ 0: Sabit benchmark puanı düşmüyor mu?
  - Growth: Model bilgi biriktirdikçe performans artıyor mu?
  - Latency: Per-chunk TTT güncelleme süresi

Kullanım:
  .venv/bin/python experiments/m6_evaluation.py
"""
import time, json, torch
from pathlib import Path
from collections import defaultdict


class ContinualLearningBenchmark:
    """
    Basit continual-learning benchmark.
    
    Görevler:
      T1: İsim-hatırlama (5 yeni isim)
      T2: Sayı-dizisi (3 sayı)
      T3: Yer-ismi (3 şehir)
    """
    
    def __init__(self):
        self.tasks = [
            {
                "name": "T1: İsimler",
                "context": "Alice is 28 years old. Bob is 35. Carol is 42. Dave is 19. Eve is 31.",
                "questions": [
                    ("How old is Alice?", "28"),
                    ("How old is Bob?", "35"),
                    ("How old is Eve?", "31"),
                ]
            },
            {
                "name": "T2: Sayılar",
                "context": "The secret code is 7391. The backup code is 4820. The master key is 1563.",
                "questions": [
                    ("What is the secret code?", "7391"),
                    ("What is the backup code?", "4820"),
                    ("What is the master key?", "1563"),
                ]
            },
            {
                "name": "T3: Şehirler",
                "context": "Project Alpha is based in Istanbul. Project Beta is in Ankara. Project Gamma is in Izmir.",
                "questions": [
                    ("Where is Project Alpha based?", "Istanbul"),
                    ("Where is Project Beta based?", "Ankara"),
                    ("Where is Project Gamma based?", "Izmir"),
                ]
            },
        ]
        
        # Sabit benchmark soruları (öğretilmeyen, genel bilgi)
        self.anchor_questions = [
            ("What is the capital of France?", "Paris"),
            ("What is 2+2?", "4"),
            ("What color is the sky?", "blue"),
        ]
    
    def evaluate_accuracy(self, generate_fn, task):
        """Bir görevdeki soruların doğruluk oranını ölç."""
        correct = 0
        for question, expected in task["questions"]:
            prompt = f"{task['context']}\n\nQuestion: {question}\nAnswer:"
            answer = generate_fn(prompt)
            ans_lower = answer.lower()
            if expected.lower() in ans_lower:
                correct += 1
        return correct / len(task["questions"])
    
    def evaluate_forgetting(self, generate_fn, task, baseline_accuracy):
        """Forgetting = baseline - current accuracy."""
        current = self.evaluate_accuracy(generate_fn, task)
        return max(0, baseline_accuracy - current)
    
    def evaluate_anchor(self, generate_fn):
        """Sabit benchmark — unutma kontrolü."""
        correct = 0
        for question, expected in self.anchor_questions:
            answer = generate_fn(f"Question: {question}\nAnswer:")
            if expected.lower() in answer.lower():
                correct += 1
        return correct / len(self.anchor_questions)


def run_benchmark(model_name="distilgpt2"):
    """
    Tam continual-learning değerlendirme koşumu.
    
    1. Baseline: tüm görevleri TTT olmadan test et
    2. Sequential learning: T1 → T2 → T3 (her birini TTT ile öğret)
    3. Her adımda tüm görevleri ve anchor'ı test et
    4. Forward/backward transfer, forgetting, growth hesapla
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"M6 Evaluation Suite — {model_name}")
    print(f"Device: {device}  ROCm: {torch.version.hip if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Model yükle
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # TTT setup — son MLP katmanı
    proj_layer = None
    for name, module in model.named_modules():
        if 'mlp' in name and 'proj' in name.lower() and hasattr(module, 'weight'):
            if len(module.weight.shape) == 2:
                proj_layer = module
    if proj_layer is None:
        raise RuntimeError("MLP projeksiyon katmanı bulunamadı")
    
    print(f"  TTT layer: {list(proj_layer.weight.shape)}")
    
    def generate(prompt, max_new=32):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, 
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def ttt_learn(context, lr=1e-3):
        """Bir context'i TTT ile öğren."""
        model.train()
        tokens = tokenizer(context, return_tensors="pt").to(device)
        outputs = model(**tokens)
        logits = outputs.logits[:, :-1].contiguous()
        targets = tokens.input_ids[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        grad = torch.autograd.grad(loss, proj_layer.weight, retain_graph=False)[0]
        with torch.no_grad():
            proj_layer.weight.add_(-lr * grad)
        model.zero_grad()
        model.eval()
        return loss.item()
    
    bench = ContinualLearningBenchmark()
    
    # ======== Baseline ========
    print("\n--- BASELINE (donuk model) ---")
    original_W = proj_layer.weight.clone()
    
    baseline_scores = {}
    for task in bench.tasks:
        acc = bench.evaluate_accuracy(generate, task)
        baseline_scores[task["name"]] = acc
        print(f"  {task['name']}: {acc:.0%}")
    
    anchor_baseline = bench.evaluate_anchor(generate)
    print(f"  Anchor (genel bilgi): {anchor_baseline:.0%}")
    
    # ======== Sequential Learning ========
    print("\n--- SEQUENTIAL LEARNING (TTT açık) ---")
    
    results = {
        "baseline": baseline_scores,
        "anchor_baseline": anchor_baseline,
        "sequence": [],
    }
    
    t0_total = time.time()
    
    for step, task in enumerate(bench.tasks):
        print(f"\n  Adım {step+1}: {task['name']} öğretiliyor...")
        
        # TTT ile öğret
        t0 = time.time()
        loss = ttt_learn(task["context"], lr=1e-3)
        t_learn = time.time() - t0
        
        # Tüm görevleri test et
        scores = {}
        for prev_task in bench.tasks[:step+2]:
            acc = bench.evaluate_accuracy(generate, prev_task)
            scores[prev_task["name"]] = acc
        
        anchor_now = bench.evaluate_anchor(generate)
        weight_change = (proj_layer.weight - original_W).norm().item()
        
        step_result = {
            "step": step + 1,
            "learned_task": task["name"],
            "loss": loss,
            "learn_time_ms": t_learn * 1000,
            "scores": scores,
            "anchor": anchor_now,
            "weight_change": weight_change,
        }
        
        # Forward transfer: yeni görev, eskiyi bozdu mu?
        if step > 0:
            prev_task = bench.tasks[step-1]["name"]
            forward_transfer = scores.get(prev_task, 0) - baseline_scores[prev_task]
            step_result["forward_transfer"] = forward_transfer
        
        results["sequence"].append(step_result)
        
        print(f"    Loss: {loss:.4f}  Time: {t_learn*1000:.0f}ms  ΔW: {weight_change:.4f}")
        for name, acc in scores.items():
            marker = "← yeni" if name == task["name"] else ""
            print(f"    {name}: {acc:.0%} {marker}")
        
        # Forgetting kontrolü
        if step > 0:
            for prev_step in range(step):
                prev_name = bench.tasks[prev_step]["name"]
                prev_score = scores.get(prev_name, 0)
                baseline = baseline_scores[prev_name]
                forgetting = max(0, baseline - prev_score)
                if forgetting > 0:
                    print(f"    ⚠ Forgetting ({prev_name}): {forgetting:.0%}")
        
        print(f"    Anchor: {anchor_now:.0%} (baseline: {anchor_baseline:.0%})")
    
    total_time = time.time() - t0_total
    
    # ======== ÖZET ========
    print("\n" + "=" * 60)
    print("M6 DEĞERLENDİRME ÖZETİ")
    print("=" * 60)
    
    print(f"\n  Model: {model_name}")
    print(f"  Toplam süre: {total_time:.1f}s")
    print(f"  Görev sayısı: {len(bench.tasks)}")
    
    # Forward transfer
    ft_values = [s.get("forward_transfer", 0) for s in results["sequence"][1:]]
    if ft_values:
        avg_ft = sum(ft_values) / len(ft_values)
        print(f"\n  Forward Transfer (avg): {avg_ft:+.0%}")
        print(f"    (>0 = olumlu, <0 = olumsuz transfer)")
    
    # Forgetting
    final_scores = results["sequence"][-1]["scores"]
    forgetting_total = 0
    for task in bench.tasks[:-1]:  # Son görev hariç hepsi
        f = max(0, baseline_scores[task["name"]] - final_scores.get(task["name"], 0))
        forgetting_total += f
    avg_forgetting = forgetting_total / max(1, len(bench.tasks) - 1)
    print(f"\n  Forgetting (avg): {avg_forgetting:.0%}")
    print(f"    (Hedef: ≈0)")
    
    # Anchor (catastrophic forgetting koruması)
    anchor_final = results["sequence"][-1]["anchor"]
    anchor_drift = anchor_baseline - anchor_final
    print(f"\n  Anchor drift: {anchor_drift:+.0%}")
    print(f"    (Hedef: ≈0, donuk base sayesinde)")
    
    # Growth (öğrenilen görevlerin toplamı)
    final_total = sum(final_scores.values())
    baseline_total = sum(baseline_scores.values())
    growth = final_total - baseline_total
    print(f"\n  Growth: {growth:+.2f} puan")
    print(f"    Baseline total: {baseline_total:.2f} → Final: {final_total:.2f}")
    
    # Latency
    latencies = [s["learn_time_ms"] for s in results["sequence"]]
    print(f"\n  Latency per task (avg): {sum(latencies)/len(latencies):.0f}ms")
    
    # Verdict
    verdicts = []
    if avg_forgetting < 0.15:
        verdicts.append("✓ Forgetting düşük")
    else:
        verdicts.append("✗ Forgetting yüksek")
    
    if anchor_drift < 0.1:
        verdicts.append("✓ Anchor korundu (donuk base)")
    else:
        verdicts.append("✗ Anchor bozuldu")
    
    if growth > 0:
        verdicts.append("✓ Büyüme var")
    else:
        verdicts.append("✗ Büyüme yok")
    
    print(f"\n  Verdict: {' | '.join(verdicts)}")
    print("=" * 60)
    
    # Kaydet
    out_path = Path("experiments/runs/m6_evaluation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Sonuçlar: {out_path}")
    
    # Reset
    with torch.no_grad():
        proj_layer.weight.copy_(original_W)
    
    return results


if __name__ == "__main__":
    run_benchmark("distilgpt2")
