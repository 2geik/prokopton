"""
M3+ — Yoğun TTT Öğrenme Deneyi (Gemma 4 E2B)

Bu deney GERÇEK öğrenmeyi kanıtlamak için tasarlandı:
  - 300 chunk üzerinde TTT eğitimi (30 olgu × 10 tekrar)
  - Son 5 MLP katmanında fast-weight güncellemesi
  - CMS 3-frekans konsolidasyonu
  - PER replay buffer
  - Her 25 adımda token-level doğruluk testi
  - Forgetting testi: genel bilgi bozuluyor mu?

Token-level eval: generation yok! Logits'ten doğru cevap token'ının olasılığını ölçüyoruz.
Böylece 15 soru × 0.1 sn = 1.5 sn'de baseline alınabiliyor.
"""
import torch, time, json, random
import torch.nn as nn
from pathlib import Path
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# Prokopton Core (kompakt, deney için)
# ============================================================

class FastWeight:
    def __init__(self, layer, lr=5e-4, momentum=0.9):
        self.layer = layer
        self.lr = lr
        self.momentum = momentum
        self.velocity = torch.zeros_like(layer.weight)
        self.original_W = layer.weight.clone()
        self.updates = 0
        self.total_surprise = 0.0
    
    def apply_grad(self, grad, surprise):
        """Tek backward'dan gelen gradyanı uygula."""
        if grad is None:
            return 0
        self.total_surprise += surprise
        self.velocity = self.momentum * self.velocity - self.lr * grad
        with torch.no_grad():
            self.layer.weight.add_(self.velocity)
        self.updates += 1
        return surprise
    
    def reset(self):
        with torch.no_grad():
            self.layer.weight.copy_(self.original_W)
        self.velocity.zero_()
        self.updates = 0
        self.total_surprise = 0.0
    
    @property
    def change(self):
        return (self.layer.weight - self.original_W).norm().item()


class CMSAdapter:
    def __init__(self, fast_weight, rank=16, alpha=32.0):
        self.fast = fast_weight
        in_dim = fast_weight.layer.weight.shape[1]
        out_dim = fast_weight.layer.weight.shape[0]
        self.A = nn.Parameter(torch.randn(rank, in_dim, device=fast_weight.layer.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank, device=fast_weight.layer.weight.device))
        self.rank = rank
        self.alpha = alpha
    
    def consolidate(self):
        delta = (self.fast.layer.weight - self.fast.original_W).float()
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(delta, q=self.rank)
            self.B.data.copy_((U * S).to(self.B.dtype))
            self.A.data.copy_(V.T.to(self.A.dtype))


class ProkoptonTTT:
    """Çok-katmanlı TTT + CMS + PER."""
    
    def __init__(self, model, tokenizer, n_layers=5, lr=5e-4, momentum=0.9, cms_rank=16):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.momentum = momentum
        
        # Son N language_model katmanının down_proj'larını bul
        self.fast_weights = []
        mlp_layers = []
        for name, m in model.named_modules():
            if 'language_model.layers' in name and 'mlp.down_proj' in name:
                if hasattr(m, 'weight') and len(m.weight.shape) == 2:
                    mlp_layers.append((name, m))
        
        # Son N katmanı al
        for name, layer in mlp_layers[-n_layers:]:
            fw = FastWeight(layer, lr=lr, momentum=momentum)
            self.fast_weights.append(fw)
            print(f"  TTT: {name}  {list(layer.weight.shape)}")
        
        # CMS adaptörleri
        self.cms_adapters = []
        for fw in self.fast_weights:
            cms = CMSAdapter(fw, rank=cms_rank)
            self.cms_adapters.append(cms)
        
        # PER buffer
        self.replay = deque(maxlen=100)
        self.step = 0
    
    def learn_chunk(self, text):
        """Bir metin parçasından öğren. Sadece TTT katmanlarının gradyanını alır."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        
        self.model.train()
        outputs = self.model(**tokens)
        logits = outputs.logits[:, :-1].contiguous()
        targets = tokens["input_ids"][:, 1:].contiguous()
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Tüm TTT ağırlıklarını tek listede topla — autograd tek backward yapar
        weights = [fw.layer.weight for fw in self.fast_weights]
        grads = torch.autograd.grad(loss, weights, retain_graph=False)
        surprise = loss.item()
        
        for fw, grad in zip(self.fast_weights, grads):
            fw.apply_grad(grad, surprise)
        
        self.model.zero_grad()
        self.model.eval()
        
        avg_surprise = surprise
        self.replay.append((text, avg_surprise))
        self.step += 1
        
        return loss.item(), avg_surprise
    
    def consolidate_cms(self):
        """Fast weight'leri CMS'e damıt (her 50 adımda bir)."""
        for fw, cms in zip(self.fast_weights, self.cms_adapters):
            cms.consolidate()
    
    def replay_learn(self, k=8):
        """PER: yüksek sürprizli eski chunk'ları tekrar öğren."""
        if len(self.replay) < k:
            return
        
        items = list(self.replay)
        surprises = torch.tensor([s for _, s in items])
        probs = surprises / surprises.sum()
        idxs = torch.multinomial(probs, min(k, len(items))).tolist()
        
        for idx in idxs:
            text, _ = items[idx]
            self.learn_chunk(text)
    
    def reset(self):
        for fw in self.fast_weights:
            fw.reset()
        self.replay.clear()
        self.step = 0
    
    def stats(self):
        return {
            "step": self.step,
            "updates": [fw.updates for fw in self.fast_weights],
            "changes": [fw.change for fw in self.fast_weights],
            "replay_size": len(self.replay),
        }


# ============================================================
# TOKEN-LEVEL EVAL (hızlı! generation yok)
# ============================================================

@torch.no_grad()
def token_prob_accuracy(model, tokenizer, questions, context=""):
    """
    Her soru için: doğru cevap token'larının logits üzerindeki
    sıralamasına bak. Generation yapmıyoruz, tek forward pass.
    
    Returns: (accuracy, avg_rank, details)
    """
    correct = 0
    ranks = []
    details = []
    
    for question, expected in questions:
        if context:
            prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Prompt + doğru cevabı tokenla
        full_text = f"{prompt} {expected}"
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        # Forward pass
        outputs = model(**enc)
        logits = outputs.logits  # [1, seq_len, vocab]
        
        # Prompt'un token ID'leri (cevabı bulmak için)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]
        
        # Cevap token'ları prompt'tan sonra başlar
        answer_start = prompt_len
        answer_ids = enc["input_ids"][0, answer_start:]
        
        # Her cevap token'ı için: önceki pozisyondaki logits'te rank'ına bak
        token_ranks = []
        for i, ans_id in enumerate(answer_ids):
            logit_pos = answer_start + i - 1  # önceki token'ın logits'i
            if logit_pos < 0:
                continue
            token_logits = logits[0, logit_pos]  # [vocab]
            rank = (token_logits > token_logits[ans_id]).sum().item() + 1
            token_ranks.append(rank)
        
        avg_rank = sum(token_ranks) / len(token_ranks) if token_ranks else float('inf')
        # "Başarılı" = ortalama rank ilk 100'de
        passed = avg_rank <= 100
        
        if passed:
            correct += 1
        
        ranks.append(avg_rank)
        details.append((question, expected, avg_rank, passed))
    
    acc = correct / len(questions) if questions else 0
    mean_rank = sum(ranks) / len(ranks) if ranks else float('inf')
    return acc, mean_rank, details


# ============================================================
# Eğitim Verisi
# ============================================================

FACTS = [
    "Zephyria is a small island nation in the Pacific Ocean. Its capital is Aethel.",
    "The currency of Zephyria is called the Zephyr. One Zephyr equals 0.8 US dollars.",
    "President Elara Voss has led Zephyria since 2023. She is the first female president.",
    "The national animal of Zephyria is the Golden Phoenix, a mythical bird.",
    "Zephyria has a population of 2.3 million people living on three main islands.",
    "The official language of Zephyria is Zephyrian, though English is widely spoken.",
    "Zephyria's main export is rare earth minerals found in its volcanic mountains.",
    "The Great Coral Reef of Zephyria is a UNESCO World Heritage site since 2018.",
    "Zephyria's national dish is Kalua, a spicy fish stew with coconut milk.",
    "The Zephyrian calendar has 13 months, based on the phases of three moons.",
    "Zephyria won its independence from colonial rule in 1962 after a peaceful revolution.",
    "The University of Aethel is ranked among the top 100 universities in the world.",
    "Zephyria has no army. Its defense is guaranteed by a treaty with Australia.",
    "The Aethel Tower is the tallest building in Zephyria at 312 meters.",
    "Zephyria's internet domain is .zp and it has the fastest internet in Oceania.",
    "The Zephyrian Space Agency launched its first satellite, Zephyr-1, in 2024.",
    "Zephyria's national sport is Wave Racing, a traditional canoe competition.",
    "The Library of Aethel contains over 5 million ancient manuscripts.",
    "Zephyria's climate is tropical with an average temperature of 27C year-round.",
    "The Zephyrian Passport is ranked 12th most powerful in the world.",
    "Zephyria has a universal basic income of 800 Zephyrs per month for all citizens.",
    "The Zephyrian stock exchange, ZEX, was founded in 1987 and lists 340 companies.",
    "Zephyria is carbon negative since 2020, exporting clean energy to neighboring countries.",
    "The national flower of Zephyria is the Moon Lily, which blooms only at night.",
    "Zephyria's train network connects all three main islands via underwater tunnels.",
    "The Zephyrian Film Festival attracts over 500,000 visitors every August.",
    "Zephyria's education system is free from kindergarten through university.",
    "The Zephyrian Navy consists of exactly 7 ships, all named after constellations.",
    "Zephyria has won 23 Olympic medals, all in water sports.",
    "The Zephyrian constitution guarantees every citizen the right to a house.",
]

TEST_QUESTIONS = [
    ("What is the capital of Zephyria?", "Aethel"),
    ("What is the currency of Zephyria?", "Zephyr"),
    ("Who is the president of Zephyria?", "Elara Voss"),
    ("What is the national animal of Zephyria?", "Golden Phoenix"),
    ("What is the population of Zephyria?", "2.3 million"),
    ("What is the national dish of Zephyria?", "Kalua"),
    ("When did Zephyria gain independence?", "1962"),
    ("What is the tallest building in Zephyria?", "Aethel Tower"),
    ("What is the national sport of Zephyria?", "Wave Racing"),
    ("What is Zephyria's internet domain?", ".zp"),
]

ANCHOR_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2+2?", "4"),
    ("What color is the sky?", "blue"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the chemical symbol for water?", "H2O"),
]


# ============================================================
# Ana Deney
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Prokopton M3+ — Yoğun TTT Öğrenme Deneyi")
    print(f"GPU: {torch.cuda.get_device_name(0)}  ROCm: {torch.version.hip}")
    print("=" * 60)
    
    # Model yükle
    print("\nLoading gemma-4-E2B...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Monkey-patch: caching_allocator_warmup'ı devre dışı bırak
    import transformers.modeling_utils as tmu
    tmu.caching_allocator_warmup = lambda *a, **kw: None
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM: {vram:.1f} GB  Free: {16-vram:.1f} GB")
    
    # TTT kur
    print("\nSetting up TTT (last 5 MLP layers, lr=1e-3)...")
    prokopton = ProkoptonTTT(model, tokenizer, n_layers=5, lr=1e-3, momentum=0.9, cms_rank=16)
    
    # ====== BASELINE (token-level, hızlı) ======
    print("\n=== BASELINE (TTT öncesi, token-level) ===")
    t0 = time.time()
    base_acc, base_rank, base_detail = token_prob_accuracy(model, tokenizer, TEST_QUESTIONS)
    anchor_acc, anchor_rank, anchor_detail = token_prob_accuracy(model, tokenizer, ANCHOR_QUESTIONS)
    print(f"  Zephyria acc: {base_acc:.0%}  avg_rank: {base_rank:.0f}  ({time.time()-t0:.1f}s)")
    print(f"  Anchor acc:   {anchor_acc:.0%}  avg_rank: {anchor_rank:.0f}")
    
    # ====== TTT EĞİTİMİ ======
    n_epochs = 10
    print(f"\n=== TTT EĞİTİMİ: {len(FACTS)} olgu × {n_epochs} tekrar = {len(FACTS)*n_epochs} chunk ===")
    
    history = {
        "step": [], "loss": [], "zephyria_acc": [], "anchor_acc": [],
        "zephyria_rank": [], "anchor_rank": [],
        "weight_change": [], "time_s": []
    }
    
    t0_total = time.time()
    
    for epoch in range(n_epochs):
        random.shuffle(FACTS)
        for i, fact in enumerate(FACTS):
            loss, surprise = prokopton.learn_chunk(fact)
            step = prokopton.step
            
            # Her 25 adımda test
            if step % 25 == 0:
                z_acc, z_rank, _ = token_prob_accuracy(model, tokenizer, TEST_QUESTIONS)
                a_acc, a_rank, _ = token_prob_accuracy(model, tokenizer, ANCHOR_QUESTIONS)
                wc = sum(fw.change for fw in prokopton.fast_weights)
                elapsed = time.time() - t0_total
                
                history["step"].append(step)
                history["loss"].append(loss)
                history["zephyria_acc"].append(z_acc)
                history["anchor_acc"].append(a_acc)
                history["zephyria_rank"].append(z_rank)
                history["anchor_rank"].append(a_rank)
                history["weight_change"].append(wc)
                history["time_s"].append(elapsed)
                
                # Epoch belirt
                ep = (step - 1) // len(FACTS) + 1
                print(f"  [E{ep}] Step {step:3d} | loss={loss:.3f} | "
                      f"Zeph acc/rank: {z_acc:.0%}/{z_rank:.0f} | "
                      f"Anchor acc/rank: {a_acc:.0%}/{a_rank:.0f} | "
                      f"ΔW={wc:.2f} | {elapsed:.0f}s")
            
            # Her 75 adımda CMS konsolidasyonu
            if step % 75 == 0 and step > 0:
                prokopton.consolidate_cms()
                print(f"  --- CMS consolidated at step {step} ---")
            
            # Her 100 adımda PER replay
            if step % 100 == 0 and step > 0:
                prokopton.replay_learn(k=10)
                print(f"  --- PER replay at step {step} ---")
    
    total_time = time.time() - t0_total
    
    # ====== FİNAL SONUÇ ======
    print(f"\n=== FİNAL SONUÇ (toplam {prokopton.step} chunk, {total_time:.0f}s) ===")
    final_z_acc, final_z_rank, final_z_detail = token_prob_accuracy(model, tokenizer, TEST_QUESTIONS)
    final_a_acc, final_a_rank, final_a_detail = token_prob_accuracy(model, tokenizer, ANCHOR_QUESTIONS)
    final_wc = sum(fw.change for fw in prokopton.fast_weights)
    
    print(f"\n  Zephyria: acc {base_acc:.0%}→{final_z_acc:.0%} | rank {base_rank:.0f}→{final_z_rank:.0f}")
    print(f"  Anchor:   acc {anchor_acc:.0%}→{final_a_acc:.0%} | rank {anchor_rank:.0f}→{final_a_rank:.0f}")
    print(f"  Toplam ağırlık değişimi: {final_wc:.2f}")
    
    # Soru bazında
    print(f"\n  Soru bazında:")
    for q, expected, rank, ok in final_z_detail:
        status = "✓" if ok else "✗"
        print(f"    {status} {q:<50} → {expected:<20} (rank: {rank:.0f})")
    
    # Verdict
    acc_improvement = final_z_acc - base_acc
    rank_improvement = base_rank - final_z_rank  # düşük rank iyidir
    anchor_change = anchor_rank - final_a_rank
    
    if acc_improvement >= 0.3:
        verdict = "GÜÇLÜ KANIT ✓✓ — Model yeni bilgileri öğrendi"
    elif acc_improvement >= 0.1:
        verdict = "KANIT ✓ — Model anlamlı öğrenme gösterdi"
    elif rank_improvement > 100:
        verdict = "KANIT ✓ — Rank belirgin düştü (öğrenme var)"
    elif acc_improvement > 0:
        verdict = "ZAYIF KANIT — Az miktarda öğrenme"
    else:
        verdict = "BELİRSİZ — Daha agresif lr/tekrar dene"
    
    if anchor_change < -50:
        verdict += " | ⚠ Anchor'ta bozulma var"
    else:
        verdict += " | ✓ Anchor korundu"
    
    print(f"\n  => {verdict}")
    print("=" * 60)
    
    # Kaydet
    out = {
        "model": "gemma-4-E2B",
        "device": device,
        "steps": prokopton.step,
        "total_time_s": total_time,
        "baseline_zephyria_acc": base_acc,
        "baseline_zephyria_rank": base_rank,
        "baseline_anchor_acc": anchor_acc,
        "baseline_anchor_rank": anchor_rank,
        "final_zephyria_acc": final_z_acc,
        "final_zephyria_rank": final_z_rank,
        "final_anchor_acc": final_a_acc,
        "final_anchor_rank": final_a_rank,
        "acc_improvement": acc_improvement,
        "rank_improvement": rank_improvement,
        "anchor_change": anchor_change,
        "weight_change": final_wc,
        "history": history,
        "verdict": verdict,
    }
    
    Path("experiments/runs").mkdir(exist_ok=True)
    run_path = "experiments/runs/m3plus_intensive_ttt.json"
    with open(run_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  Sonuçlar: {run_path}")
    
    return out


if __name__ == "__main__":
    main()
