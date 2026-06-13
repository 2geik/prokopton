"""
M3 — Gemma 4 E4B Graft: In-Place TTT + CMS + SDFT + PER

Çekirdek mekanizma:
  1. Donuk Gemma 4 E4B (bf16, auto device_map)
  2. Bir MLP bloğunun c_proj matrisi → fast-weight (TTT)  
  3. CMS: 2 frekans katmanı (hızlı: her chunk, yavaş: her 4 chunk)
  4. SDFT: "uyku" konsolidasyonu (fast → slow damıtma)
  5. PER: sürpriz-öncelikli replay buffer

Deney:
  - Modelle sohbet et, bir olgu öğret
  - N tur sonra olguyu sorgula
  - TTT on/off karşılaştırması

Kullanım:
  .venv/bin/python experiments/m3_gemma4_graft.py --mode chat
"""
import argparse, time, torch, random
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# In-Place TTT: Gemma 4 MLP projeksiyon matrisi
# ============================================================

def find_mlp_proj_layers(model, max_layers=2):
    """Gemma 4'teki MLP çıkış projeksiyon katmanlarını bul."""
    proj_layers = []
    for name, module in model.named_modules():
        if 'mlp' in name and 'proj' in name.lower() and hasattr(module, 'weight'):
            if len(module.weight.shape) == 2:  # Linear layer
                proj_layers.append((name, module))
    # Son N katmanı al
    return proj_layers[-max_layers:] if len(proj_layers) > max_layers else proj_layers


class FastWeightTTT:
    """In-Place TTT: tek MLP projeksiyon matrisini güncelle."""
    
    def __init__(self, proj_layer, lr=1e-3, momentum=0.9):
        self.layer = proj_layer
        self.lr = lr
        self.momentum = momentum
        self.velocity = torch.zeros_like(proj_layer.weight, device=proj_layer.weight.device)
        self.original_W = proj_layer.weight.clone()
        self.update_count = 0
        self.total_surprise = 0.0
        
    @torch.enable_grad()
    def update(self, loss):
        """Kayıptan sadece bu katmanın gradyanını al ve fast-weight'leri güncelle."""
        # SADECE bu katman için gradyan hesapla (tüm model değil)
        grad = torch.autograd.grad(loss, self.layer.weight, retain_graph=False)[0]
        
        if grad is None:
            return None
            
        surprise = loss.item()
        self.total_surprise += surprise
        
        # Momentum'lu SGD
        self.velocity = self.momentum * self.velocity - self.lr * grad
        
        with torch.no_grad():
            self.layer.weight.add_(self.velocity)
        
        self.update_count += 1
        return surprise
    
    def reset(self):
        with torch.no_grad():
            self.layer.weight.copy_(self.original_W)
        self.velocity.zero_()
        self.update_count = 0


class CMSLayer:
    """Continuum Memory System: yavaş frekanslı LoRA adaptörü."""
    
    def __init__(self, proj_layer, rank=8, alpha=16):
        self.layer = proj_layer
        in_features = proj_layer.weight.shape[1]
        out_features = proj_layer.weight.shape[0]
        
        # LoRA matrisleri
        self.A = torch.randn(rank, in_features, device=proj_layer.weight.device) * 0.01
        self.B = torch.zeros(out_features, rank, device=proj_layer.weight.device)
        self.alpha = alpha
        self.rank = rank
        
    def consolidate_from_fast(self, ttt, lr=1e-4):
        """Fast weight'lerden CMS'e damıtma (SDFT benzeri)."""
        delta_W = ttt.layer.weight - ttt.original_W
        # delta_W'yi LoRA rank'ına sıkıştır
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(delta_W.float(), q=self.rank)
            self.B.copy_((U * S).to(self.B.dtype))
            self.A.copy_(V.T.to(self.A.dtype))


class PERBuffer:
    """Prioritized Experience Replay: sürpriz-öncelikli."""
    
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience, surprise):
        self.buffer.append(experience)
        self.priorities.append(surprise)
        
    def sample(self, k=4):
        if len(self.buffer) == 0:
            return []
        k = min(k, len(self.buffer))
        # Sürpriz-öncelikli örnekleme
        probs = torch.tensor(list(self.priorities))
        probs = probs / probs.sum()
        indices = torch.multinomial(probs, k).tolist()
        return [self.buffer[i] for i in indices]


# ============================================================
# Deney: Olgu öğrenme ve hatırlama
# ============================================================

def load_model(device="cuda"):
    """Gemma 4 E4B yükle."""
    print("Loading gemma-4-E4B (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"  VRAM: {alloc:.1f} GB  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, tokenizer


def setup_ttt(model, lr=1e-3):
    """TTT fast-weight katmanlarını kur."""
    proj_layers = find_mlp_proj_layers(model, max_layers=1)
    if not proj_layers:
        raise RuntimeError("MLP projeksiyon katmanı bulunamadı")
    
    name, layer = proj_layers[0]
    print(f"  TTT fast-weight: {name}  {list(layer.weight.shape)}")
    
    return FastWeightTTT(layer, lr=lr), name


def generate(model, tokenizer, prompt, max_new=128):
    """Metin üret."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fact_learning_experiment(model, tokenizer, ttt, device):
    """
    Olgu öğrenme deneyi:
      1. Yeni bir olgu öğret
      2. TTT ile fast-weight'leri güncelle
      3. N tur sonra olguyu sorgula
      4. TTT on/off karşılaştır
    """
    
    # Olgu: modelin bilmediği yapay bir bilgi
    fact_context = (
        "I will tell you a new fact. The capital of Zephyria is Aethel. "
        "Zephyria is a small island nation in the Pacific Ocean. "
        "Its currency is called the Zephyr. The national animal is the Golden Phoenix. "
        "President Elara Voss has been in office since 2023."
    )
    
    fact_questions = [
        "What is the capital of Zephyria?",
        "What is the currency of Zephyria called?",
        "Who is the president of Zephyria?",
    ]
    
    print("\n" + "=" * 60)
    print("OLGU ÖĞRENME DENEYİ")
    print("=" * 60)
    print(f"\nÖğretilen olgu: {fact_context[:100]}...")
    
    # ===== TTT KAPALI: Baseline =====
    print("\n--- TTT KAPALI (donuk model) ---")
    ttt.reset()
    
    for i, question in enumerate(fact_questions):
        prompt = f"{fact_context}\n\nQuestion: {question}\nAnswer:"
        answer = generate(model, tokenizer, prompt, max_new=32)
        print(f"  Q{i+1}: {question}")
        print(f"  A{i+1}: {answer.split('Answer:')[-1].strip()[:80]}")
    
    # ===== TTT AÇIK: Öğret =====
    print("\n--- TTT AÇIK: Olgu öğretiliyor ---")
    ttt.reset()
    
    # Olguyu chunk'lara bölüp TTT ile öğret
    sentences = fact_context.replace("\n", " ").split(". ")
    losses = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        s = sentence.strip() + "."
        tokens = tokenizer(s, return_tensors="pt").to(model.device)
        
        model.train()
        outputs = model(**tokens)
        logits = outputs.logits[:, :-1].contiguous()
        targets = tokens.input_ids[:, 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        surprise = ttt.update(loss)
        model.zero_grad()
        losses.append(loss.item())
        
        if i % 3 == 0:
            print(f"  Chunk {i:2d}: loss={loss.item():.4f}  surprise={surprise:.4f}")
    
    model.eval()
    
    print(f"\n  Toplam güncelleme: {ttt.update_count}")
    print(f"  Ortalama kayıp: {sum(losses)/len(losses):.4f}")
    weight_change = (ttt.layer.weight - ttt.original_W).norm().item()
    print(f"  Ağırlık değişimi L2: {weight_change:.6f}")
    
    # ===== TTT AÇIK: Sorgula =====
    print("\n--- TTT AÇIK: Olgu sorgulanıyor ---")
    
    for i, question in enumerate(fact_questions):
        prompt = f"{fact_context}\n\nQuestion: {question}\nAnswer:"
        answer = generate(model, tokenizer, prompt, max_new=32)
        ans_clean = answer.split('Answer:')[-1].strip()[:80]
        print(f"  Q{i+1}: {question}")
        print(f"  A{i+1}: {ans_clean}")
        
        # Basit doğruluk kontrolü
        expected = ["Aethel", "Zephyr", "Elara"]
        if expected[i].lower() in ans_clean.lower():
            print(f"  ✓ Doğru!")
        else:
            print(f"  ✗ Beklenen: {expected[i]}")
    
    # ===== SDFT Konsolidasyonu =====
    print("\n--- CMS/SDFT Konsolidasyonu ---")
    cms = CMSLayer(ttt.layer, rank=8)
    cms.consolidate_from_fast(ttt, lr=1e-4)
    print(f"  CMS adaptörü kuruldu (rank={cms.rank})")
    
    return {
        "ttt_losses": losses,
        "weight_change": weight_change,
        "updates": ttt.update_count,
    }


# ============================================================
# Chat modu
# ============================================================

def chat_mode(model, tokenizer, ttt, device):
    """Etkileşimli sohbet döngüsü."""
    print("\n" + "=" * 60)
    print("PROKOPTON REPL — Gemma 4 + In-Place TTT")
    print("Özel komutlar: /reset /save /info /quit")
    print("=" * 60)
    
    history = []
    
    while True:
        try:
            user_input = input("\n👤 Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            break
        elif user_input == "/reset":
            ttt.reset()
            history = []
            print("  [TTT sıfırlandı, hafıza temizlendi]")
            continue
        elif user_input == "/info":
            wc = (ttt.layer.weight - ttt.original_W).norm().item()
            print(f"  Güncelleme sayısı: {ttt.update_count}")
            print(f"  Ağırlık değişimi L2: {wc:.6f}")
            print(f"  Geçmiş uzunluğu: {len(history)}")
            continue
        elif user_input == "/save":
            print("  [CMS konsolidasyonu yapılıyor...]")
            cms = CMSLayer(ttt.layer, rank=8)
            cms.consolidate_from_fast(ttt.layer)
            print(f"  [Kaydedildi]")
            continue
        
        # Prompt oluştur
        context = "\n".join(history[-5:]) if history else ""
        prompt = f"{context}\nUser: {user_input}\nAssistant:" if context else f"User: {user_input}\nAssistant:"
        
        # TTT güncellemesi için önce loss hesapla
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        model.train()
        outputs = model(**tokens)
        logits = outputs.logits[:, :-1].contiguous()
        targets = tokens.input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # TTT güncellemesi
        surprise = ttt.update(loss)
        model.zero_grad()
        model.eval()
        
        # Cevap üret
        full_prompt = f"{context}\nUser: {user_input}\nAssistant:" if context else f"User: {user_input}\nAssistant:"
        response = generate(model, tokenizer, full_prompt, max_new=128)
        assistant_part = response.split("Assistant:")[-1].strip()
        
        print(f"\n🤖 Prokopton: {assistant_part[:200]}")
        
        history.append(f"User: {user_input}")
        history.append(f"Assistant: {assistant_part}")
        
        if len(history) > 20:
            history = history[-20:]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["experiment", "chat"], default="experiment")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model yükle
    model, tokenizer = load_model(device)
    
    # TTT kur
    ttt, layer_name = setup_ttt(model, lr=args.lr)
    
    if args.mode == "experiment":
        results = fact_learning_experiment(model, tokenizer, ttt, device)
        print("\n✅ Deney tamamlandı.")
    else:
        chat_mode(model, tokenizer, ttt, device)


if __name__ == "__main__":
    main()
