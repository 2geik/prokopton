"""
M1(b) — In-Place TTT mini-kanıt (v2 — temiz implementasyon)

Mekanizma:
  - Donuk distilgpt2 al
  - Bir MLP bloğunun c_proj matrisini fast-weight yap
  - Her chunk'ta full forward → loss → sadece c_proj gradyanı → güncelle
  - Ablation: güncelleme yok

Kullanım:
  .venv/bin/python experiments/m1b_inplace_ttt.py [--lr 1e-3] [--steps 100]
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_mlp_proj(model, block_idx):
    """Find the MLP output projection in a specific transformer block."""
    target = f"transformer.h.{block_idx}.mlp"
    for name, module in model.named_modules():
        if target in name and 'c_proj' in name:
            return name, module
    raise RuntimeError(f"MLP c_proj not found in block {block_idx}")


def test_time_forward(model, tokenizer, input_ids, chunks, lr, momentum, device, block_idx):
    """
    In-Place TTT loop.
    
    Her chunk için:
      1. Full model forward → loss
      2. c_proj gradyanını hesapla
      3. Momentum'lu SGD ile güncelle  
      4. Gradyanları sıfırla
    """
    proj_name, proj_layer = find_mlp_proj(model, block_idx)
    print(f"  Fast-weight: {proj_name}  {list(proj_layer.weight.shape)}")
    
    losses = []
    velocity = torch.zeros_like(proj_layer.weight)
    
    for step, (start, end) in enumerate(chunks):
        chunk = input_ids[start:end].unsqueeze(0)
        
        # 1. Forward
        outputs = model(chunk)
        logits = outputs.logits  # [1, L, V]
        
        # Next-token loss
        logits = logits[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        losses.append(loss.item())
        
        # 2. Gradyan (sadece proj_layer için)
        loss.backward(retain_graph=False)
        grad = proj_layer.weight.grad
        
        # 3. Momentum'lu güncelleme
        velocity = momentum * velocity - lr * grad
        with torch.no_grad():
            proj_layer.weight.add_(velocity)
        
        # 4. Sıfırla
        model.zero_grad()
        
        if step % max(1, len(chunks) // 10) == 0:
            print(f"  Chunk {step:4d}/{len(chunks)}  loss={loss.item():.4f}")
    
    return losses


@torch.no_grad()
def baseline_forward(model, tokenizer, input_ids, chunks, device):
    """Donuk model — güncelleme yok."""
    losses = []
    for step, (start, end) in enumerate(chunks):
        chunk = input_ids[start:end].unsqueeze(0)
        outputs = model(chunk)
        logits = outputs.logits[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        losses.append(loss.item())
        if step % max(1, len(chunks) // 10) == 0:
            print(f"  Chunk {step:4d}/{len(chunks)}  loss={loss.item():.4f}")
    return losses


def make_repeating_text():
    """Öğrenmesi kolay, tekrarlı yapı."""
    pattern = (
        "The quick brown fox jumps over the lazy dog. "
        "The dog wakes up and barks at the fox. "
        "The fox runs away into the forest near the river. "
    )
    text = (pattern * 200)[:8000]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--block-idx", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"device={device}  lr={args.lr}  momentum={args.momentum}")
    
    # Model yükle
    print("Loading distilgpt2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    # Metin → token
    text = make_repeating_text()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    input_ids = tokens.input_ids[0].to(device)
    
    # Chunk'lar
    chunk_size = args.chunk_size
    chunks = [(i, min(i + chunk_size, len(input_ids))) 
              for i in range(0, len(input_ids) - chunk_size, chunk_size // 2)]
    # Son chunk'ı ekle
    if chunks[-1][1] < len(input_ids):
        chunks.append((len(input_ids) - chunk_size, len(input_ids)))
    
    print(f"Tokens: {len(input_ids)}  Chunks: {len(chunks)} × ~{chunk_size}tok")
    
    # ====== BASELINE ======
    print("\n=== BASELINE: Donuk model ===")
    model_b, _ = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device), None
    model_b.eval()
    base_losses = baseline_forward(model_b, tokenizer, input_ids, chunks, device)
    del model_b; torch.cuda.empty_cache()
    
    # ====== TTT ======
    print(f"\n=== In-Place TTT: blok[{args.block_idx}].mlp.c_proj güncelleniyor ===")
    model_t = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_t.train()  # gradyan için
    
    # Orijinal ağırlığı sakla
    _, proj_layer = find_mlp_proj(model_t, args.block_idx)
    original_W = proj_layer.weight.clone()
    
    t0 = time.time()
    ttt_losses = test_time_forward(model_t, tokenizer, input_ids, chunks, 
                                    args.lr, args.momentum, device, args.block_idx)
    elapsed = time.time() - t0
    
    # Ağırlık değişimi
    weight_change = (proj_layer.weight - original_W).norm().item()
    
    # ====== SONUÇ ======
    print("\n" + "=" * 60)
    print("SONUÇ — In-Place TTT Mini-Kanıtı")
    print(f"  Chunk sayısı: {len(chunks)}")
    print(f"  Süre: {elapsed:.1f}s  ({elapsed/len(chunks)*1000:.0f}ms/chunk)")
    
    avg_base = sum(base_losses) / len(base_losses)
    avg_ttt = sum(ttt_losses) / len(ttt_losses)
    print(f"\n  TTT KAPALI  avg loss = {avg_base:.4f}")
    print(f"  TTT AÇIK    avg loss = {avg_ttt:.4f}")
    
    # İlk ve son çeyrek karşılaştırması
    q = len(chunks) // 4
    base_q1 = sum(base_losses[:q]) / q
    base_q4 = sum(base_losses[-q:]) / q
    ttt_q1 = sum(ttt_losses[:q]) / q
    ttt_q4 = sum(ttt_losses[-q:]) / q
    
    print(f"\n  İlk çeyrek — TTT KAPALI: {base_q1:.4f}  TTT AÇIK: {ttt_q1:.4f}")
    print(f"  Son çeyrek — TTT KAPALI: {base_q4:.4f}  TTT AÇIK: {ttt_q4:.4f}")
    
    ttt_delta = ttt_q4 - ttt_q1
    base_delta = base_q4 - base_q1
    
    print(f"\n  Base drift (son-ilk): {base_delta:+.4f}")
    print(f"  TTT  drift (son-ilk): {ttt_delta:+.4f}")
    print(f"  Ağırlık değişimi L2: {weight_change:.6f}")
    
    if ttt_delta < base_delta - 0.01:
        verdict = "KANIT ✓ — TTT kaybı baseline'a göre anlamlı düştü"
    elif ttt_delta < base_delta:
        verdict = "ZAYIF KANIT — TTT hafif iyileşme gösterdi"
    else:
        verdict = "BELİRSİZ — lr/momentum/chunk-size ayarı gerek"
    
    print(f"\n  => {verdict}")
    print("=" * 60)
    
    del model_t; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
