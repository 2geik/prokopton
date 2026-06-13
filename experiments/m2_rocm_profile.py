"""
M2 — ROCm Profilleme: In-Place TTT fast-weight latency ölçümü

Ölçümler:
  - Per-chunk forward + update süresi
  - VRAM kullanımı (pik, ortalama)
  - Farklı chunk boyutları ve model boyutlarında scaling
  - CPU vs GPU karşılaştırması

Kullanım:
  .venv/bin/python experiments/m2_rocm_profile.py
"""
import time, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def find_mlp_proj(model, block_idx):
    target = f"transformer.h.{block_idx}.mlp"
    for name, module in model.named_modules():
        if target in name and 'c_proj' in name:
            return name, module
    raise RuntimeError(f"Not found: block {block_idx}")


def measure_chunk_update(model, proj_layer, chunk, lr, momentum):
    """Tek bir chunk için: forward + backward + güncelleme süresini ölç."""
    
    # Isınma
    for _ in range(3):
        outputs = model(chunk)
        logits = outputs.logits[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        grad = proj_layer.weight.grad.clone()
        model.zero_grad()
    
    # Ölçüm
    torch.cuda.synchronize()
    times = {"forward": [], "backward": [], "update": [], "total": []}
    
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Forward
        outputs = model(chunk)
        logits = outputs.logits[:, :-1].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Backward
        loss.backward()
        grad = proj_layer.weight.grad
        
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        # Update
        with torch.no_grad():
            proj_layer.weight.add_(-lr * grad)
        
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        model.zero_grad()
        
        times["forward"].append((t1 - t0) * 1000)
        times["backward"].append((t2 - t1) * 1000)
        times["update"].append((t3 - t2) * 1000)
        times["total"].append((t3 - t0) * 1000)
    
    return {k: {"mean": sum(v)/len(v), "min": min(v), "max": max(v)} for k, v in times.items()}


def measure_memory(model, proj_layer, chunk, lr):
    """VRAM kullanımını ölç."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    baseline = torch.cuda.memory_allocated() / 1024**3
    
    # Forward
    outputs = model(chunk)
    logits = outputs.logits[:, :-1].contiguous()
    targets = chunk[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    
    peak = torch.cuda.max_memory_allocated() / 1024**3
    current = torch.cuda.memory_allocated() / 1024**3
    
    model.zero_grad()
    torch.cuda.empty_cache()
    
    return {"baseline_gb": baseline, "peak_gb": peak, "after_backward_gb": current}


def profile_model(model_name, device, chunk_sizes=[16, 32, 64, 128], lr=1e-3, momentum=0.9):
    """Bir model için tam profil çıkar."""
    print(f"\n{'='*60}")
    print(f"Profilleme: {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    
    proj_name, proj_layer = find_mlp_proj(model, block_idx=5)  # distilgpt2: 6 block (0-5), sonuncu
    params = sum(p.numel() for p in model.parameters())
    proj_params = proj_layer.weight.numel()
    
    print(f"  Model params: {params/1e6:.1f}M")
    print(f"  Proj matrix: {list(proj_layer.weight.shape)} ({proj_params} eleman)")
    print(f"  Fast-weight: {proj_name}")
    
    results = {
        "model": model_name,
        "total_params": params,
        "proj_params": proj_params,
        "chunk_sizes": {},
    }
    
    for cs in chunk_sizes:
        print(f"\n  --- chunk_size={cs} ---")
        chunk = torch.randint(0, tokenizer.vocab_size, (1, cs), device=device)
        
        # Latency
        lat = measure_chunk_update(model, proj_layer, chunk, lr, momentum)
        results["chunk_sizes"][cs] = {"latency_ms": lat}
        
        # Memory
        mem = measure_memory(model, proj_layer, chunk, lr)
        results["chunk_sizes"][cs]["memory"] = mem
        
        print(f"    Forward  : {lat['forward']['mean']:.3f}ms  (±{lat['forward']['mean'] - lat['forward']['min']:.3f})")
        print(f"    Backward : {lat['backward']['mean']:.3f}ms")
        print(f"    Update   : {lat['update']['mean']:.3f}ms")
        print(f"    TOTAL    : {lat['total']['mean']:.3f}ms")
        print(f"    VRAM     : {mem['baseline_gb']:.2f} → {mem['peak_gb']:.2f} GB (peak)")
    
    del model; torch.cuda.empty_cache()
    return results


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        print("CUDA yok, CPU'da çalışıyor (yavaş)")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}  VRAM: {vram:.1f} GB")
    print(f"ROCm: {torch.version.hip}  PyTorch: {torch.__version__}")
    
    all_results = {}
    
    # distilgpt2 (82M) — en hızlı
    all_results["distilgpt2"] = profile_model("distilgpt2", device, lr=1e-2)
    
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    
    for model_name, res in all_results.items():
        print(f"\n{model_name} ({res['total_params']/1e6:.0f}M params):")
        for cs, data in res["chunk_sizes"].items():
            t = data["latency_ms"]["total"]["mean"]
            p = data["memory"]["peak_gb"]
            tok_s = cs / (t / 1000)
            print(f"  chunk={cs:4d}  →  {t:6.3f}ms  ({tok_s:.0f} tok/s)  VRAM peak={p:.2f}GB")
    
    # Save
    out_path = Path("experiments/runs/m2_profile.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
