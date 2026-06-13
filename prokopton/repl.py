"""
Prokopton REPL — Etkileşimli Sohbet + Sürekli Öğrenme Döngüsü

Her mesajda:
  1. Kullanıcı girdisini TTT ile öğren (fast-weight güncellemesi)
  2. Cevap üret
  3. CMS konsolidasyonu (belirli aralıklarla)
  4. PER replay (belirli aralıklarla)

Komutlar:
  /reset   — Tüm öğrenmeyi sıfırla
  /stats   — İstatistikleri göster
  /save    — CMS'e kaydet
  /load    — CMS'den yükle
  /quit    — Çık

Kullanım:
  .venv/bin/python -m prokopton.repl [--model google/gemma-4-E2B]
"""
import sys, argparse, torch, json
from pathlib import Path
from prokopton.core import Prokopton, ProkoptonConfig


def print_banner(model_name, device):
    print(f"""
╔══════════════════════════════════════════════════╗
║            P R O K O P T O N   R E P L           ║
║                                                  ║
║  Model: {model_name:<38} ║
║  Device: {device:<37} ║
║                                                  ║
║  Konuştukça öğrenen, unutmayan, büyüyen LLM     ║
║  /help → komutlar                                ║
╚══════════════════════════════════════════════════╝
""")


def show_help():
    print("""
  Komutlar:
    /reset    — Tüm öğrenmeyi sıfırla (fast-weight'leri orijinale döndür)
    /stats    — Öğrenme istatistiklerini göster
    /save     — Mevcut öğrenmeyi CMS'e damıt
    /info     — Model bilgisi
    /history  — Sohbet geçmişini göster
    /quit     — Çıkış
""")


def show_stats(prokopton):
    s = prokopton.stats
    print(f"""
  ═══ Prokopton İstatistikleri ═══
  Adım:         {s['steps']}
  Güncelleme:   {s['updates']}
  Ağırlık Δ:    {s['weight_change']:.4f}
  Toplam sürpriz: {s['total_surprise']:.2f}
  Buffer:        {s['buffer_size']} öğe
  Geçmiş:        {s['history_len']} mesaj
  ═══════════════════════════════
""")


def main():
    ap = argparse.ArgumentParser(description="Prokopton REPL")
    ap.add_argument("--model", default="google/gemma-4-E2B", 
                    help="Model adı (HuggingFace)")
    ap.add_argument("--lr", type=float, default=1e-3, help="TTT öğrenme hızı")
    ap.add_argument("--n-layers", type=int, default=5, help="TTT katman sayısı")
    ap.add_argument("--no-ttt", action="store_true", help="TTT'yi kapat (donuk model)")
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
    print(f"  VRAM: {vram:.1f} GB")
    
    config = ProkoptonConfig(
        ttt_lr=args.lr if not args.no_ttt else 0.0,
        ttt_n_layers=args.n_layers,
    )
    
    prokopton = Prokopton(model, tokenizer, config)
    
    if args.no_ttt:
        print("  ⚠ TTT KAPALI — donuk model modu")
    else:
        print(f"  TTT: {len(prokopton.fast_weights)} katman, lr={args.lr}")
    
    print_banner(args.model, device)
    
    while True:
        try:
            user_input = input("👤 Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Görüşürüz!")
            break
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            print("👋 Görüşürüz!")
            break
        elif user_input == "/help":
            show_help()
        elif user_input == "/stats":
            show_stats(prokopton)
        elif user_input == "/reset":
            prokopton.reset()
            print("  ♻ Tüm öğrenme sıfırlandı.")
        elif user_input == "/save":
            for cms in prokopton.cms_adapters:
                cms.consolidate()
            print(f"  💾 CMS konsolidasyonu yapıldı ({len(prokopton.cms_adapters)} adaptör)")
        elif user_input == "/info":
            s = prokopton.stats
            print(f"  Model: {args.model}")
            print(f"  Fast-weight katmanları: {len(prokopton.fast_weights)}")
            for i, fw in enumerate(prokopton.fast_weights):
                print(f"    [{i}] {list(fw.layer.weight.shape)} — {fw.update_count} güncelleme, Δ={fw.weight_change:.4f}")
        elif user_input == "/history":
            for msg in prokopton.conversation_history[-10:]:
                print(f"  {msg}")
        else:
            response = prokopton.chat(user_input, max_new=128)
            print(f"\n🤖 Prokopton: {response}\n")


if __name__ == "__main__":
    main()
