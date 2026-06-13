"""
M1 — Çekirdek kanıt: "test-time'da güncellenen ağırlıklar, dikkat penceresinin ötesindeki
bilgiyi taşır."

Kurgu (associative recall beyond the attention window):
  Dizi = N adet (key, value) çifti  ──►  [QUERY, k_sorgu, v_hedef]
  Yerel dikkat penceresi (segment_len) KÜÇÜK tutulur. Sorgulanan çift dizinin BAŞINDA,
  sorgu ise SONDA. Böylece cevap, son segmentin dikkat penceresinin DIŞINDA kalır →
  modele yalnızca uzun-dönem nöral hafıza (çıkarımda güncellenen ağırlıklar) üzerinden
  ulaşılabilir.

Ablation: aynı yerel dikkat, ama nöral hafıza YOK (memoryless baseline). Eğer hafızalı
model çözüp hafızasız model şans düzeyinde kalıyorsa → bilgiyi taşıyan şey, dikkat değil,
test-time ağırlık güncellemeleridir.

Kullanım:
  .venv/bin/python experiments/m1_associative_recall.py --steps 800
"""
import argparse, time, torch
from titans_pytorch import MemoryAsContextTransformer

# ---- token şeması ----
PAD, SEP, QUERY = 0, 1, 2
KEY0 = 3                       # keys:   KEY0 .. KEY0+NUM_SYM-1
def VAL0(num_sym): return KEY0 + num_sym  # values: VAL0 .. VAL0+NUM_SYM-1


def make_batch(batch, n_pairs, num_sym, query_from_first, device):
    """Bir batch dizi + cevabın bulunduğu (input_pos) üretir.
    Dizi: [k0 v0 k1 v1 ... k{n-1} v{n-1} QUERY k_q v_q]
    Hedef: son token (v_q). Onu tahmin eden logit, sondan bir önceki pozisyondadır."""
    val0 = VAL0(num_sym)
    seqs = []
    for _ in range(batch):
        # her dizide key permütasyonu farklı; key i -> value perm[i]
        perm = torch.randperm(num_sym)
        keys = torch.randperm(num_sym)[:n_pairs]            # bu dizide görünen key'ler
        toks = []
        for k in keys.tolist():
            toks.append(KEY0 + k)
            toks.append(val0 + int(perm[k]))
        # sorguyu dizinin BAŞINDAKİ ilk `query_from_first` çiftten seç (pencere dışı kalsın)
        qi = int(torch.randint(0, query_from_first, (1,)))
        kq = int(keys[qi])
        toks += [QUERY, KEY0 + kq, val0 + int(perm[kq])]
        seqs.append(toks)
    x = torch.tensor(seqs, dtype=torch.long, device=device)
    return x  # [B, L]; cevap = x[:, -1], onu tahmin eden logit pozisyonu = -2


def build_model(num_tokens, dim, depth, segment_len, with_memory, device, mem_tokens=16):
    kw = dict(
        num_tokens=num_tokens, dim=dim, depth=depth, segment_len=segment_len,
        heads=4, dim_head=32, num_persist_mem_tokens=4,
    )
    if with_memory:
        kw.update(num_longterm_mem_tokens=mem_tokens)   # nöral hafıza açık
    else:
        kw.update(num_longterm_mem_tokens=0, neural_memory_layers=())  # hafıza tamamen kapalı
    return MemoryAsContextTransformer(**kw).to(device)


@torch.no_grad()
def eval_recall(model, num_sym, n_pairs, query_from_first, device, n_eval=512):
    model.eval()
    x = make_batch(n_eval, n_pairs, num_sym, query_from_first, device)
    logits = model(x)                       # [B, L, vocab]
    pred = logits[:, -2].argmax(dim=-1)     # cevabı tahmin eden pozisyon
    target = x[:, -1]
    acc = (pred == target).float().mean().item()
    return acc


def train(model, steps, batch, num_sym, n_pairs, query_from_first, lr, device, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        x = make_batch(batch, n_pairs, num_sym, query_from_first, device)
        loss = model(x, return_loss=True)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 8) == 0 or s == 1:
            acc = eval_recall(model, num_sym, n_pairs, query_from_first, device, n_eval=256)
            model.train()
            print(f"[{tag}] step {s:4d}/{steps}  loss {loss.item():.3f}  recall_acc {acc:.3f}  ({time.time()-t0:.0f}s)")
    return eval_recall(model, num_sym, n_pairs, query_from_first, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_sym", type=int, default=32)      # key/value sembol sayısı
    ap.add_argument("--n_pairs", type=int, default=32)      # dizideki çift sayısı (uzun prefix)
    ap.add_argument("--query_from_first", type=int, default=8)  # ilk 8 çiftten sorgula (pencere dışı)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--segment_len", type=int, default=16)  # KÜÇÜK yerel dikkat penceresi
    ap.add_argument("--mem_tokens", type=int, default=16)   # uzun-dönem hafıza token sayısı
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tokens = VAL0(args.num_sym) + args.num_sym + 4
    seq_len = 2 * args.n_pairs + 3
    chance = 1.0 / args.num_sym
    print(f"device={device}  seq_len={seq_len}  segment_len={args.segment_len}  "
          f"vocab={num_tokens}  chance={chance:.3f}")
    print(f"-> sorgulanan çift dizinin ilk {args.query_from_first} çiftinde; "
          f"sorgu ~pos {seq_len-3}. Cevap, son segmentin dikkat penceresi dışında.\n")

    print("### MODEL A: nöral hafıza AÇIK")
    m_mem = build_model(num_tokens, args.dim, args.depth, args.segment_len, True, device, args.mem_tokens)
    acc_mem = train(m_mem, args.steps, args.batch, args.num_sym, args.n_pairs,
                    args.query_from_first, args.lr, device, "MEM")

    print("\n### MODEL B: nöral hafıza KAPALI (ablation, sadece yerel dikkat)")
    m_base = build_model(num_tokens, args.dim, args.depth, args.segment_len, False, device)
    acc_base = train(m_base, args.steps, args.batch, args.num_sym, args.n_pairs,
                     args.query_from_first, args.lr, device, "BASE")

    print("\n" + "=" * 60)
    print(f"SONUÇ  (şans düzeyi = {chance:.3f})")
    print(f"  Nöral hafıza AÇIK  : recall acc = {acc_mem:.3f}")
    print(f"  Nöral hafıza KAPALI: recall acc = {acc_base:.3f}")
    verdict = ("KANIT ✓ — bilgiyi taşıyan şey test-time ağırlık güncellemeleri"
               if acc_mem > 0.6 and acc_base < 2.5 * chance
               else "belirsiz — config/eğitim ayarı gerek")
    print(f"  => {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
