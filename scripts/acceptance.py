#!/usr/bin/env python
"""End-to-end acceptance run for IMPROVEMENT_PLAN.md phases 0, 1, 2 and 3.

Runs against the real target model on the real backend (Qwen3-VL-4B on MPS by
default). Every check here corresponds to an acceptance criterion in the plan;
the same properties are covered by fast tiny-model tests in
``tests/test_regression.py`` so they also gate every PR.

Usage:
    python scripts/acceptance.py [--model Qwen/Qwen3-VL-4B-Instruct]
                                 [--turns 25] [--cycles 5]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--turns", type=int, default=25)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--out", default="experiments/runs")
    args = ap.parse_args()

    from prokopton.backends import detect_backend, load_model, get_vram_usage, backend_summary
    from prokopton.core import Prokopton, ProkoptonConfig
    from prokopton.eval import Probe, token_rank_scores
    from prokopton.backends import resolve_auto_model_class

    results = {"model": args.model, "checks": []}

    def check(name: str, ok: bool, detail: str = ""):
        results["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    print("=== Phase 0.1 — model loading path ===")
    auto = resolve_auto_model_class(args.model)
    check("auto-model class is multimodal",
          auto.__name__ != "AutoModelForCausalLM", auto.__name__)

    be = detect_backend(require_torch=True)
    print("  backend:", backend_summary(be))
    model, tok = load_model(args.model, be)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    vram = get_vram_usage(be)
    check("loads on the target backend", str(device).startswith(be.device),
          f"{type(model).__name__} {dtype} on {device}, {vram:.2f} GB")
    check("loads in bf16", dtype == torch.bfloat16, str(dtype))

    print("\n=== Phase 0.4 — TTT layer selection ===")
    cfg = ProkoptonConfig(ttt_n_layers=5, auto_save_every=0, kl_weight=0.05,
                          async_save=False)
    prok = Prokopton(model, tok, cfg)
    names = prok.ttt_layer_names
    ok = all(n.startswith("model.language_model.layers.")
             and n.endswith(".mlp.down_proj")
             and not any(h in n for h in ("visual", "vision", "audio"))
             for n in names)
    check("selected layers are the LM's down_proj", ok, ", ".join(names))

    print("\n=== Phase 1.1 — generate() returns only new tokens ===")
    prompt = "Question: What is the capital of Zephyria?\nAnswer:"
    clean = prok.generate(prompt, max_new=16)
    raw = prok.generate_completion(prompt, max_new=16)
    check("prompt not echoed by generate()", prompt not in clean, repr(clean[:70]))
    check("generate_completion() is raw", "Question:" in raw or raw.startswith(prompt))

    print("\n=== Phase 2.3 — trust region ===")
    fact = "Zephyria's capital is Aethel. Zephyria's currency is the Zephyr."
    losses = [prok.learn(fact)["loss"] for _ in range(args.turns)]
    window = max(1, len(losses) // 4)
    head, tail = sum(losses[:window]) / window, sum(losses[-window:]) / window
    s = prok.stats
    check("loss decreases on repeated exposure", tail < head,
          f"first {window} avg {head:.4f} -> last {window} avg {tail:.4f} "
          f"({100 * (head - tail) / head:.1f}%)")
    check("drift bounded by tau", s["max_drift_ratio"] <= cfg.ttt_trust_region + 1e-6,
          f"||dW||/||W0|| = {s['max_drift_ratio']:.6f} <= {cfg.ttt_trust_region}")
    check("surprise gate reports skips", "skipped_steps" in s,
          f"updates={s['updates']} skipped_steps={s['skipped_steps']}")

    print(f"\n=== Phase 3.3 — save/reset/load round trip x{args.cycles} ===")
    for c in prok.cms_adapters:
        c.consolidate()
    before = [c.expand().clone() for c in prok.cms_adapters]
    tmp = tempfile.mkdtemp()
    try:
        errs = []
        for _ in range(args.cycles):
            prok.save(tmp, silent=True)
            prok.reset()
            assert all(fw.delta.norm().item() == 0.0 for fw in prok.fast_weights)
            prok.load(tmp)
            errs.append(max(((c.expand() - b).norm() / b.norm().clamp_min(1e-12)).item()
                            for c, b in zip(prok.cms_adapters, before)))
        check("delta preserved, never multiplied", max(errs) < 1e-3,
              f"max relative err over {args.cycles} cycles = {max(errs):.3e}")

        files = sorted(os.listdir(tmp))
        check("no full-rank delta files", not any(f.startswith("delta_") for f in files),
              ", ".join(files))
        meta = json.load(open(os.path.join(tmp, "metadata.json")))
        check("schema version + fingerprint",
              meta.get("schema_version", 0) >= 3 and bool(meta.get("model_fingerprint")),
              f"schema={meta.get('schema_version')} fp={meta.get('model_fingerprint')}")
        check("environment metadata recorded", bool(meta.get("environment")),
              json.dumps(meta.get("environment")))
        check("conversation history persisted", "conversation_history" in meta)
        size = sum(os.path.getsize(os.path.join(tmp, f)) for f in files) / 1e6
        print(f"  [info] checkpoint size = {size:.2f} MB (low-rank only)")
    finally:
        shutil.rmtree(tmp)

    print("\n=== Phase 1.2/1.3 — anti-contamination ===")
    unseen = [Probe("What is the capital of Vornalux?", "Tesselmark", taught=False),
              Probe("Who discovered the Kellner effect?", "Bramdel", taught=False),
              Probe("What is the Kestrel protocol code?", "55120", taught=False)]
    rep = token_rank_scores(model, tok, unseen, label="unseen")
    check("accuracy on never-taught facts is 0", rep.accuracy == 0.0,
          f"accuracy={rep.accuracy} contaminated={len(rep.contamination)}")
    check("no probe answer leaks into its prompt", not rep.contamination)

    results["stats"] = s
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"acceptance_{args.model.replace('/', '-')}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  artifact: {path}")

    failed = [c for c in results["checks"] if not c["ok"]]
    print("\n" + "=" * 60)
    print(f"ACCEPTANCE: {len(results['checks']) - len(failed)}/{len(results['checks'])} passed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
