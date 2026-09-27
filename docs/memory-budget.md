# Memory budget — Prokopton on Apple M4 Pro / 24 GB

Measured on the target machine (Apple M4 Pro, 24 GB unified memory, MPS,
torch 2.14.0, transformers 5.17.0). Numbers in **bold** are measured, not
estimated.

## Model footprints (bf16, MPS)

| Model | Params | Weights | + TTT state (5 layers) | Fits 24 GB? |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | 2.2 B | ~4.5 GB | +0.4 GB | yes |
| **`Qwen/Qwen3-VL-4B-Instruct`** | 4.0 B | **8.33 GB measured** | +1.2 GB | **yes, default** |
| `Qwen/Qwen3-VL-8B-Instruct` | 8.0 B | ~16 GB | +1.2 GB | tight |
| `google/gemma-4-E2B` (any-to-any) | 5.1 B | ~9.5 GB | +1.2 GB | yes, needs transformers ≥ 5.5 |

## TTT state per layer (`down_proj`, shape `2560 × 9728` = 24.9 M params)

| Component | `parametrization="raw"` | `parametrization="lowrank"` (r=16) |
|---|---|---|
| `W0` (immutable reference) | 24.9 M × 2 B = **49.8 MB** | 49.8 MB |
| delta state | 24.9 M × 4 B = 99.6 MB (fp32) | 0.20 MB (fp32) |
| momentum / velocity | 24.9 M × 2 B = 49.8 MB (bf16) | 0.20 MB (bf16) |
| CMS factors (checkpoint) | 64 × (2560+9728) × 4 B = 3.1 MB | 3.1 MB |
| **per layer total** | **~202 MB** | **~53 MB** |
| **5 layers total** | **~1.0 GB** | **~0.27 GB** |

Notes:

* The delta and the low-rank factors are held in **float32 on purpose**. Storing
  them in the model's bf16 re-quantises the delta on every `restore()` and the
  error compounds across reloads (measured: 4.7 × 10⁻³ after five cycles, versus
  **1.2 × 10⁻⁵** with fp32 state). See
  `tests/test_regression.py::test_delta_state_is_not_rounded_by_the_model_dtype`.
* Momentum is stored in the model dtype (bf16) — it is a smoothed direction, not
  a value that has to survive a save → load round trip.
* Checkpoints are **only** CMS factors + metadata. No full-rank `delta_i.pt` is
  ever written (measured: 5 layers → **15.7 MB** on disk).

## Measured headroom on 24 GB (4B model, 5 TTT layers)

```
model weights (bf16)        8.33 GB
TTT state (raw, 5 layers)   ~1.0 GB
activations / KV (batch 1)  ~0.6 GB
python + transformers       ~0.6 GB
                            ────────
total                      ~10.5 GB      → ~13 GB headroom
```

The 8B model would sit at ~18 GB total: workable but with little room for a
long context or a larger `ttt_n_layers`.

## Optimizer-state trade-off (raw vs low-rank)

Measured on `Qwen/Qwen3-VL-4B-Instruct` / MPS, 40 repeats of one fact, τ = 0.01:

| `parametrization` | `ttt_lr` | loss reduction | final `‖ΔW‖/‖W₀‖` |
|---|---|---|---|
| **raw** | **1e-2** | **98.7 %** | **0.0026** |
| raw | 3e-3 | 68.1 % | 0.0010 |
| raw | 1e-3 | 31.3 % | 0.0004 |
| lowrank (r=16) | 1e-2 | 0.1 % | 0.0002 |
| lowrank (r=16) | 1e-1 | 26.7 % | 0.0026 |
| lowrank (r=16) | 1e0 | 100 % | 0.0100 (hits τ) |
| lowrank (r=64) | 1e-1 | 56.0 % | 0.0037 |

Conclusions recorded here because they answer an open question in
`IMPROVEMENT_PLAN.md` §Open decisions:

1. **Raw updates are the default.** They are the update rule of the In-Place TTT
   paper (which edits `down_proj` directly) and they are the fastest learner per
   unit learning rate. With clipping, decay and the τ = 0.01 trust region they
   are bounded: 1000 turns never take `‖ΔW‖/‖W₀‖` past τ
   (`tests/test_regression.py::test_drift_bounded_after_1000_turns`).
2. **Low-rank is the compact fallback**, not the default. It needs 10–100× the
   learning rate for the same effect, but cuts TTT state by ~4× and makes the
   checkpoint exact rather than a rank-`cms_rank` projection.
3. **Persistence is low-rank in both modes.** `save()` consolidates the delta to
   rank `cms_rank` (default 64) and writes only `cms_{i}.pt`. With raw updates
   accumulated over ≲ 60 steps the rank of the delta stays under `cms_rank`, so
   the round trip is exact to fp32 noise.

## Practical limits

| Knob | Cost | Guidance |
|---|---|---|
| `ttt_n_layers` | +~202 MB per layer (raw) | 5 is comfortable; 36 would need ~7 GB |
| `ttt_rank` (lowrank) | +(r × (in+out)) × 4 B per layer | 16 for compactness, 64 for expressiveness |
| `cms_rank` | checkpoint size only | 64 covers ~60 raw updates; raise it for longer sessions |
| `per_capacity` | strings only | free |
| `kl_weight > 0` | +1 forward on the anchor batch per step | 0.05 costs ~30 % extra step time |

## Threat model

User text is turned into gradients on `down_proj` — the poisoning surface is
real. The constraints that bound it:

* `ttt_trust_region` caps `‖ΔW‖/‖W₀‖` regardless of what the user says.
* `ttt_decay` pulls the delta back toward `W₀`.
* The surprise gate skips updates on unsurprising input, so repetition alone is
  a weak attack.
* `chat()` never learns from the raw user utterance — only from assistant
  responses, explicit `learn(fact)` calls, or `confirm_learning()` content.
* `load()` refuses a checkpoint whose `model_fingerprint` does not match.

None of this makes poisoning *impossible*; it makes it bounded and auditable
(`stats["max_drift_ratio"]`, `stats["skipped_steps"]`).
