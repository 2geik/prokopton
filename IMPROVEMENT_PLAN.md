# Prokopton — Improvement Plan

An ordered, step-by-step remediation and validation plan derived from a full read of the
repository plus empirical verification of each claim by running the actual code.

**Status:** **implemented** (v0.5.0). Every phase below is done and gated by
tests in `tests/test_regression.py` (the Phase 6 defect table) plus
`scripts/acceptance.py` for the real-model acceptance criteria. Open decisions
are answered at the end of this document.

---

## Scope and constraints

| | |
|---|---|
| **Hardware** | Apple M4 Pro, 24 GB unified memory, MPS backend (MLX not installed) |
| **Principles** | No quality downgrades — no precision reduction, no model shrinking, no quantization |
| **Scope** | Phased: first correctness/safety (stop active harm), then real validation (make claims measurable) |
| **Multimodal** | Freeze the encoder-free tokenizers; make the native-VLM path (`ProkoptonVL`) primary |

---

## Verified findings that shaped this plan

Each row was confirmed by running code or querying the model registry, not inferred.

| Finding | Evidence | Impact on plan |
|---|---|---|
| `torch.svd_lowrank` **crashes on MPS** | `NotImplementedError: aten::linalg_qr.out`. `torch.linalg.svd` works. | CMS consolidation **cannot run on this Mac at all** today → Phase 0 |
| MLX backend is **incompatible with TTT** | `load_model` returns a non-torch MLX model; `Prokopton` requires `torch.autograd.grad` | The README-recommended `pip install -e ".[mlx]"` silently breaks all learning → Phase 0 |
| `google/gemma-4-E2B` **is natively multimodal** | `config.json` has `vision_config`, `audio_config`, `image_token_id`, `audio_token_id`; pipeline is `image-text-to-text` | The "E2B is text-only" premise is wrong; the entire encoder-free audio effort was unnecessary → Phase 4 |
| Gemma-4 requires `transformers 5.5.0.dev0` | Model config; highest installable version on the index is **4.57.6** | E2B **cannot be loaded** in the current environment → Phase 0 decision point |
| Qwen3-VL **4B / 8B are official and loadable** | HF registry; `qwen3_vl` present in transformers 4.57.6 mapping | Moving 2B → 4B is an **upgrade**, not a downgrade |
| Evaluation is 100% contaminated | Answer is placed in the prompt, then searched for in the echoed output | Every accuracy/forgetting number is a tautology → Phase 1 |
| `‖ΔW‖/‖W‖` reaches **107%** after 300 turns | Measured; top-1 predictions flip on 5/5 anchor probes by turn 10 | Unbounded drift is the core hallucination driver → Phase 2 |
| Each `load()` **doubles** the learned delta | Measured: 1.99× → 3.99× → 7.98× over four reloads | Persistence corrupts the model on every session → Phase 3 |
| All 45 tests pass and catch none of the above | Test suite asserts shapes, init, and file existence only | Regression tests must be rewritten → Phase 6 |

### Environment probes (M4 Pro, MPS)

```
mps available        : True
bf16 matmul          : OK
torch.linalg.svd     : OK
torch.fft.rfft       : OK
torch.autograd.grad  : OK
torch.svd_lowrank    : FAIL  (aten::linalg_qr.out not implemented for MPS)
```

### Model recommendation

**Primary: `Qwen/Qwen3-VL-4B-Instruct`** — fp16 ≈ 8 GB plus ~0.5 GB TTT state, comfortable in
24 GB, and twice the current 2B. Stretch: **8B-Instruct** (≈16 GB, workable but tighter).
Gemma-4-E2B stays a separate target for when transformers 5.x becomes installable; it has a
unique native-audio advantage worth revisiting.

---

## Phase 0 — Make it loadable (P0, blocking)

Nothing downstream is measurable until this phase is done.

### 0.1 Fix the model loading path
- `prokopton/backends.py:262` — replace `AutoModelForCausalLM` with `AutoModelForImageTextToText`
  for Qwen3-VL. Read the model card's `transformersInfo.auto_model` field and dispatch on it;
  Gemma4 declares `AutoModelForMultimodalLM`.
- `prokopton/models/__init__.py:49` — rebuild `AVAILABLE_MODELS` from verified models.
- **Acceptance:** `load_model("Qwen/Qwen3-VL-4B-Instruct")` loads on MPS in bf16 and reports VRAM.

### 0.2 Close the MLX trap
- `prokopton/backends.py:253` — the MLX path returns a model TTT cannot touch. `detect_backend()`
  must not return MLX in a learning context; fall back to MPS or raise a clear error.
- Permit MLX only for `--no-ttt` (frozen chat).
- **Acceptance:** with MLX installed, `prokopton` still uses MPS for learning; MLX + TTT raises
  an explicit error instead of failing obscurely.

### 0.3 Fix SVD on MPS (confirmed broken)
- `prokopton/core/__init__.py:126` — `torch.svd_lowrank` is unavailable on MPS. Two options:
  (a) `torch.linalg.svd` + truncate, (b) move the delta to CPU for the SVD.
  **Recommend (b)** — deterministic, and consolidation is infrequent so the transfer is cheap.
- **Acceptance:** `CMSAdapter.consolidate()` runs without error on MPS; CPU and GPU results agree
  within 1e-4.

### 0.4 Verify TTT layer selection against the real model
- `prokopton/core/__init__.py:356` — the primary branch requires `'language_model.layers'` **and**
  `'mlp.down_proj'` in the module name. Print actual module names for Qwen3-VL / Gemma4 and verify.
- **Latent bug:** the fallback branch (`linears[-5:]`, line 367) takes the last five `nn.Linear`
  modules in registration order. On a multimodal model those may belong to the **vision or audio
  tower**, so TTT would train the vision encoder instead of the language model. Remove the
  fallback or scope it explicitly to the language model.
- **Acceptance:** every selected layer name matches `...language_model.layers.N.mlp.down_proj`,
  enforced by a test.

### 0.5 Dependency and version hygiene
- `pyproject.toml` — pin `transformers` to a range that is actually installable; regenerate
  `requirements.lock.txt`.
- Deduplicate the three competing versions (`__init__.py:6` = 0.3.3, pyproject = 0.3.2, git = v0.4.0).
- **Acceptance:** clean venv, `pip install -e .`, `import prokopton` succeeds.

---

## Phase 1 — Measurement layer (do this first)

Every later phase is judged by this one.

### 1.1 Make `generate()` return only new tokens
- `prokopton/core/__init__.py:795` — `decode(outputs[0])` includes the prompt. Slice with
  `outputs[0][inputs_len:]`. Separate `generate_completion()` (raw) from `generate()` (clean).
- **Acceptance:** `generate(prompt)` output does not contain the prompt.

### 1.2 Build a contamination-proof evaluator
- `prokopton/eval/__init__.py:69` — the context is injected into the prompt and
  `expected in answer` is then checked. **Remove this.**
- Promote the closed-book token-rank metric from `m3plus_intensive_ttt.py:177` into
  `prokopton/eval/`. Also fix its tokenizer-boundary bug: `prompt` and `full_text` are tokenized
  separately, so `answer_start` can drift.
- Report both absolute accuracy and **Δ vs a frozen baseline**.

### 1.3 Add anti-contamination tests (regression guard)
- Accuracy on never-taught facts must be ≈ 0.
- The expected answer string must **not** appear in the prompt; if it does, the test fails.
- **Acceptance:** both tests are mandatory in CI. Had they existed, the current eval suite
  would fail.

### 1.4 Real continual-learning metrics
- Forgetting, backward/forward transfer, anchor drift, growth — all measured **against a frozen
  baseline control**.
- Expand the anchor set (3 questions is far too weak; it saturates by chance).
- Report mean + confidence interval + seed.

### 1.5 Commit experiment artifacts
- Remove `experiments/runs/` from `.gitignore` (line 21). Every run writes JSON plus environment
  metadata (torch/transformers versions, device, model sha).
- **Acceptance:** every claim in the README links to a committed artifact.

### 1.6 Ablation harness
- Run a `--no-ttt` control in every experiment. TTT contribution = TTT-on minus TTT-off.

---

## Phase 2 — Stabilize the update (anti-hallucination)

Measured problem: `‖ΔW‖/‖W‖` → **107%** after 300 turns; top-1 predictions flip on **5/5**
general-knowledge probes by turn 10.

### 2.1 Immutable reference `W₀`
- `FastWeight.original_W` is currently mutated by `apply_to_model()` (`core/__init__.py:139`).
  Keep a genuinely immutable `W₀` that `reset()` restores from.

### 2.2 Parameterize fast weights as low-rank deltas
- Instead of editing the raw weight, use `W = W₀ + B·A`. Benefits: a natural trust region, tiny
  optimizer state, tiny checkpoints, and `W₀` preserved by construction. This also resolves much
  of Phase 3.

### 2.3 Gradient clipping and trust region
- Global gradient-norm clipping plus a hard cap `‖ΔW‖/‖W₀‖ ≤ τ` (configurable, default ≈ 0.01);
  rescale rather than truncate when exceeded.
- **Acceptance:** `‖ΔW‖/‖W₀‖ ≤ τ` even after 1000 turns.

### 2.4 Decay toward `W₀`
- Scale the delta by `(1 − λ)` each step. This structurally prevents unbounded drift and sets the
  "new knowledge must not erase old" trade-off explicitly.

### 2.5 Fix momentum + surprise double-scaling
- `core/__init__.py:70,488` — the surprise cap is 5× and momentum 0.9 amplifies the steady-state
  step by `1/(1−0.9)` = 10×, giving an **effective step up to 50× `ttt_lr`**. Either drop momentum
  or normalize it into the LR (Adam-style `lr·ĝ/(√v+ε)`). Report the **effective** LR in `stats`.

### 2.6 Make the surprise gate actually gate
- `core/__init__.py:76` — with `surprise_threshold = 0.0` and loss always ≥ 0, the gate **never
  fires**. Set a meaningful default (e.g. a z-score over the running EMA) and report skipped steps.

### 2.7 Anchor / KL regularization — the real source of "never forgetting"
- `plan.md` presents SDFT (arXiv 2601.19897) as the consolidation mechanism, but **no SDFT, KL
  term, teacher, or self-distillation exists anywhere in the codebase**. Minimum viable version:
  add a KL penalty against the frozen model on an anchor batch at every `learn()` step.
- Full SDFT (demonstration-conditioned self-teacher) comes afterwards.

### 2.8 Apply the chat template
- `core/__init__.py:817` — replace the raw `"User: ...\nAssistant:"` string with the tokenizer's
  chat template. Bypassing instruct alignment is an independent hallucination source.

### 2.9 Constrain the learning target
- Today the target is the raw user utterance, including questions, typos, and false premises.
  Restrict to (a) assistant responses, (b) facts explicitly supplied via `learn(fact)`, or
  (c) user-confirmed content.
- **Acceptance:** a user repeating a false fact 50 times cannot push `‖ΔW‖` past τ.

---

## Phase 3 — Persistence correctness

Measured problem: every `load()` multiplies the learned delta by ~2× (7.98× after four reloads).

### 3.1 Separate `restore` from `commit`
- `apply_to_model()` currently tries to do both. Define `restore()` = `W = W₀ + Δ_saved`
  (idempotent) and `commit()` = `W₀ ← W` (explicit, one-way).

### 3.2 Fix the `alpha/rank` scaling
- `core/__init__.py:133` — `alpha/rank = 32/16 = 2.0` silently doubles every delta. Set
  `alpha == rank` or fold the scaling into initialization.

### 3.3 Add a round-trip property test
- After `save → reset → load`, `‖ΔW_after − ΔW_before‖ ≈ 0`. Repeat **five times** — the current
  bug is only visible across repeated reloads.

### 3.4 Persist the full state
- `conversation_history`, step counter, RNG state, config, and a **model fingerprint**
  (architecture + shape hash). History is not saved today, so "continue where you left off" is false.

### 3.5 Schema version and architecture guard
- Refuse to load memory into a mismatched model; raise a clear error rather than silently corrupting.

### 3.6 Genuinely low-rank checkpoints
- The README advertises "low-rank CMS adapters", but `save()` also writes a **full-rank**
  `delta_{i}.pt` per layer (`core/__init__.py:878`). Phase 2.2 makes these unnecessary. Also fix
  the incremental `is_dirty` logic (`core/__init__.py:97`), which is almost always true so
  incrementality never engages.

---

## Phase 4 — Multimodal: freeze encoder-free, invest in the VLM

### 4.1 Freeze the encoder-free path (no new work)
- Fix the off-by-one (`core/__init__.py:623-634`) so it is not left broken, then mark
  `VisualTokenizer` / `AudioTokenizer` `@deprecated`.
- The rationale is already established: Gemma-4-E2B has a **native audio encoder**, and the
  Mel-LLM paper itself reports that encoder-free approaches need a multimodal init.
- Fix M10's "first loss vs last loss" metric — it compares different samples; use a rolling average.

### 4.2 `ProkoptonVL` baseline fixes
- `vlm.py:274` — `cms.reset()` does not exist → `AttributeError`. Add it (or use `mark_clean()`).
- Add `save()` / `load()`; today only a full model dump exists, with no path to reload memory.

### 4.3 Loss masking (critical)
- `vlm.py:168-176` — the loss spans **all** tokens: image tokens, chat-template boilerplate, and
  the system prompt. The model is fit to nonsense targets. Mask to **assistant/caption** tokens only.
- **Acceptance:** a unit test asserts that image-token label positions are `-100`.

### 4.4 Add a learn→generate mode for attribution
- `vlm.py:234` — generation happens *before* learning, so no successful output can be attributed
  to TTT. Add a `learn_then_generate` mode and an A/B comparison flag.

### 4.5 Move up to 4B
- Default to `Qwen3-VL-4B-Instruct` (2B → 4B; comfortable at 24 GB). Keep 8B optional.

### 4.6 Multimodal evaluation
- Vision VQA, plus an audio-emotion probe if the Gemma4 path opens. Again closed-book and
  frozen-baseline controlled.

---

## Phase 5 — Efficiency and memory

### 5.1 Shrink optimizer state
- Each `FastWeight` keeps `velocity` plus `original_W` — two full copies. For Qwen3-VL-4B across
  5 layers that is ~124M params ≈ 0.5 GB in bf16. Keep velocity in bf16 and hold a single `W₀`;
  Phase 2.2 shrinks this a further ~50×.

### 5.2 Make PER replay non-recursive
- `core/__init__.py:505` — `learn()` calls `learn()`; 60 turns produce 68 steps. Convert to a
  bounded, queue-based, non-recursive loop.

### 5.3 Schedule consolidation
- Offload SVD to CPU (Phase 0.3) and schedule per layer frequency instead of consolidating every
  adapter every step.

### 5.4 Asynchronous saving
- `auto_save_every=100` is synchronous and writes hundreds of MB. Use a background thread with
  atomic writes (tmp file + rename).

### 5.5 Memory budget document
- Tabulate model + state + activations and state which model scales fit in 24 GB.

---

## Phase 6 — Tests and CI

The existing 45 tests pass but catch none of the defects above. Write a regression test per phase:

| Defect | Test |
|---|---|
| Eval contamination | Expected answer must not appear in the prompt; accuracy on untaught facts ≈ 0 |
| Unbounded drift | After 1000 turns, `‖ΔW‖/‖W₀‖ ≤ τ` |
| Memory compounding | After 5× `save→reset→load`, the delta is preserved, not multiplied |
| CMS idempotency | `consolidate→apply_to_model` twice yields an unchanged result |
| Multimodal labels | `labels[j] == combined_ids[j+1]` |
| Layer selection | Selected layers are the language model's `down_proj` modules |
| Surprise gate | Updates are skipped at low surprise |
| Chat template | Output does not echo the prompt; the template is applied |

**Integration test:** teach N facts → closed-book accuracy **increases**, anchor accuracy
**does not change**. Written today, this test would fail against the current code.

**CI:** small-model CPU run on every PR; nightly MPS run with the 4B model.

---

## Phase 7 — Documentation and claims

### 7.1 Triage every README claim
Mark each row **verified** (with an artifact link) / **unverified** / **wrong**.
Priority items: "Zero Forgetting", "60% → 90% accuracy", "100% cross-session recall", "M3+ ✓",
and the claim that E2B is text-only.

### 7.2 Re-measure the performance table
Rebuild it for M4 Pro / 24 GB.

### 7.3 Document `ProkoptonVL`
It is the headline v0.4.0 feature and is entirely absent from the README.

### 7.4 Add a threat-model section
User input directly modifies weights, so the poisoning surface is real. Document the limits and
the recommended constraints.

### 7.5 Keep `plan.md` honest
Update "Risks / honest limits": SDFT is not implemented, CMS is a snapshot rather than an
accumulator, and MLX does not support TTT.

---

## Ordering and dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5
(blocks)   (measurement) (stability) (persistence) (efficiency)
                             │
                             └──► Phase 4 (VLM; depends on 0.1 + 1.2)

Phase 6 (tests)      written in parallel with every phase
Phase 7 (docs)       last, once measurements exist
```

**Do not skip Phase 1.** It is the only way to prove that the Phase 2–5 improvements work;
without it, "fixed" is untestable.

---

## Appendix — defect inventory

| # | Defect | Location | Severity |
|---|---|---|---|
| 1 | Evaluation contaminated by prompt echo + answer-in-prompt | `eval/__init__.py:69`, `core/__init__.py:795` | Critical |
| 2 | Unbounded weight drift, no trust region | `core/__init__.py:468` | Critical |
| 3 | Persistence compounds the delta ~2× per load | `core/__init__.py:133,135` | Critical |
| 4 | No frozen reference; SDFT/KL absent despite being claimed | whole codebase | Critical |
| 5 | `torch.svd_lowrank` unavailable on MPS | `core/__init__.py:126` | Critical (on target HW) |
| 6 | MLX backend silently incompatible with TTT | `backends.py:253` | Critical |
| 7 | Multimodal labels off by one | `core/__init__.py:623-634` | Critical |
| 8 | `consolidate()` never absorbs or clears the delta | `core/__init__.py:122` | High |
| 9 | Fallback layer selection may target the vision tower | `core/__init__.py:367` | High |
| 10 | `ProkoptonVL.reset()` calls nonexistent `cms.reset()` | `vlm.py:274` | High |
| 11 | No chat template applied | `core/__init__.py:817` | High |
| 12 | Surprise gate never fires | `core/__init__.py:76` | High |
| 13 | Momentum + surprise effective step up to 50× | `core/__init__.py:70,488` | High |
| 14 | VLM loss spans image tokens and chat boilerplate | `vlm.py:168-176` | High |
| 15 | `load()` does not restore conversation history | `core/__init__.py:952` | Medium |
| 16 | PER replay recurses into `learn()` | `core/__init__.py:505` | Medium |
| 17 | Incremental save never engages | `core/__init__.py:97` | Medium |
| 18 | Full-rank deltas written despite "low-rank" docs | `core/__init__.py:878` | Medium |
| 19 | No experiment artifacts committed | `.gitignore:21` | Medium |
| 20 | Three conflicting version numbers | `__init__.py:6`, `pyproject.toml`, git tag | Low |
| 21 | `ProkoptonVL` undocumented | README | Low |

---

## Open decisions — resolved

1. **Model target** → **Qwen3-VL-4B-Instruct is the default.** `transformers 5.x`
   *is* installable on this index (5.17.0 verified), so Gemma-4-E2B is no longer
   blocked; it is kept as a separate target for its unique native-audio path.
   Measured on MPS: 8.33 GB in bf16, comfortable in 24 GB.

2. **Phase 2.2 (low-rank vs raw)** → **raw is the default, low-rank is the
   compact fallback.** Measured on the 4B model, 40 repeats of one fact, τ = 0.01
   (`docs/memory-budget.md`):

   | `parametrization` | `ttt_lr` | loss reduction | `‖ΔW‖/‖W₀‖` |
   |---|---|---|---|
   | **raw** | **1e-2** | **98.7 %** | **0.0026** |
   | raw | 1e-3 | 31.3 % | 0.0004 |
   | lowrank (r=16) | 1e-1 | 26.7 % | 0.0026 |
   | lowrank (r=16) | 1e0 | 100 % | 0.0100 (hits τ) |

   Raw updates are the In-Place TTT paper's rule and the fastest learner; with
   clipping, decay and the trust region they stay bounded across 1000 turns.
   Low-rank needs 10–100× the LR for the same effect but cuts TTT state ~4× and
   makes checkpoints exact. **Persistence is low-rank in both modes** (CMS SVD),
   so 3.6 holds regardless.

3. **Phase 2.9 (learning target)** → **all three, via explicit APIs.**
   `learn(fact)` learns the fact; `learn(prompt, target=reply)` learns only the
   reply; `chat()` learns from its **own assistant response** (never from the raw
   user utterance); `confirm_learning(text)` handles user-confirmed content.

### Implementation notes beyond the plan

* **LoRA dead-gradient trap.** Low-rank factors must not both start at zero —
  `d/dB = g·Aᵀ = 0` then, and nothing ever learns. `B` starts at zero (so the
  model is unchanged at init) and `A` carries a small random basis.
* **Bias-corrected momentum.** The historical `v = 0.9v − lr·g` reached 50× `lr`
  in steady state; the uncorrected normalized EMA overshoots the other way and
  makes the first step 10× too small. Both are now bias-corrected so the
  effective step is exactly `lr · ĝ`.
* **fp32 delta state.** Keeping the delta in the model's bf16 re-rounds it on
  every `restore()` and the error compounds across reloads (4.7e-3 after 5
  cycles vs **1.2e-5** with fp32 state).
* **Consolidation is scheduled, not per-step.** A full SVD of a
  `hidden × intermediate` matrix cost 2 s per `learn()`; with
  `cms_interval_base=32` the same call is 1.6 s.
