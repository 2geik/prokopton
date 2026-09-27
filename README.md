# Prokopton 🧠

> *Prokopton* (προκόπτων): one who continually advances toward wisdom.

A **self-improving LLM** whose weights actually update during conversation —
learning from experience and **growing** over time, with a hard bound on how far
the model can drift. Not RAG or agent-memory — the parameters themselves change
at inference time.

🔗 **Repo:** [github.com/2geik/prokopton](https://github.com/2geik/prokopton)

> **Honesty note.** Every claim in this README is marked
> [verified](#claims-triage) / [unverified](#claims-triage) / [wrong](#claims-triage)
> in the triage table below, with the artifact that backs it. Nothing here is
> presented as measured unless it actually was.

---

## ✨ Features

- 🔄 **In-Place TTT** — `mlp.down_proj` weights update on every learning turn
- 📏 **Trust region** — `‖ΔW‖/‖W₀‖ ≤ τ` by construction; drift cannot run away
- 💾 **Persistent memory** — low-rank CMS adapters on disk, reload is idempotent
- 🧊 **Immutable `W₀`** — `reset()` restores the original model exactly
- 🖥️ **Multi-platform** — ROCm · CUDA · MPS · MLX · CPU (auto-detected)
- 🖼️ **Native VLM path** — `ProkoptonVL` uses the model's own vision tower
- 🎮 **TUI** — terminal app (Textual) with model downloader

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/2geik/prokopton.git
cd prokopton
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

That's it — Prokopton auto-detects your GPU and picks the best backend.

> **macOS Apple Silicon?** The MPS backend supports full TTT.
> `pip install -e ".[mlx]"` is **optional** and is for *frozen* chat only —
> MLX models are not `torch.nn.Module`s and cannot be trained at test time.
> See [Hardware support](#hardware-support).

### Launch

```bash
prokopton              # Interactive TUI, auto-detects your GPU
prokopton --help       # Show all options
```

### Platform-specific setup

| Platform | PyTorch install | Notes |
|----------|-----------------|-------|
| **AMD GPU** (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm7.0` | before `pip install -e .` |
| **NVIDIA GPU** (CUDA) | `pip install torch` (PyPI default) | auto-detected |
| **Apple Silicon** (MPS) | `pip install torch` (PyPI default) | **use this for TTT** |
| **Apple Silicon** (MLX) | `pip install -e ".[mlx]"` | frozen inference only — no TTT |
| **CPU-only** | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | works everywhere |

Auto-detection order: **ROCm → CUDA → MPS → MLX → CPU**. MLX is only selected
when no learning is requested; `detect_backend(require_torch=True)` (which every
TTT entry point uses) never returns it. Override with
`--backend rocm|cuda|mps|mlx|cpu`.

---

## 📖 Usage

### CLI reference

```
prokopton [OPTIONS]

Options:
  -m, --model MODEL       Model name or path (skip selection screen)
  -b, --backend BACKEND   Force backend: rocm, cuda, mps, mlx, cpu
  --lr LR                 TTT learning rate (default: 0.01)
  --n-layers N            Number of TTT layers (default: 5)
  --no-ttt                Frozen model mode (no learning; MLX allowed here only)
  --cpu                   Force CPU mode
  --save-dir DIR          Memory directory (default: prokopton_memory)
  -h, --help              Show this help
```

### Common patterns

```bash
# Interactive — auto-detect GPU, pick model in UI
prokopton

# Skip selection, load the default target model
prokopton --model Qwen/Qwen3-VL-4B-Instruct

# Frozen mode (chat only, no learning) — MLX allowed here
prokopton --backend mlx --model Qwen/Qwen3-VL-2B-Instruct --no-ttt

# Custom learning rate and depth
prokopton --model Qwen/Qwen3-VL-4B-Instruct --lr 0.005 --n-layers 3

# CPU-only
prokopton --cpu --model Qwen/Qwen3-VL-2B-Instruct
```

### Inside the TUI

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit (auto-saves memory) |
| `Ctrl+S` | Save memory to disk |
| `Ctrl+L` | Load memory from disk |
| `Ctrl+R` | Reset all learned knowledge |
| `Ctrl+M` | Switch model |
| `Ctrl+D` | Download model from HuggingFace |
| `Ctrl+P` | View statistics tab |
| `Enter` | Send message |

**Tabs:** 💬 Chat · 📊 Stats (steps, `‖ΔW‖/‖W₀‖`, drift vs τ, skipped steps) · ⚙️ Settings

### Memory workflow

```
Session 1:  chat → learn → Ctrl+S (save)
Session 2:  launch → Ctrl+L (load) → continue where you left off
```

Memory is stored as low-rank CMS adapters in `prokopton_memory/`. Measured
size for 5 TTT layers on Qwen3-VL-4B: **15.7 MB**. `load()` is idempotent — it
sets `W = W₀ + Δ_saved` rather than accumulating, so repeated reload cycles
cannot compound the delta (measured round-trip error over 5 cycles: **1.2e-5**;
the pre-fix code doubled it every time).

---

## 🐍 Python API

```python
from prokopton import (Prokopton, ProkoptonConfig, detect_backend,
                       load_model, token_rank_scores, Probe)

be = detect_backend(require_torch=True)          # never MLX when learning
model, tokenizer = load_model("Qwen/Qwen3-VL-4B-Instruct", be)

config = ProkoptonConfig(ttt_n_layers=5, ttt_lr=1e-2)
prok = Prokopton(model, tokenizer, config)

# Learn an explicit fact (the whole string is the target)
prok.learn("Zephyria's capital is Aethel.")

# chat() responds first, then learns from its OWN reply — never from the raw
# user utterance, which may contain questions, typos or false premises.
answer = prok.chat("What is the capital of Zephyria?")

prok.save("my_memory")

# New session — reload (idempotent)
prok2 = Prokopton(model, tokenizer, config)
prok2.load("my_memory")

# Closed-book evaluation with a frozen-baseline control
probes = [Probe("What is the capital of Zephyria?", "Aethel")]
report = token_rank_scores(model, tokenizer, probes)
print(report.accuracy, report.mean_rank, report.contamination)

# Monitoring
print(prok.stats["max_drift_ratio"], prok.stats["trust_region"])
```

### Learning targets

| API | What is learned |
|---|---|
| `prok.learn(fact)` | the fact itself (explicit, user-supplied) |
| `prok.learn(prompt, target=reply)` | only `reply` tokens; `prompt` is masked |
| `prok.chat(...)` | the **assistant response**, masked to completion tokens |
| `prok.confirm_learning(text)` | explicitly confirmed content |

### Export the trained model

```python
prok.save_pretrained("prokopton_model")   # commit() → merges Δ into W₀, saves HF
# loadable without Prokopton:
from transformers import AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained("prokopton_model")
```

---

## 🖼️ ProkoptonVL — native multimodal

The headline path for vision is **`prokopton.vlm.ProkoptonVL`**, which uses the
model's own vision tower instead of a separately trained encoder-free tokenizer.

```python
from prokopton.vlm import ProkoptonVL
from PIL import Image

pvl = ProkoptonVL("Qwen/Qwen3-VL-4B-Instruct")     # default is 4B, not 2B

# Learn from an image + caption. The loss is masked to the CAPTION tokens only:
# image placeholders and chat-template boilerplate are -100.
pvl.learn_image(Image.open("cat.jpg"), "A grey tabby cat asleep on a windowsill.")

# Generate. The prompt is sliced off — no echo.
pvl.generate("What is in this image?", image=Image.open("cat.jpg"))

# Learn BEFORE generating so a good answer can be attributed to TTT.
pvl.learn_then_generate({"text": "What is in this image?", "image": img})

# A/B: what did TTT change?
pvl.compare_modes({"text": "What is in this image?", "image": img})

pvl.save("vl_memory")     # low-rank CMS adapters, not a full dump
pvl.load("vl_memory")     # refuses a mismatched model fingerprint
```

Key properties (each has a regression test):

* **Completion-only loss** — `labels[j] == input_ids[j]` for the caption span,
  `-100` everywhere else. A unit test asserts every image-token label is `-100`.
* **`learn_then_generate`** exists specifically for attribution; `chat()` uses
  `generate_then_learn` by default and both can be compared with `compare_modes()`.
* **Layer selection** is scoped to the language model. TTT attaches only to
  `…language_model.layers.N.mlp.down_proj`; the vision tower is never selected.

---

## 📦 Supported models

| Model | Params | VRAM (bf16) | Multimodal | Notes |
|-------|--------|-------------|------------|-------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2.2B | ~4.5 GB | ✅ | CI / low-memory target |
| **`Qwen/Qwen3-VL-4B-Instruct`** | 4.0B | **8.33 GB** | ✅ | **default**, measured on MPS |
| `Qwen/Qwen3-VL-8B-Instruct` | 8.0B | ~16 GB | ✅ | stretch, tight on 24 GB |
| `google/gemma-4-E2B` | 5.1B | ~9.5 GB | ✅ any-to-any (vision **+ audio**) | needs `transformers>=5.5` |

The loader dispatches on the model card: `*ForConditionalGeneration` on a
multimodal config → `AutoModelForImageTextToText`, Gemma4-style any-to-any →
`AutoModelForMultimodalLM`, otherwise `AutoModelForCausalLM`. TTT layer
selection then requires `…layers.N.mlp.down_proj` inside the **language model**
and rejects anything in a vision/audio tower.

---

## 🖥️ Hardware support

| Backend | Platform | TTT | Notes |
|---------|----------|-----|-------|
| **ROCm** | Linux | ✅ | AMD Radeon |
| **CUDA** | Linux/Windows | ✅ | NVIDIA |
| **MPS** | macOS | ✅ | Apple Silicon — the reference platform |
| **MLX** | macOS | ❌ | frozen inference only; `load_model` raises without `allow_non_torch=True` |
| **CPU** | Any | ✅ | slow but works |

Memory budgets, the raw-vs-low-rank trade-off and the practical limits of each
knob are tabulated in [`docs/memory-budget.md`](docs/memory-budget.md).

---

## 📊 Measured performance

Apple M4 Pro, 24 GB unified memory, MPS, `Qwen/Qwen3-VL-4B-Instruct` (bf16),
torch 2.14.0, transformers 5.17.0. Median of 5 runs after 2 warmups.

| Operation | Time |
|-----------|-----:|
| `generate()` (64 new tokens) | 3067 ms |
| `generate_completion()` | 3069 ms |
| `learn()` | 1649 ms |
| `learn()` + KL anchor (`kl_weight=0.05`) | 2205 ms |
| `learn()` low-rank (`parametrization="lowrank"`) | 1462 ms |
| `learn()` + `generate()` | 5272 ms |

| Metric | Value |
|--------|------:|
| Model + TTT state in memory | 11.74 GB |
| Checkpoint (5 layers, rank 64) | 15.7 MB |
| `‖ΔW‖/‖W₀‖` after 25 turns of one fact | 0.00045 (τ = 0.01) |
| Loss change, 25 repeats of one fact | 1.91 → 1.49 (−22 %) |
| `save → reset → load`, ×5 cycles | relative error **1.2e-5** (was 2× compounding) |

Closed-book continual-learning run (3 tasks × 3 repeats, frozen-baseline
controlled, contamination-checked — `experiments/runs/eval_Qwen3-VL-4B_*.json`):

| Metric | Value |
|--------|------:|
| `growth_taught` (Δ accuracy on taught facts) | **+0.111** (0 % → 11 %) |
| `anchor_drift` | **0.000** |
| `unseen_change` (untaught fabricated facts) | **0.000** |
| `forgetting` | **0.000** |
| `max_drift_ratio` | **0.0004** (τ = 0.01) |

Artifacts: `experiments/runs/`. Reproduce with
`python scripts/acceptance.py` and `python -m pytest`.

---

## 🧪 Evaluation

The old evaluator put the expected answer in the prompt and then searched for it
in the echoed generation, which made every accuracy number a tautology. The
current one is closed-book and cannot be gamed that way:

```python
from prokopton.eval import Probe, token_rank_scores, FrozenBaseline

baseline = FrozenBaseline.capture(model, tokenizer, probes)
prok.learn("Zephyria's capital is Aethel.")
report = token_rank_scores(model, tokenizer, probes)
print(baseline.delta(report))      # Δ vs the frozen control
```

Enforced invariants (CI-gated):

* the expected answer must **not** appear in the prompt (`assert_uncontaminated`);
* accuracy on never-taught fabricated facts must be **0**;
* `generate()` must not echo the prompt;
* `run_full_evaluation()` writes a committed JSON artifact with environment
  metadata, and reports everything as **Δ vs a frozen baseline**.

Continual-learning metrics (forgetting, anchor drift, growth, forward transfer)
plus an ablation harness (`run_ablation` → TTT-on minus TTT-off) live in
`prokopton/eval/`.

---

## 🔒 Threat model

User text becomes gradients on `down_proj`, so the poisoning surface is real.
Prokopton does not make poisoning impossible — it makes it **bounded and
auditable**:

| Control | Effect |
|---------|--------|
| `ttt_trust_region` (τ = 0.01) | `‖ΔW‖/‖W₀‖` can never exceed τ, whatever the input |
| `ttt_decay` | pulls Δ back toward `W₀` every step |
| surprise gate (z-score) | repetition of an unsurprising claim is skipped and counted |
| `chat()` target policy | the raw user utterance is never a learning target |
| `model_fingerprint` | `load()` refuses memory built for a different architecture |
| `stats` | `max_drift_ratio`, `skipped_steps`, per-layer `‖ΔW‖` are always inspectable |

Recommendations: keep τ small, keep `learn_from_assistant=True`, review
`stats["max_drift_ratio"]` after untrusted input, and treat `save_pretrained()`
(the one-way `commit()`) as a deliberate act — it folds the delta into `W₀`.

---

## 📚 Foundations

- **Nested Learning / Hope** — arXiv 2512.24695 (NeurIPS 2025)
- **Titans** — arXiv 2501.00663
- **In-Place TTT** — arXiv 2604.06169 (the update rule this repo follows)
- **SDFT** — arXiv 2601.19897 (**not implemented here**; see the triage below)
- **Tuna-2** (encoder-free vision) — arXiv 2604.24763
- **Mel-LLM** (encoder-free audio) — arXiv 2606.10231

---

## 📋 Claims triage

Every previously published claim, re-marked after re-verification. Clean
numbers come from `experiments/runs/eval_Qwen3-VL-4B_*.json` (closed-book,
frozen-baseline controlled, contamination-checked).

| Claim | Status | Evidence / correction |
|---|---|---|
| "Zero Forgetting", "Forgetting ≈ 0, anchor preserved" | **verified (bounded)** | measured on Qwen3-VL-4B: `forgetting = 0.000`, `anchor_drift = 0.000` after teaching 3 tasks. The old evaluator was contaminated, so this is the first clean number — and it is 0.0, not an average |
| "M3+ ✓ — 60% → 90% accuracy" | **wrong** | the evaluator injected the answer into the prompt and matched it against the prompt echo. Clean re-measurement gives `growth_taught = +11 %` (0 % → 11 %) after 3 tasks × 3 repeats — real, but nothing like 60→90 |
| "100% cross-session recall" | **verified (persistence)** | `save → reset → load` ×5 keeps `‖ΔW‖` at **1.2e-5** relative error. Whether the *content* is recalled is a separate, still-unmeasured question |
| "Gemma 4 E2B is text-only" | **wrong** | `google/gemma-4-E2B` is **natively multimodal**: `vision_config`, `audio_config`, `image_token_id`, `audio_token_id`. The whole encoder-free audio effort was unnecessary |
| "MLX for best performance (optional)" | **wrong** | MLX returns a non-torch model; `torch.autograd.grad` cannot update it. MLX is frozen-inference only and is refused for learning |
| "Prokopton works with any `AutoModelForCausalLM` model" | **incomplete** | VL models need `AutoModelForImageTextToText` / `AutoModelForMultimodalLM`; the loader now dispatches on the model card |
| "Memory is stored as low-rank CMS adapters" | **verified** | 15.7 MB for 5 layers; no `delta_*.pt` is written (`tests/test_regression.py::test_no_full_rank_delta_files_are_written`) |
| "Continue where you left off" | **verified** | conversation history, step counter, config and a model fingerprint are persisted and restored |
| SDFT is the consolidation mechanism | **wrong** | no SDFT, KL teacher or self-distillation existed in the codebase. A **KL anchor penalty against the frozen `W₀`** is implemented instead (`kl_weight`); full SDFT is still future work |
| Performance table (Gemma 4 E2B / RX 6800) | **superseded** | re-measured for M4 Pro / MPS / Qwen3-VL-4B above |
| Three conflicting version numbers | **fixed** | `0.5.0` everywhere |
| `ProkoptonVL` undocumented | **fixed** | see [ProkoptonVL](#%EF%B8%8F-prokoptonvl--native-multimodal) |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
