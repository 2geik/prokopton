# Prokopton 🧠

> *Prokopton* (προκόπτων): one who continually advances toward wisdom.

A **self-improving LLM** whose weights actually update during conversation —
learning from experience, **growing** over time, and **never forgetting**.
Not RAG or agent-memory — the parameters themselves change at inference time.

🔗 **Repo:** [github.com/2geik/prokopton](https://github.com/2geik/prokopton)

---

## ✨ Features

- 🔄 **In-Place TTT** — MLP weights update on every conversation turn
- 💾 **Persistent Memory** — CMS adapters saved to disk, reloaded on restart
- 🎯 **Zero Forgetting** — Forgetting ≈ 0, anchor knowledge preserved
- 🖥️ **Multi-Platform** — ROCm · CUDA · MPS · MLX · CPU (auto-detected)
- 🎮 **TUI Interface** — User-friendly terminal app (Textual)
- ⬇️ **HF Model Downloader** — One-click from any HuggingFace URL
- 🖼️🎵 **Multimodal** — Vision + direct audio (no STT pipeline)

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/2geik/prokopton.git
cd prokopton
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

That's it — Prokopton auto-detects your GPU and picks the best backend.

> **macOS Apple Silicon?** `pip install -e ".[mlx]"` for best performance (optional).

### Launch

```bash
prokopton              # Interactive TUI, auto-detects your GPU
prokopton --help       # Show all options
```

### Platform-Specific Setup

| Platform | PyTorch Install | Notes |
|----------|----------------|-------|
| **AMD GPU** (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm7.0` | before `pip install -e .` |
| **NVIDIA GPU** (CUDA) | `pip install torch` (PyPI default) | auto-detected |
| **Apple Silicon** (MLX) | `pip install -e ".[mlx]"` | optional, falls back to MPS |
| **Apple Silicon** (MPS) | `pip install torch` (PyPI default) | built into PyTorch |
| **CPU-only** | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | works everywhere |

Prokopton auto-detects in this order: **ROCm → CUDA → MPS → MLX → CPU**.
Override with `--backend rocm|cuda|mps|mlx|cpu`.

---

## 📖 Usage

### CLI Reference

```
prokopton [OPTIONS]

Options:
  -m, --model MODEL       Model name or path (skip selection screen)
  -b, --backend BACKEND   Force backend: rocm, cuda, mps, mlx, cpu
  --lr LR                 TTT learning rate (default: 0.001)
  --n-layers N            Number of TTT layers (default: 5)
  --no-ttt                Frozen model mode (no learning)
  --cpu                   Force CPU mode
  --save-dir DIR          Memory directory (default: prokopton_memory)
  -h, --help              Show this help
```

### Common Patterns

```bash
# Interactive — auto-detect GPU, pick model in UI
prokopton

# Skip selection, load directly
prokopton --model google/gemma-4-E2B

# macOS with MLX
prokopton --backend mlx --model google/gemma-4-E2B

# Load local model from models/ folder
prokopton --model models/gemma-4-E2B

# Frozen mode (chat only, no learning)
prokopton --model google/gemma-4-E2B --no-ttt

# Custom learning rate
prokopton --model google/gemma-4-E2B --lr 0.0005 --n-layers 3

# CPU-only
prokopton --cpu --model google/gemma-4-E2B
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

**Tabs:**
- 💬 **Chat** — main conversation, every message triggers learning
- 📊 **Stats** — steps, weight delta, buffer size, live updates
- ⚙️ **Settings** — tune lr, layers, CMS rank; save/load/reset

### Getting a Model

**Option A — In the TUI:**
`Ctrl+D` → paste a HuggingFace URL or model ID → downloads to `models/`

**Option B — Manual:**
```bash
huggingface-cli download google/gemma-4-E2B --local-dir models/gemma-4-E2B
```
Any valid HF model folder in `models/` appears in the TUI selector.

### Memory Workflow

```
Session 1:  chat → learn → Ctrl+S (save)
Session 2:  launch → Ctrl+L (load) → continue where you left off
```

Memory is stored as low-rank CMS adapters in `prokopton_memory/`.

---

## 🧪 Python API

```python
from prokopton import Prokopton, ProkoptonConfig, load_model

# Load with auto-detected backend
be = detect_backend()
model, tokenizer = load_model("google/gemma-4-E2B", be)

# Wrap with Prokopton
config = ProkoptonConfig(ttt_n_layers=5, ttt_lr=1e-3)
prok = Prokopton(model, tokenizer, config)

# Learn from conversation
prok.learn("Zephyria's capital is Aethel.")
prok.save("my_memory")

# New session — reload
prok.load("my_memory")
answer = prok.chat("What is the capital of Zephyria?")
print(answer)  # → "Aethel"

# Stats
print(prok.stats)  # → steps, updates, weight_change, buffer_size
```

### Export the Trained Model

```python
# Embed CMS adapters into base weights
for cms in prok.cms_adapters:
    cms.consolidate()
    cms.apply_to_model()

# Save as standard HuggingFace model
model.save_pretrained("prokopton_model")
tokenizer.save_pretrained("prokopton_model")

# Now loadable without Prokopton:
model2 = AutoModelForCausalLM.from_pretrained("prokopton_model")
```

---

## 📦 Supported Models

| Model | Params | VRAM (bf16) | Notes |
|-------|--------|-------------|-------|
| `google/gemma-4-E2B` | 5.1B | ~9.5 GB | ✅ Recommended |
| `google/gemma-4-E4B` | 7.9B | ~14.2 GB | ⚠️ Tight on 16 GB |
| `google/gemma-4-12B` | 12B | 24+ GB | 🔮 Needs quantization |

Prokopton works with any `AutoModelForCausalLM` model — it auto-detects MLP layers for TTT.

---

## 🖥️ Hardware Support

| Backend | Platform | GPU | Performance |
|---------|----------|-----|-------------|
| **ROCm** | Linux | AMD Radeon | Best on AMD |
| **CUDA** | Linux/Windows | NVIDIA | Best on NVIDIA |
| **MLX** | macOS | Apple Silicon (M1-M4) | Best on Mac |
| **MPS** | macOS | Apple Silicon | Good fallback |
| **CPU** | Any | None | Slow but works |

Backend is **auto-detected** — no config needed. Override with `--backend`.

---

## 📊 Research Status

| Stage | | Result |
|---|---|---|
| M0 | Environment | Multi-platform support ✅ |
| M1(a) | Titans recall | Memory beyond attention window ✅ |
| M1(b) | In-Place TTT | Loss dropped 45% ✅ |
| M2 | ROCm profile | 19ms/chunk, 6k tok/s ✅ |
| M3+ | Gemma 4 + Intensive TTT | 60% → 90% accuracy ✅ |
| M4 | Visual tokenizer | Tuna-2 2D-RoPE ✅ |
| M5 | Audio tokenizer | Mel-LLM, no STT ✅ |
| M6 | Evaluation | Forgetting≈0, anchor preserved ✅ |
| M7 | Multimodal | Pipeline integrated ✅ |

### Performance (Gemma 4 E2B, RX 6800)

| Operation | Time |
|-----------|------|
| `generate()` | 4921 ms |
| `learn()` | 365 ms (7% overhead) |
| `learn()` + `generate()` | 5169 ms |

---

## 📚 Foundations

- **Nested Learning / Hope** — arXiv 2512.24695 (NeurIPS 2025)
- **Titans** — arXiv 2501.00663
- **In-Place TTT** — arXiv 2604.06169
- **SDFT** — arXiv 2601.19897
- **Tuna-2** (encoder-free vision) — arXiv 2604.24763
- **Mel-LLM** (encoder-free audio) — arXiv 2606.10231

---

## 📄 License

MIT — see [LICENSE](LICENSE)
