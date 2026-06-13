# Prokopton 🧠

> *Prokopton* (προκόπτων): one who continually advances toward wisdom.

A **self-improving LLM** whose weights actually update during conversation —
learning from experience, **growing** over time, and **never forgetting**.
Not RAG or agent-memory — the parameters themselves change at inference time.

- 🔄 **In-Place TTT**: MLP weights updated on every conversation turn
- 💾 **Persistent Memory**: CMS adapters saved to disk, reloaded on restart
- 🎯 **No Forgetting**: Forgetting ≈ 0, anchor knowledge preserved
- 🖼️🎵 **Multimodal**: Vision + direct audio (no STT pipeline)
- 🎮 **TUI Interface**: User-friendly terminal app with Textual
- ⬇️ **HF Model Downloader**: One-click download from any HuggingFace URL

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/2geik/prokopton.git
cd prokopton

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# PyTorch for ROCm (AMD GPU)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# Install Prokopton
pip install -e .
```

> **NVIDIA GPU?** Skip the `--index-url` line — `pip install -e .` pulls stock PyTorch.

### 2. Launch the TUI

```bash
prokopton
# or
python -m prokopton.tui
```

In the TUI:
1. Select a model (type an HF ID or pick from the list)
2. Start chatting — it learns from every message
3. `Ctrl+S` to save memory, `Ctrl+L` to load it back later

### 3. Model Setup

**Option A — Auto-download in the TUI:**
`Ctrl+D` → paste any HF URL or model ID → done

**Option B — Manual `models/` folder:**
```bash
huggingface-cli download google/gemma-4-E2B --local-dir models/gemma-4-E2B
# Any valid HF model folder placed in models/ will appear in the TUI
```

## 📋 TUI Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+S` | Save memory to disk |
| `Ctrl+L` | Load memory from disk |
| `Ctrl+R` | Reset all learned knowledge |
| `Ctrl+M` | Switch model |
| `Ctrl+D` | Download model from HuggingFace |
| `Ctrl+P` | View statistics |
| `Enter` | Send message |

## 🧪 Python API

```python
from prokopton import Prokopton
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
tokenizer.pad_token = tokenizer.eos_token

# Wrap with Prokopton — enables live learning
prok = Prokopton(model, tokenizer)

# Learn from conversation
prok.learn("Zephyria's capital is Aethel.")
prok.save("my_memory")  # persist to disk

# New session — reload what was learned
prok.load("my_memory")
answer = prok.chat("What is the capital of Zephyria?")
print(answer)  # "Aethel"
```

## 📦 Supported Models

| Model | Params | VRAM (bf16) | Notes |
|-------|--------|-------------|-------|
| `google/gemma-4-E2B` | 5.1B | ~9.5 GB | ✅ Recommended |
| `google/gemma-4-E4B` | 7.9B | ~14.2 GB | ⚠️ Tight on 16 GB |
| `google/gemma-4-12B` | 12B | 24+ GB | 🔮 Needs quantization |

> **Other models?** Prokopton works with any HuggingFace `AutoModelForCausalLM` model.
> It auto-detects MLP projection layers for TTT.

## 🖥️ Hardware

- **Recommended:** AMD Radeon RX 6800+ (16 GB VRAM), ROCm 7.0+
- **Minimum:** Any ROCm-compatible AMD GPU
- **CPU fallback:** Works but slow
- **RAM:** 32 GB recommended

## 📊 Status

| Stage | | Result |
|---|---|---|
| M0 | Environment | ROCm 7.2.4 + PyTorch 2.12.0 ✅ |
| M1(a) | Titans recall | Memory beyond attention window ✅ |
| M1(b) | In-Place TTT | Loss dropped 45% ✅ |
| M2 | ROCm profile | 19ms/chunk, 6k tok/s ✅ |
| M3+ | Gemma 4 + Intensive TTT | 60% → 90% accuracy ✅ |
| M4 | Visual tokenizer | Tuna-2 2D-RoPE ✅ |
| M5 | Audio tokenizer | Mel-LLM, no STT ✅ |
| M6 | Evaluation | Forgetting≈0, anchor preserved ✅ |
| M7 | Multimodal | Pipeline integrated ✅ |

## 📚 Research Foundations

- **Nested Learning / Hope** — arXiv 2512.24695 (NeurIPS 2025)
- **Titans** — arXiv 2501.00663
- **In-Place TTT** — arXiv 2604.06169
- **SDFT** — arXiv 2601.19897
- **Tuna-2** (encoder-free vision) — arXiv 2604.24763
- **Mel-LLM** (encoder-free audio) — arXiv 2606.10231

## 📄 License

MIT — see [LICENSE](LICENSE)
