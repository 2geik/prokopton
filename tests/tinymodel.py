"""Tiny language-model test doubles.

The models here deliberately mirror the module naming of a real multimodal
transformer (``model.language_model.layers.N.mlp.down_proj`` plus a vision
tower) so the layer-selection rules and the TTT plumbing can be exercised
without downloading a multi-billion-parameter checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        return x + self.mlp.down_proj(F.gelu(self.mlp.gate_proj(self.ln(x))))


class _LanguageModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, intermediate: int, n_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(_Block(hidden, intermediate) for _ in range(n_layers))


class _VisualTower(nn.Module):
    """A vision tower that must never be selected for TTT."""

    def __init__(self, hidden: int, intermediate: int, n_layers: int = 2):
        super().__init__()
        self.patch_embed = nn.Linear(3 * 8 * 8, hidden, bias=False)
        self.layers = nn.ModuleList(_Block(hidden, intermediate) for _ in range(n_layers))


class TinyConfig:
    def __init__(self, hidden_size: int, vocab_size: int, num_hidden_layers: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.model_type = "tiny_lm"
        self.text_config = self

    def __getattr__(self, name):
        # Multimodal probes (vision_config / audio_config) are absent.
        raise AttributeError(name)


class TinyVLModel(nn.Module):
    """A tiny multimodal-shaped decoder with a vision tower and an LM tower."""

    def __init__(self, vocab: int = 128, hidden: int = 32, intermediate: int = 64,
                 n_layers: int = 4, with_visual: bool = True):
        super().__init__()
        self.config = TinyConfig(hidden, vocab, n_layers)
        self.model = nn.Module()
        self.model.language_model = _LanguageModel(vocab, hidden, intermediate, n_layers)
        if with_visual:
            self.model.visual = _VisualTower(hidden, intermediate)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._device = torch.device("cpu")

    # -- torch-ish surface -------------------------------------------------
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            h = self.get_input_embeddings()(input_ids)
        else:
            h = inputs_embeds
        for layer in self.model.language_model.layers:
            h = layer(h)
        return _Output(self.lm_head(h))

    @torch.no_grad()
    def generate(self, input_ids=None, max_new_tokens: int = 8, **kwargs):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=ids).logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
        return ids


class _Output:
    def __init__(self, logits):
        self.logits = logits


class TinyTokenizer:
    """Character-level tokenizer with a chat template.

    Deterministic and dependency-free; ``encode``/``decode`` round-trip exactly
    so tests can reason about token boundaries.
    """

    def __init__(self, vocab: int = 128, use_chat_template: bool = True):
        self.vocab = vocab
        self.eos_token = "</s>"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.use_chat_template = use_chat_template
        self._pad = self.eos_token

    @property
    def pad_token(self):
        return self._pad

    @pad_token.setter
    def pad_token(self, v):
        self._pad = v

    def _ids(self, text: str):
        # ASCII round-trips exactly: id == ord(c). Anything outside the small
        # printable range collapses to '?' so ids always stay in vocab.
        return [ord(c) if 1 <= ord(c) < self.vocab else ord("?") for c in text]

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None,
                 padding=False, **kwargs):
        texts = text if isinstance(text, (list, tuple)) else [text]
        rows = [self._ids(t) for t in texts]
        if truncation and max_length:
            rows = [r[:max_length] for r in rows]
        if padding or len(rows) > 1:
            width = max(len(r) for r in rows)
            rows = [r + [self.pad_token_id] * (width - len(r)) for r in rows]
        ids = torch.tensor(rows, dtype=torch.long)
        if return_tensors is None:
            return {"input_ids": ids.tolist()}
        return {"input_ids": ids, "attention_mask": (ids != self.pad_token_id).long()}

    def encode(self, text, **kwargs):
        return self._ids(text)

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(i) for i in ids if i not in (0,))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not self.use_chat_template:
            raise AttributeError("no chat template")
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"<|{m.get('role', 'user')}|> {content}")
        out = "\n".join(parts)
        if add_generation_prompt:
            out += "\n<|assistant|> "
        return out


def tiny_model_and_tokenizer(vocab=128, hidden=32, intermediate=64, n_layers=4,
                             with_visual=True, use_chat_template=True):
    torch.manual_seed(0)
    model = TinyVLModel(vocab=vocab, hidden=hidden, intermediate=intermediate,
                        n_layers=n_layers, with_visual=with_visual)
    tok = TinyTokenizer(vocab=vocab, use_chat_template=use_chat_template)
    return model, tok
