"""
ProkoptonVL — Prokopton + native multimodal VLM (Qwen3-VL)
==========================================================

Uses the model's own vision tower instead of the deprecated encoder-free
tokenizers. TTT fast-weights are attached to the **language model's** MLP
``down_proj`` modules only.

Key correctness properties:

* **Completion-only loss** — labels are ``-100`` for every image placeholder,
  every chat-template token and the whole system/user prompt. Only assistant /
  caption tokens contribute gradient. A unit test asserts image-token label
  positions are ``-100``.
* **Learn → generate** — ``learn_then_generate()`` (and ``chat(...,
  mode="learn_then_generate")``) learns *before* generating so a successful
  output can be attributed to TTT; ``mode="generate_then_learn"`` is the
  original order and is available for A/B comparison via ``compare_modes()``.
* **Memory** — :meth:`save` / :meth:`load` persist the low-rank CMS adapters,
  matching the text path.

Usage::

    from prokopton.vlm import ProkoptonVL
    pvl = ProkoptonVL("Qwen/Qwen3-VL-4B-Instruct")
    pvl.chat({"text": "What is in this image?", "image": pil_image})
"""

from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prokopton.core import (
    MEMORY_SCHEMA_VERSION,
    CMSAdapter,
    FastWeight,
    ProkoptonConfig,
    select_ttt_layers,
)

DEFAULT_VLM = "Qwen/Qwen3-VL-4B-Instruct"


class ProkoptonVL:
    """Prokopton with a native multimodal VLM backend."""

    def __init__(self, model_id: str = DEFAULT_VLM,
                 config: ProkoptonConfig = None, device_map: str = "auto",
                 dtype=torch.float16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.config = config or ProkoptonConfig()
        self.step_counter = 0
        self.skipped_steps = 0
        self.conversation_history: List[Dict[str, str]] = []

        print(f"📥 Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        self.lm = self._find_language_model()
        self.hidden_size = self.lm.config.hidden_size
        self.device = next(self.model.parameters()).device

        params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ {params / 1e9:.1f}B params | hidden={self.hidden_size} | device={self.device}")

        self.fast_weights: List[FastWeight] = []
        self.cms_adapters: List[CMSAdapter] = []
        self._setup_ttt()

        print(f"   🔧 TTT: {len(self.fast_weights)} layers | CMS: {len(self.cms_adapters)} adapters")
        print(f"      {self.ttt_layer_names}")

    # ── Architecture ──────────────────────────────────────────
    def _find_language_model(self):
        """Find the language model inside the VLM."""
        inner = getattr(self.model, "model", self.model)
        for attr in ("language_model", "decoder", "text_model"):
            lm = getattr(inner, attr, None)
            if lm is not None and hasattr(lm, "layers"):
                return lm
        if hasattr(inner, "layers"):
            return inner
        raise ValueError("Cannot find language model in VLM architecture")

    def _setup_ttt(self):
        """Attach TTT fast-weights to the language model's MLP layers only."""
        n_layers = min(self.config.ttt_n_layers, 5)

        # Search the whole model so the names carry the `language_model` scope,
        # then let `select_ttt_layers` reject anything in a vision/audio tower.
        selected = select_ttt_layers(self.model, n_layers)

        auto_freqs = [max(1, int(self.config.cms_interval_base)) * 2 ** i
                      for i in range(len(selected))]
        for i, (name, layer) in enumerate(selected):
            fw = FastWeight(
                layer,
                lr=self.config.ttt_lr,
                momentum=self.config.ttt_momentum,
                surprise_threshold=self.config.ttt_surprise_threshold,
                surprise_warmup=self.config.ttt_surprise_warmup,
                rank=self.config.ttt_rank,
                parametrization=self.config.ttt_parametrization,
                trust_region=self.config.ttt_trust_region,
                decay=self.config.ttt_decay,
                grad_clip=self.config.ttt_grad_clip,
                optimizer=self.config.ttt_optimizer,
            )
            fw.layer_name = name
            self.fast_weights.append(fw)
            self.cms_adapters.append(
                CMSAdapter(fw, self.config.cms_rank, self.config.cms_alpha,
                           frequency=auto_freqs[i]))
        self.ttt_layer_names = [n for n, _ in selected]

    # ── Label construction (Phase 4.3) ────────────────────────
    def build_training_inputs(self, prompt_messages: List[dict],
                              target: str, image=None, max_length: int = 512):
        """Tokenise ``prompt + target`` and mask the loss to the target only.

        Returns ``(inputs, labels)`` where ``labels[j] == input_ids[j]`` for the
        target span and ``-100`` everywhere else — in particular at every image
        placeholder and every chat-template token.
        """
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + target

        images = [image] if image is not None else None
        prompt_inputs = self.processor(
            text=prompt_text, images=images, return_tensors="pt",
            truncation=True, max_length=max_length)
        full_inputs = self.processor(
            text=full_text, images=images, return_tensors="pt",
            truncation=True, max_length=max_length)

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)
        labels[0, prompt_len:] = input_ids[0, prompt_len:]

        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
        labels = labels.to(self.device)
        return full_inputs, labels

    def _check_labels(self, input_ids: torch.Tensor, labels: torch.Tensor,
                      image_token_id: Optional[int] = None) -> Dict[str, int]:
        """Verify the masking invariants (used by tests and debug asserts)."""
        if image_token_id is not None:
            img_pos = (input_ids == image_token_id).nonzero(as_tuple=True)[1]
            unmasked_images = int((labels[0, img_pos] != -100).sum().item())
        else:
            unmasked_images = 0
        n_supervised = int((labels != -100).sum().item())
        return {"supervised_tokens": n_supervised,
                "unmasked_image_tokens": unmasked_images}

    # ── Learning ──────────────────────────────────────────────
    def _learn_from_labels(self, inputs: Dict[str, torch.Tensor],
                           labels: torch.Tensor) -> Dict[str, Any]:
        self.model.train()
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100)
        surprise = float(loss.item())

        weights = [fw.layer.weight for fw in self.fast_weights]
        grads = torch.autograd.grad(loss, weights, retain_graph=False, allow_unused=True)

        skipped = 0
        for fw, grad in zip(self.fast_weights, grads):
            if grad is None:
                grad = torch.zeros_like(fw.layer.weight)
            if fw.apply_grad(grad, surprise)["skipped"]:
                skipped += 1

        self.model.zero_grad()
        self.model.eval()
        if skipped == len(self.fast_weights):
            self.skipped_steps += 1

        self.step_counter += 1
        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0 and cms.fast.is_dirty:
                cms.consolidate()

        return {"loss": surprise, "surprise": surprise, "step": self.step_counter,
                "skipped": skipped,
                "drift": max((fw.drift_ratio for fw in self.fast_weights), default=0.0)}

    def learn(self, text: str, target: Optional[str] = None) -> Dict[str, Any]:
        """Learn from text. ``target`` defaults to the text itself (fact form)."""
        target = text if target is None else target
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs, labels = self.build_training_inputs(messages, target)
        return self._learn_from_labels(inputs, labels)

    def learn_image(self, image, caption: str = None,
                    instruction: str = "Describe this image in detail.") -> Dict[str, Any]:
        """Learn from an image with a caption.

        Loss is restricted to the caption tokens; image placeholders and the
        chat template are masked out.
        """
        caption = caption or instruction
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}]
        inputs, labels = self.build_training_inputs(messages, caption, image=image)
        return self._learn_from_labels(inputs, labels)

    # ── Generation ────────────────────────────────────────────
    def generate(self, text: str, image=None, max_new_tokens: int = 128) -> str:
        """Generate only the new tokens (prompt is never echoed)."""
        if image is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        kwargs = {"text": prompt, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def generate_completion(self, text: str, image=None, max_new_tokens: int = 128) -> str:
        """Raw generation: prompt + completion."""
        if image is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        kwargs = {"text": prompt, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def learn_then_generate(self, user_input, max_new: int = 128) -> str:
        """Learn first, then generate.

        Use this when you need to attribute a successful output to TTT — the
        weights already contain the new information at generation time.
        """
        text, image = self._split_input(user_input)
        if image is not None:
            self.learn_image(image, text)
        else:
            self.learn(text)
        return self.generate(text, image, max_new)

    def chat(self, user_input, max_new: int = 128,
             mode: str = "generate_then_learn") -> str:
        """Chat with an explicit learning/generation order.

        ``mode="generate_then_learn"`` (default, legacy): respond with the clean
        model, then learn from the interaction.

        ``mode="learn_then_generate"``: learn first so the response reflects the
        update — required for attribution.
        """
        if mode == "learn_then_generate":
            return self.learn_then_generate(user_input, max_new)
        if mode != "generate_then_learn":
            raise ValueError(f"unknown mode {mode!r}")

        text, image = self._split_input(user_input)
        response = self.generate(text, image, max_new)
        if image is not None:
            self.learn_image(image, text)
        else:
            self.learn(text)
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def compare_modes(self, user_input, max_new: int = 128) -> Dict[str, Any]:
        """A/B: what does TTT change? Runs both orders on a snapshot state."""
        snapshot = self.snapshot_state()
        try:
            baseline = self.generate(*self._split_input(user_input), max_new_tokens=max_new)
            after = self.learn_then_generate(user_input, max_new)
            return {"generate_then_learn": baseline, "learn_then_generate": after,
                    "changed": baseline != after}
        finally:
            self.restore_state(snapshot)

    @staticmethod
    def _split_input(user_input) -> Tuple[str, Optional[Any]]:
        if isinstance(user_input, dict):
            return user_input.get("text", ""), user_input.get("image", None)
        return user_input, None

    # ── Persistence (Phase 4.2) ───────────────────────────────
    def save(self, path: str, silent: bool = False) -> Path:
        """Persist low-rank CMS adapters + conversation state (no full dump)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for i, cms in enumerate(self.cms_adapters):
            cms.consolidate()
            torch.save(cms.state_dict(), out / f"cms_{i}.pt")
            cms.mark_clean()
        metadata = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint(),
            "ttt_layer_names": list(self.ttt_layer_names),
            "steps": self.step_counter,
            "skipped_steps": self.skipped_steps,
            "conversation_history": self.conversation_history,
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {},
            "saved_at": datetime.datetime.now().isoformat(),
        }
        tmp = out / "metadata.json.tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        tmp.replace(out / "metadata.json")
        if not silent:
            print(f"💾 ProkoptonVL memory saved: {out}/")
        return out

    def load(self, path: str, silent: bool = False) -> bool:
        """Restore CMS adapters. Idempotent — never compounds the delta."""
        out = Path(path)
        meta_path = out / "metadata.json"
        if not meta_path.exists():
            if not silent:
                print(f"⚠ No memory at {out}")
            return False
        with open(meta_path) as f:
            metadata = json.load(f)

        fingerprint = metadata.get("model_fingerprint")
        if fingerprint and fingerprint != self.model_fingerprint():
            raise ValueError(
                f"Memory fingerprint {fingerprint} does not match this model "
                f"({self.model_fingerprint()}); refusing to load.")

        loaded = 0
        for i, cms in enumerate(self.cms_adapters):
            p = out / f"cms_{i}.pt"
            if p.exists():
                cms.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                cms.apply_to_model()
                loaded += 1

        self.step_counter = int(metadata.get("steps", 0))
        self.skipped_steps = int(metadata.get("skipped_steps", 0))
        self.conversation_history = list(metadata.get("conversation_history", []))
        if not silent:
            print(f"📂 ProkoptonVL memory loaded: {out}/ ({loaded} adapters)")
        return True

    def model_fingerprint(self) -> str:
        import hashlib
        parts = [f"{n}:{tuple(fw.W0.shape)}" for n, fw in
                 zip(self.ttt_layer_names, self.fast_weights)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def save_pretrained(self, path: str):
        """Commit all learned deltas into the base model and save in full."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for cms in self.cms_adapters:
            cms.consolidate()
            cms.commit()
        self.model.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))
        print(f"💾 Merged model saved: {out}/")

    # ── State snapshot (for A/B comparisons) ──────────────────
    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "weights": [fw.layer.weight.detach().clone() for fw in self.fast_weights],
            "factors": [fw.get_factors() if fw.parametrization == "lowrank" else None
                        for fw in self.fast_weights],
            "steps": self.step_counter,
        }

    def restore_state(self, snap: Dict[str, Any]):
        for fw, w, fac in zip(self.fast_weights, snap["weights"], snap["factors"]):
            with torch.no_grad():
                fw.layer.weight.copy_(w)
            if fac is not None:
                fw.set_factors(*fac)
        self.step_counter = snap["steps"]

    # ── Utility ───────────────────────────────────────────────
    def reset(self):
        """Reset TTT state. ``W0`` is untouched."""
        for fw in self.fast_weights:
            fw.reset()
        for cms in self.cms_adapters:
            cms.mark_clean()
        self.step_counter = 0
        self.skipped_steps = 0
        self.conversation_history = []

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self.step_counter,
            "skipped_steps": self.skipped_steps,
            "updates": sum(fw.update_count for fw in self.fast_weights),
            "weight_change": sum(fw.weight_change for fw in self.fast_weights),
            "max_drift_ratio": max((fw.drift_ratio for fw in self.fast_weights), default=0.0),
            "trust_region": self.config.ttt_trust_region,
            "layers": list(self.ttt_layer_names),
        }
