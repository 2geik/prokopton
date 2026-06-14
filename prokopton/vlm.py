"""
ProkoptonVL — Prokopton + Gerçek Multimodal VLM (Qwen3-VL)
==========================================================

Encoder-free tokenizer'lar yerine Qwen3-VL'nin kendi vision encoder'ını
kullanır. TTT fast-weight'leri language model katmanlarına uygulanır.

Kullanım:
    from prokopton.vlm import ProkoptonVL
    pvl = ProkoptonVL("Qwen/Qwen3-VL-2B-Instruct")
    pvl.chat({"text": "Bu görselde ne var?", "image": pil_image})
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from PIL import Image

from prokopton.core import FastWeight, CMSAdapter, ProkoptonConfig


class ProkoptonVL:
    """Prokopton with native multimodal VLM backend."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
                 config: ProkoptonConfig = None, device_map: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.config = config or ProkoptonConfig()
        self.step_counter = 0
        self.conversation_history: List[str] = []

        # ── Load model & processor ──
        print(f"📥 Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map
        )
        self.model.eval()

        # ── Find language model ──
        self.lm = self._find_language_model()
        self.hidden_size = self.lm.config.hidden_size
        self.device = next(self.model.parameters()).device

        params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ {params/1e9:.1f}B params | hidden={self.hidden_size} | device={self.device}")

        # ── Setup TTT on language model MLP layers ──
        self.fast_weights: List[FastWeight] = []
        self.cms_adapters: List[CMSAdapter] = []
        self._setup_ttt()

        print(f"   🔧 TTT: {len(self.fast_weights)} katman | CMS: {len(self.cms_adapters)} adaptör")

    def _find_language_model(self):
        """Find the language model inside the VLM."""
        # Qwen3-VL structure: model.language_model
        inner = self.model.model  # Qwen3VLModel

        # Try language_model attribute
        lm = getattr(inner, 'language_model', None)
        if lm is not None and hasattr(lm, 'layers'):
            return lm

        # Try decoder
        lm = getattr(inner, 'decoder', None)
        if lm is not None and hasattr(lm, 'layers'):
            return lm

        # Fallback: inner itself has layers
        if hasattr(inner, 'layers'):
            return inner

        raise ValueError("Cannot find language model in VLM architecture")

    def _setup_ttt(self):
        """Attach TTT fast-weights to language model MLP layers."""
        n_layers = min(self.config.ttt_n_layers, 5)

        # Find MLP down_proj layers
        mlp_layers = []
        for name, module in self.lm.named_modules():
            if 'down_proj' in name and isinstance(module, nn.Linear):
                mlp_layers.append((name, module))

        if not mlp_layers:
            # Fallback: find any linear layer in MLP blocks
            for name, module in self.lm.named_modules():
                if 'mlp' in name.lower() and isinstance(module, nn.Linear):
                    mlp_layers.append((name, module))
                if len(mlp_layers) >= 20:
                    break

        # Select last N layers
        selected = mlp_layers[-n_layers:] if len(mlp_layers) >= n_layers else mlp_layers

        # Multi-frequency CMS
        auto_freqs = [2**i for i in range(len(selected))]

        for i, (name, layer) in enumerate(selected):
            fw = FastWeight(layer, self.config.ttt_lr, self.config.ttt_momentum,
                          self.config.ttt_surprise_threshold)
            self.fast_weights.append(fw)
            cms = CMSAdapter(fw, self.config.cms_rank, self.config.cms_alpha,
                           frequency=auto_freqs[i])
            self.cms_adapters.append(cms)

    def learn(self, text: str) -> Dict[str, Any]:
        """Learn from text using TTT on language model."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        self.lm.train()
        outputs = self.lm(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs.input_ids[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # TTT update
        weights = [fw.layer.weight for fw in self.fast_weights]
        grads = torch.autograd.grad(loss, weights, retain_graph=False)

        for i, fw in enumerate(self.fast_weights):
            eff_lr = fw.effective_lr(loss.item())
            fw.total_surprise += loss.item()
            fw.velocity = fw.momentum * fw.velocity - eff_lr * grads[i]
            with torch.no_grad():
                fw.layer.weight.add_(fw.velocity)
            fw.update_count += 1
            fw._dirty = True

        self.lm.zero_grad()
        self.lm.eval()

        self.step_counter += 1
        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0:
                cms.consolidate()

        return {"loss": loss.item(), "step": self.step_counter}

    def learn_image(self, image: Image.Image, caption: str = None) -> Dict[str, Any]:
        """Learn from an image with optional caption via full multimodal forward."""
        if caption is None:
            caption = "Describe this image in detail."

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": caption}
            ]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(self.device)

        # Full multimodal forward
        self.model.train()
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs.input_ids[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # TTT update on LM MLP weights
        weights = [fw.layer.weight for fw in self.fast_weights]
        grads = torch.autograd.grad(loss, weights, retain_graph=False)

        for i, fw in enumerate(self.fast_weights):
            eff_lr = fw.effective_lr(loss.item())
            fw.total_surprise += loss.item()
            fw.velocity = fw.momentum * fw.velocity - eff_lr * grads[i]
            with torch.no_grad():
                fw.layer.weight.add_(fw.velocity)
            fw.update_count += 1
            fw._dirty = True

        self.model.zero_grad()
        self.model.eval()
        self.step_counter += 1

        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0:
                cms.consolidate()

        return {"loss": loss.item(), "step": self.step_counter}

    def generate(self, text: str, image: Image.Image = None,
                 max_new_tokens: int = 128) -> str:
        """Generate text, optionally conditioning on an image."""
        if image is not None:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]}
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs_kwargs = {"text": prompt, "return_tensors": "pt"}
        if image is not None:
            inputs_kwargs["images"] = [image]
        inputs = self.processor(**inputs_kwargs).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def chat(self, user_input, max_new: int = 128) -> str:
        """Chat: generate response first, then learn from the interaction."""
        if isinstance(user_input, dict):
            text = user_input.get("text", "")
            image = user_input.get("image", None)
            # Generate first (clean model)
            response = self.generate(text, image, max_new)
            # Then learn from this interaction
            if image is not None:
                self.learn_image(image, text)
            else:
                self.learn(text)
        else:
            response = self.generate(user_input, max_new=max_new)
            self.learn(user_input)

        # Extract assistant part
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response

    def save_pretrained(self, path: str):
        """Save full model with TTT weights."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))

        # Save TTT state
        ttt_state = {
            f"ttt_{i}": {
                "velocity": fw.velocity.clone().cpu() if fw.velocity is not None else None,
                "update_count": fw.update_count,
                "total_surprise": fw.total_surprise,
                "weight": fw.layer.weight.data.clone().cpu(),
            }
            for i, fw in enumerate(self.fast_weights)
        }
        torch.save(ttt_state, out / "prokopton_ttt.pt")
        print(f"💾 Kaydedildi: {out}/")

    def reset(self):
        """Reset TTT state."""
        for fw in self.fast_weights:
            fw.reset()
        for cms in self.cms_adapters:
            cms.reset()
        self.step_counter = 0
