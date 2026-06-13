"""
Prokopton — Self-Improving, Never-Forgetting LLM Framework

Fully integrated: In-Place TTT + CMS + PER + Multimodal + PERSISTENT MEMORY.
Learned knowledge saved as CMS adapters to disk, reloaded on restart.
"""
import torch, torch.nn as nn, math, json, datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ProkoptonConfig:
    """Framework configuration."""
    # TTT
    ttt_lr: float = 1e-3
    ttt_momentum: float = 0.9
    ttt_n_layers: int = 5
    ttt_surprise_threshold: float = 0.0
    # CMS
    cms_frequencies: List[int] = field(default_factory=lambda: [1, 4, 16])
    cms_rank: int = 16
    cms_alpha: float = 32.0
    # PER
    per_capacity: int = 128
    per_sample_size: int = 8
    # Kalıcı bellek
    save_dir: str = "prokopton_memory"
    auto_save_every: int = 100  # her N adımda bir otomatik kaydet
    # Multimodal
    vision_patch_size: int = 48
    vision_embed_dim: int = 1024
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_patch_frames: int = 4


class FastWeight:
    """Single MLP layer's test-time updatable weight."""
    def __init__(self, layer: nn.Linear, lr: float = 1e-3, momentum: float = 0.9):
        self.layer = layer
        self.lr = lr
        self.momentum = momentum
        self.velocity = torch.zeros_like(layer.weight)
        self.original_W = layer.weight.clone()
        self.update_count = 0
        self.total_surprise = 0.0

    def reset(self):
        with torch.no_grad():
            self.layer.weight.copy_(self.original_W)
        self.velocity.zero_()
        self.update_count = 0
        self.total_surprise = 0.0

    @property
    def delta(self) -> torch.Tensor:
        """Current weight minus original weight."""
        return self.layer.weight - self.original_W

    @property
    def weight_change(self) -> float:
        return self.delta.norm().item()


class CMSAdapter:
    """Distills fast-weight delta into low-rank matrices via SVD."""
    def __init__(self, fast_weight: FastWeight, rank: int = 16, alpha: float = 32.0):
        self.fast = fast_weight
        device = fast_weight.layer.weight.device
        dtype = fast_weight.layer.weight.dtype
        in_dim = fast_weight.layer.weight.shape[1]
        out_dim = fast_weight.layer.weight.shape[0]
        self.A = nn.Parameter(torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
        self.rank = rank
        self.alpha = alpha

    def consolidate(self):
        """SVD-distill fast-weight delta into A@B."""
        delta = self.fast.delta.float()
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(delta, q=self.rank)
            self.B.data.copy_((U * S).to(self.B.dtype))
            self.A.data.copy_(V.T.to(self.A.dtype))

    def expand(self) -> torch.Tensor:
        """Expand A@B to full delta matrix."""
        return (self.B @ self.A) * (self.alpha / self.rank)

    def apply_to_model(self):
        """Apply CMS adapter delta back to base model weights."""
        delta = self.expand()
        with torch.no_grad():
            self.fast.original_W.add_(delta.to(self.fast.original_W.dtype))
            self.fast.layer.weight.copy_(self.fast.original_W)
        self.fast.velocity.zero_()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"A": self.A.data.cpu(), "B": self.B.data.cpu()}

    def load_state_dict(self, d: Dict[str, torch.Tensor]):
        self.A.data.copy_(d["A"].to(self.A.device))
        self.B.data.copy_(d["B"].to(self.B.device))


class SurpriseBuffer:
    """Surprise-prioritized replay buffer (PER)."""
    def __init__(self, capacity: int = 128):
        self.items: deque = deque(maxlen=capacity)
        self.surprises: deque = deque(maxlen=capacity)

    def push(self, text: str, surprise: float):
        self.items.append(text)
        self.surprises.append(surprise)

    def sample(self, k: int = 8) -> List[str]:
        if not self.items: return []
        k = min(k, len(self.items))
        probs = torch.tensor(list(self.surprises), dtype=torch.float)
        probs = probs / probs.sum()
        idxs = torch.multinomial(probs, k).tolist()
        return [self.items[i] for i in idxs]

    def __len__(self) -> int: return len(self.items)


# ============================================================
# Multimodal Tokenizer'lar (aynı, kısaltılmış)
# ============================================================

class VisualTokenizer(nn.Module):
    def __init__(self, patch_size=48, embed_dim=1024, output_dim=2560, max_grid=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        half = embed_dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('freqs_x', freqs)
        self.register_buffer('freqs_y', freqs.clone())

    def forward(self, images):
        B, C, H, W = images.shape
        p = self.patch_size
        pad_h, pad_w = (p - H % p) % p, (p - W % p) % p
        if pad_h or pad_w:
            images = nn.functional.pad(images, (0, pad_w, 0, pad_h))
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        gh, gw = patches.shape[1], patches.shape[2]
        patches = patches.view(B, gh * gw, -1)
        x = self.proj(patches)
        # 2D-RoPE
        D = x.shape[-1]
        q = D // 4
        gy, gx = torch.meshgrid(torch.arange(gh, device=x.device).float(),
                                torch.arange(gw, device=x.device).float(), indexing='ij')
        tx = (gx.flatten().unsqueeze(1) * self.freqs_x).repeat_interleave(2, 1)[:, :q]
        ty = (gy.flatten().unsqueeze(1) * self.freqs_y).repeat_interleave(2, 1)[:, :q]
        c = torch.cat([tx.cos(), ty.cos()], 1)
        s = torch.cat([tx.sin(), ty.sin()], 1)
        if c.shape[-1] < D//2:
            c = nn.functional.pad(c, (0, D//2 - c.shape[-1]))
            s = nn.functional.pad(s, (0, D//2 - s.shape[-1]))
        x1, x2 = x[..., :D//2], x[..., D//2:]
        x = torch.cat([x1*c - x2*s, x1*s + x2*c], -1)
        return self.output_proj(x), {"num_tokens": gh * gw, "grid": (gh, gw)}


class AudioTokenizer(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, patch_frames=4, embed_dim=1024, output_dim=2560):
        super().__init__()
        self.sr, self.n_mels, self.pf = sample_rate, n_mels, patch_frames
        self.n_fft, self.hop = 400, 160
        self.input_proj = nn.Linear(patch_frames * n_mels, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        # Mel filterbank
        mel_f = torch.linspace(0, 2595 * math.log10(1 + (sample_rate//2)/700), n_mels + 2)
        hz_f = 700 * (10 ** (mel_f / 2595) - 1)
        self.register_buffer('fft_bins', torch.floor((self.n_fft + 1) * hz_f / sample_rate).long())
        # Positional encoding
        pe = torch.zeros(4096, embed_dim)
        pos = torch.arange(4096).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, waveform):
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
        all_tokens, all_info = [], []
        for b in range(waveform.shape[0]):
            w = waveform[b]
            n_frames = 1 + (len(w) - self.n_fft) // self.hop
            if n_frames <= 0:
                all_tokens.append(torch.zeros(1, self.output_proj.out_features, device=waveform.device))
                all_info.append({"num_tokens": 1, "duration_ms": 0})
                continue
            win = torch.hann_window(self.n_fft, device=w.device)
            frames = torch.stack([w[i*self.hop:i*self.hop+self.n_fft]*win for i in range(n_frames)])
            spec = torch.abs(torch.fft.rfft(frames))[:, :self.n_fft//2+1] ** 2
            mel = torch.zeros(n_frames, self.n_mels, device=w.device)
            for m in range(self.n_mels):
                s, e = self.fft_bins[m].item(), self.fft_bins[m+2].item()
                if e > s and e <= spec.shape[1]:
                    mel[:, m] = spec[:, s:e].mean(-1)
            mel = torch.log(mel + 1e-10)
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            npatch = n_frames // self.pf
            if npatch == 0:
                all_tokens.append(torch.zeros(1, self.output_proj.out_features, device=w.device))
                all_info.append({"num_tokens": 1, "duration_ms": 0})
                continue
            patches = torch.stack([mel[i*self.pf:(i+1)*self.pf].flatten() for i in range(npatch)])
            x = self.input_proj(patches) + self.pe[:npatch]
            all_tokens.append(self.output_proj(x))
            dur = (n_frames * self.hop / self.sr) * 1000
            all_info.append({"num_tokens": npatch, "duration_ms": dur})
        return all_tokens, all_info


# ============================================================
# Prokopton Framework — Ana Sınıf
# ============================================================

class Prokopton:
    """
    Prokopton: Self-improving, never-forgetting LLM framework.

    Usage:
        prok = Prokopton(model, tokenizer)
        prok.learn("Zephyria's capital is Aethel.")
        prok.save()  # persist to disk

        # New session
        prok2 = Prokopton(model2, tokenizer2)
        prok2.load()  # restore previous knowledge
        answer = prok2.chat("What is the capital of Zephyria?")  # "Aethel"
    """

    def __init__(self, model, tokenizer, config: ProkoptonConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ProkoptonConfig()

        self.fast_weights: List[FastWeight] = []
        self.cms_adapters: List[CMSAdapter] = []
        self.replay_buffer = SurpriseBuffer(self.config.per_capacity)
        self.step_counter = 0
        self.conversation_history: List[str] = []
        self.memory: Dict[str, Any] = {}  # metadata

        self._setup_ttt()
        self._setup_multimodal()

    def _setup_ttt(self):
        # Önce Gemma 4 tarzı katmanları ara, yoksa tüm Linear'ları al
        mlp_layers = []
        for name, m in self.model.named_modules():
            if 'language_model.layers' in name and 'mlp.down_proj' in name:
                if hasattr(m, 'weight') and len(m.weight.shape) == 2:
                    mlp_layers.append((name, m))
        
        # Gemma 4 bulunamadıysa, son N Linear katmanı bul
        if not mlp_layers:
            linears = []
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and hasattr(m, 'weight') and m.weight.requires_grad:
                    linears.append((name, m))
            mlp_layers = linears[-self.config.ttt_n_layers:]

        for name, layer in mlp_layers[-self.config.ttt_n_layers:]:
            fw = FastWeight(layer, self.config.ttt_lr, self.config.ttt_momentum)
            self.fast_weights.append(fw)
            cms = CMSAdapter(fw, self.config.cms_rank, self.config.cms_alpha)
            self.cms_adapters.append(cms)

    def _setup_multimodal(self):
        try:
            output_dim = self.model.config.hidden_size
        except (AttributeError, KeyError):
            output_dim = 2560
        self.vision_tokenizer = VisualTokenizer(
            self.config.vision_patch_size, self.config.vision_embed_dim, output_dim)
        self.audio_tokenizer = AudioTokenizer(
            self.config.audio_sample_rate, self.config.audio_n_mels,
            self.config.audio_patch_frames, self.config.vision_embed_dim, output_dim)

    # ============================================================
    # LEARNING
    # ============================================================

    def learn(self, text: str) -> Dict[str, Any]:
        """Learn from a text chunk. Updates TTT fast-weights."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        self.model.train()
        outputs = self.model(**tokens)
        logits = outputs.logits[:, :-1].contiguous()
        targets = tokens["input_ids"][:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))

        # Tek backward ile tüm TTT katmanlarının gradyanı
        weights = [fw.layer.weight for fw in self.fast_weights]
        grads = torch.autograd.grad(loss, weights, retain_graph=False)
        surprise = loss.item()

        for fw, grad in zip(self.fast_weights, grads):
            fw.total_surprise += surprise
            fw.velocity = fw.momentum * fw.velocity - fw.lr * grad
            with torch.no_grad():
                fw.layer.weight.add_(fw.velocity)
            fw.update_count += 1

        self.model.zero_grad()
        self.model.eval()
        self.replay_buffer.push(text, surprise)
        self.step_counter += 1

        # CMS konsolidasyonu (çok-frekanslı)
        for i, freq in enumerate(self.config.cms_frequencies):
            if self.step_counter % freq == 0 and i < len(self.cms_adapters):
                self.cms_adapters[i].consolidate()

        # PER replay
        if self.step_counter % 50 == 0:
            for sample_text in self.replay_buffer.sample(self.config.per_sample_size):
                self.learn(sample_text)

        # Otomatik kaydet
        if self.config.auto_save_every > 0 and self.step_counter % self.config.auto_save_every == 0:
            self.save(silent=True)

        return {"loss": loss.item(), "surprise": surprise, "step": self.step_counter}

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(self, prompt: str, max_new: int = 128) -> str:
        """Generate text (no learning)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                temperature=1.0, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def chat(self, user_input: str, max_new: int = 128) -> str:
        """Chat: learn first, then respond."""
        self.conversation_history.append(f"User: {user_input}")
        context = "\n".join(self.conversation_history[-6:])
        prompt = f"{context}\nAssistant:"

        info = self.learn(user_input)
        response = self.generate(prompt, max_new)
        assistant_part = response.split("Assistant:")[-1].strip()
        self.conversation_history.append(f"Assistant: {assistant_part}")
        return assistant_part

    # ============================================================
    # PERSISTENT MEMORY — save / load
    # ============================================================

    def save(self, path: str = None, silent: bool = False):
        """
        Save all learned knowledge to disk.
        CMS adapters + metadata + configuration.
        """
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each CMS adapter
        cms_data = {}
        for i, cms in enumerate(self.cms_adapters):
            sd = cms.state_dict()
            torch.save(sd, save_dir / f"cms_{i}.pt")
            cms_data[f"layer_{i}"] = {
                "rank": cms.rank,
                "alpha": cms.alpha,
                "shape_A": list(sd["A"].shape),
                "shape_B": list(sd["B"].shape),
                "weight_change": cms.fast.weight_change,
            }

        # Also save fast-weight deltas (for full restore)
        fw_data = {}
        for i, fw in enumerate(self.fast_weights):
            delta_path = save_dir / f"delta_{i}.pt"
            torch.save(fw.delta.cpu(), delta_path)
            fw_data[f"layer_{i}"] = {
                "updates": fw.update_count,
                "surprise": fw.total_surprise,
                "change_norm": fw.weight_change,
            }

        # Metadata
        metadata = {
            "model_type": type(self.model).__name__,
            "steps": self.step_counter,
            "saved_at": datetime.datetime.now().isoformat(),
            "config": asdict(self.config),
            "cms_layers": cms_data,
            "fast_weight_layers": fw_data,
            "history_len": len(self.conversation_history),
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.memory.update(metadata)
        if not silent:
            print(f"💾 Kaydedildi: {save_dir}/ ({len(self.cms_adapters)} CMS, "
                  f"{self.step_counter} adım, {len(self.conversation_history)} mesaj)")

    def load(self, path: str = None, silent_on_missing: bool = False):
        """
        Load previously saved knowledge.
        Loads CMS adapters and applies to base model.
        """
        save_dir = Path(path or self.config.save_dir)
        if not save_dir.exists():
            if not silent_on_missing:
                print(f"⚠ Klasör bulunamadı: {save_dir}")
            return False

        meta_path = save_dir / "metadata.json"
        if not meta_path.exists():
            print("⚠ metadata.json bulunamadı")
            return False

        with open(meta_path) as f:
            metadata = json.load(f)

        # Load CMS adapters
        loaded = 0
        for i in range(len(self.cms_adapters)):
            cms_path = save_dir / f"cms_{i}.pt"
            if cms_path.exists():
                sd = torch.load(cms_path, map_location=self.model.device, weights_only=True)
                self.cms_adapters[i].load_state_dict(sd)
                self.cms_adapters[i].apply_to_model()
                loaded += 1

        # Load fast-weight deltas (fallback, if no CMS)
        if loaded == 0:
            for i in range(len(self.fast_weights)):
                delta_path = save_dir / f"delta_{i}.pt"
                if delta_path.exists():
                    delta = torch.load(delta_path, map_location=self.model.device, weights_only=True)
                    with torch.no_grad():
                        self.fast_weights[i].layer.weight.add_(delta)
                        self.fast_weights[i].original_W = self.fast_weights[i].layer.weight.clone()
                    loaded += 1

        # Restore metadata
        self.step_counter = metadata.get("steps", 0)
        self.memory = metadata

        print(f"📂 Yüklendi: {save_dir}/ ({loaded} katman, "
              f"{metadata.get('steps', 0)} adım, "
              f"{metadata.get('history_len', 0)} mesaj)")
        return True

    # ============================================================
    # UTILITY
    # ============================================================

    def reset(self):
        """Reset all learned knowledge (RAM only)."""
        for fw in self.fast_weights:
            fw.reset()
        self.replay_buffer = SurpriseBuffer(self.config.per_capacity)
        self.step_counter = 0
        self.conversation_history = []

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self.step_counter,
            "updates": sum(fw.update_count for fw in self.fast_weights),
            "weight_change": sum(fw.weight_change for fw in self.fast_weights),
            "total_surprise": sum(fw.total_surprise for fw in self.fast_weights),
            "buffer_size": len(self.replay_buffer),
            "history_len": len(self.conversation_history),
            "saved": "metadata.json" in str(self.memory),
        }
