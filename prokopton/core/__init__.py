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
    max_audio_tokens: int = 12  # safety cap: prevents OOM on Gemma-4 text-only
    max_visual_tokens: int = 100  # safety cap for vision tokens


class FastWeight:
    """Single MLP layer's test-time updatable weight with surprise-gating."""

    def __init__(self, layer: nn.Linear, lr: float = 1e-3, momentum: float = 0.9,
                 surprise_threshold: float = 0.0, surprise_cap: float = 5.0):
        self.layer = layer
        self.lr = lr
        self.momentum = momentum
        self.surprise_threshold = surprise_threshold
        self.surprise_cap = surprise_cap
        self.velocity = torch.zeros_like(layer.weight)
        self.original_W = layer.weight.clone()
        self.update_count = 0
        self.total_surprise = 0.0
        self.running_surprise = 1.0  # EMA for normalization
        self._dirty = False
        self._last_saved_delta_norm = 0.0

    def reset(self):
        with torch.no_grad():
            self.layer.weight.copy_(self.original_W)
        self.velocity.zero_()
        self.update_count = 0
        self.total_surprise = 0.0
        self.running_surprise = 1.0
        self._dirty = False
        self._last_saved_delta_norm = 0.0

    def effective_lr(self, surprise: float) -> float:
        """Surprise-gated adaptive learning rate.

        Scales LR by surprise ratio with cap to prevent explosions.
        Surprise below threshold → zero update (skip).
        """
        if surprise < self.surprise_threshold:
            return 0.0
        # Running EMA + scale with cap
        self.running_surprise = 0.99 * self.running_surprise + 0.01 * surprise
        scale = min(surprise / (self.running_surprise + 1e-8), self.surprise_cap)
        return self.lr * scale

    @property
    def delta(self) -> torch.Tensor:
        """Current weight minus original weight."""
        return self.layer.weight - self.original_W

    @property
    def weight_change(self) -> float:
        return self.delta.norm().item()

    @property
    def is_dirty(self) -> bool:
        """True if weights changed significantly since last save."""
        if not self._dirty:
            return False
        return self.weight_change > self._last_saved_delta_norm * 0.01

    def mark_clean(self):
        """Mark as saved — update the clean baseline."""
        self._last_saved_delta_norm = self.weight_change
        self._dirty = False


class CMSAdapter:
    """Distills fast-weight delta into low-rank matrices via SVD."""

    def __init__(self, fast_weight: 'FastWeight', rank: int = 16, alpha: float = 32.0,
                 frequency: int = 1):
        self.fast = fast_weight
        device = fast_weight.layer.weight.device
        dtype = fast_weight.layer.weight.dtype
        in_dim = fast_weight.layer.weight.shape[1]
        out_dim = fast_weight.layer.weight.shape[0]
        self.A = nn.Parameter(torch.randn(rank, in_dim, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
        self.rank = rank
        self.alpha = alpha
        self.frequency = frequency
        self._dirty = False

    def consolidate(self):
        """SVD-distill fast-weight delta into A@B."""
        delta = self.fast.delta.float()
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(delta, q=self.rank)
            self.B.data.copy_((U * S).to(self.B.dtype))
            self.A.data.copy_(V.T.to(self.A.dtype))
        self._dirty = True

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

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False

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
    """Encoder-free audio tokenizer — Mel-spectrogram patches → LLM token space.

    Optimized v2: vectorized Mel filterbank (matrix multiply instead of Python loop),
    unfold-based STFT framing, and reshape-based patching.
    """

    def __init__(self, sample_rate=16000, n_mels=80, patch_frames=4, embed_dim=1024, output_dim=2560):
        super().__init__()
        self.sr, self.n_mels, self.pf = sample_rate, n_mels, patch_frames
        self.n_fft, self.hop = 400, 160
        self.input_proj = nn.Linear(patch_frames * n_mels, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # ── Vectorized Mel filterbank ──
        n_freq_bins = self.n_fft // 2 + 1
        mel_f = torch.linspace(0, 2595 * math.log10(1 + (sample_rate // 2) / 700), n_mels + 2)
        hz_f = 700 * (10 ** (mel_f / 2595) - 1)
        fft_bins = torch.floor((self.n_fft + 1) * hz_f / sample_rate).long()

        # Build sparse weight matrix: mel_weights[freq_bin, mel_band] = 1/width
        mel_weights = torch.zeros(n_freq_bins, n_mels)
        for m in range(n_mels):
            s, e = fft_bins[m].item(), fft_bins[m + 2].item()
            s = max(0, s)
            e = min(e, n_freq_bins)
            if e > s:
                mel_weights[s:e, m] = 1.0 / (e - s)
        self.register_buffer('mel_weights', mel_weights)
        self.register_buffer('hann_window', torch.hann_window(self.n_fft))

        # ── Positional encoding ──
        pe = torch.zeros(4096, embed_dim)
        pos = torch.arange(4096).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, waveform):
        """Convert audio waveform to LLM-compatible token sequence.

        Args:
            waveform: [samples] or [B, samples] float tensor.

        Returns:
            all_tokens: list of [N_i, output_dim] tensors per batch item.
            all_info: list of dicts with num_tokens and duration_ms.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device
        all_tokens, all_info = [], []

        for b in range(waveform.shape[0]):
            w = waveform[b]
            w_len = len(w)
            n_frames = 1 + (w_len - self.n_fft) // self.hop

            if n_frames <= 0:
                all_tokens.append(torch.zeros(1, self.output_proj.out_features, device=device))
                all_info.append({"num_tokens": 1, "duration_ms": 0})
                continue

            # ── Vectorized STFT framing via unfold ──
            # unfold: [samples] → [n_frames, n_fft]
            frames = w.unfold(0, self.n_fft, self.hop)[:n_frames]
            frames = frames * self.hann_window.to(device)

            # ── Magnitude spectrogram ──
            spec = torch.abs(torch.fft.rfft(frames))[:, :self.n_fft // 2 + 1] ** 2

            # ── Vectorized Mel filterbank (matrix multiply!) ──
            # spec: [n_frames, n_freqs] @ mel_weights: [n_freqs, n_mels] → [n_frames, n_mels]
            mel = spec @ self.mel_weights.to(device)
            mel = torch.log(mel + 1e-10)
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            # ── Patchify via reshape (no list comprehension) ──
            npatch = n_frames // self.pf
            if npatch == 0:
                all_tokens.append(torch.zeros(1, self.output_proj.out_features, device=device))
                all_info.append({"num_tokens": 1, "duration_ms": 0})
                continue

            # mel[:npatch*pf] → [npatch, pf, n_mels] → [npatch, pf*n_mels]
            patches = mel[:npatch * self.pf].reshape(npatch, self.pf * self.n_mels)

            x = self.input_proj(patches) + self.pe[:npatch].to(device)
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

        n_layers = min(self.config.ttt_n_layers, len(mlp_layers))
        selected = mlp_layers[-n_layers:]

        # Multi-frequency CMS: her katmana log2-artan frekans ata
        # Layer 0 → her adım (1), Layer 1 → 2 adımda bir, Layer 2 → 4, ...
        if len(self.config.cms_frequencies) < n_layers:
            # Auto-expand: power-of-2 pattern
            auto_freqs = [2**i for i in range(n_layers)]
        else:
            auto_freqs = list(self.config.cms_frequencies[:n_layers])

        for i, (name, layer) in enumerate(selected):
            fw = FastWeight(layer, self.config.ttt_lr, self.config.ttt_momentum,
                          self.config.ttt_surprise_threshold)
            self.fast_weights.append(fw)
            cms = CMSAdapter(fw, self.config.cms_rank, self.config.cms_alpha,
                           frequency=auto_freqs[i])
            self.cms_adapters.append(cms)

    def _setup_multimodal(self):
        """Initialize multimodal tokenizers + detect model capabilities."""
        # ── Resolve hidden dimension ──
        try:
            output_dim = self.model.config.hidden_size
        except (AttributeError, KeyError):
            try:
                output_dim = self.model.config.text_config.hidden_size
            except (AttributeError, KeyError):
                output_dim = 2560

        # ── Model dtype & device ──
        try:
            model_dtype = self.model.dtype
        except AttributeError:
            model_dtype = next(self.model.parameters()).dtype
        try:
            model_device = self.model.device
        except AttributeError:
            model_device = next(self.model.parameters()).device

        # ── Detect model capabilities ──
        self._model_info = self._detect_model_type()

        # ── Create tokenizers ──
        self.vision_tokenizer = VisualTokenizer(
            self.config.vision_patch_size, self.config.vision_embed_dim, output_dim)
        self.audio_tokenizer = AudioTokenizer(
            self.config.audio_sample_rate, self.config.audio_n_mels,
            self.config.audio_patch_frames, self.config.vision_embed_dim, output_dim)

        # Move tokenizers to model device (keep float32 — FFT needs it)
        self.vision_tokenizer = self.vision_tokenizer.to(device=model_device)
        self.audio_tokenizer = self.audio_tokenizer.to(device=model_device)

    def _detect_model_type(self) -> Dict[str, Any]:
        """Detect model architecture and multimodal capabilities.

        Returns dict with keys:
            - is_gemma4: bool — Gemma 4 family
            - is_multimodal: bool — native multimodal support
            - has_text_config: bool — separate text_config (Gemma 4 style)
            - model_name: str — best-guess model identifier
            - oom_risk_multimodal: bool — known OOM risk with multimodal forward
        """
        config = self.model.config
        info = {
            "is_gemma4": False,
            "is_multimodal": False,
            "has_text_config": False,
            "model_name": getattr(config, 'model_type', 'unknown'),
            "oom_risk_multimodal": False,
        }

        # Detect Gemma 4
        model_type = getattr(config, 'model_type', '')
        if 'gemma4' in model_type or 'gemma_4' in model_type:
            info["is_gemma4"] = True

        # Detect separate text_config (Gemma 4 style)
        if hasattr(config, 'text_config'):
            info["has_text_config"] = True

        # Detect multimodal: Gemma 4 Unified has vision_config + audio_config
        has_vision = hasattr(config, 'vision_config') or hasattr(config, 'visual_config')
        has_audio = hasattr(config, 'audio_config') or hasattr(config, 'speech_config')
        if has_vision or has_audio:
            info["is_multimodal"] = True

        # Gemma 4 text-only models have OOM risk with multimodal forward
        # because internal get_per_layer_inputs does O(token²) broadcast
        if info["is_gemma4"] and not info["is_multimodal"]:
            info["oom_risk_multimodal"] = True

        return info

    # ============================================================
    # LEARNING
    # ============================================================

    def learn(self, text: str) -> Dict[str, Any]:
        """Learn from a text chunk. Updates TTT fast-weights with surprise-gated LR."""
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
            eff_lr = fw.effective_lr(surprise)  # Surprise-gated adaptive LR
            fw.total_surprise += surprise
            fw.velocity = fw.momentum * fw.velocity - eff_lr * grad
            with torch.no_grad():
                fw.layer.weight.add_(fw.velocity)
            fw.update_count += 1
            fw._dirty = True  # Mark dirty for incremental save

        self.model.zero_grad()
        self.model.eval()
        self.replay_buffer.push(text, surprise)
        self.step_counter += 1

        # Multi-frequency CMS konsolidasyonu — her adaptör kendi sıklığında
        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0:
                cms.consolidate()

        # PER replay
        if self.step_counter % 50 == 0:
            for sample_text in self.replay_buffer.sample(self.config.per_sample_size):
                self.learn(sample_text)

        # Otomatik kaydet
        if self.config.auto_save_every > 0 and self.step_counter % self.config.auto_save_every == 0:
            self.save(silent=True)

        return {"loss": loss.item(), "surprise": surprise, "step": self.step_counter}

    # ============================================================
    # MULTIMODAL LEARNING
    # ============================================================

    def _prepare_multimodal_embeds(
        self, text: str = None, image_tensor: torch.Tensor = None,
        waveform: torch.Tensor = None
    ):
        """Build combined embeddings, attention mask, and labels for multimodal forward.

        CRITICAL ORDERING: Multimodal tokens (audio, visual) come BEFORE text.
        This ensures the loss on text token predictions creates a gradient path
        through the tokenizer projection weights — the model must learn to
        interpret audio/visual features to predict text correctly.

        Layout: [AUDIO...] [VISUAL...] [TEXT...]
        Labels:  -100      -100       target_ids
                                        ↑
        Prediction at last audio position → first text token → grad flows back!
        """
        device = self.model.device
        dtype = self.model.dtype
        embeds_list, mask_list = [], []
        text_ids_for_labels = None
        n_multimodal_prefix = 0  # tracks how many tokens come before text

        # ── 1. Audio tokens FIRST (model must process these before text) ──
        max_aud = self.config.max_audio_tokens
        if self._model_info["oom_risk_multimodal"]:
            safe_max = min(max_aud, 15)
            max_aud = safe_max

        if waveform is not None:
            waveform = waveform.to(device=device)
            aud_tokens_list, aud_info = self.audio_tokenizer(waveform)
            n_trimmed = 0
            if isinstance(aud_tokens_list, list):
                for idx, at in enumerate(aud_tokens_list):
                    if at.dim() == 2:
                        at = at.unsqueeze(0)
                    if at.shape[1] > max_aud:
                        n_trimmed += at.shape[1] - max_aud
                        at = at[:, :max_aud, :]
                    aud_mask = torch.ones(at.shape[0], at.shape[1], device=device)
                    embeds_list.append(at.to(dtype))
                    mask_list.append(aud_mask)
                    n_multimodal_prefix += at.shape[1]
            else:
                if aud_tokens_list.shape[1] > max_aud:
                    n_trimmed = aud_tokens_list.shape[1] - max_aud
                    aud_tokens_list = aud_tokens_list[:, :max_aud, :]
                aud_mask = torch.ones(aud_tokens_list.shape[0], aud_tokens_list.shape[1], device=device)
                embeds_list.append(aud_tokens_list.to(dtype))
                mask_list.append(aud_mask)
                n_multimodal_prefix += aud_tokens_list.shape[1]

            if n_trimmed > 0:
                import warnings
                warnings.warn(
                    f"Audio tokens trimmed by {n_trimmed} (→ {max_aud} max). "
                    f"Model '{self._model_info['model_name']}' "
                    f"{'has known OOM risk with multimodal forward' if self._model_info['oom_risk_multimodal'] else 'exceeded token limit'}. "
                    f"Use shorter audio (≤{max_aud * self.config.audio_patch_frames * self.config.audio_sample_rate / self.config.audio_n_mels:.0f}ms) "
                    f"or increase max_audio_tokens if model supports it.",
                    RuntimeWarning
                )

        # ── 2. Visual tokens SECOND ──
        max_vis = self.config.max_visual_tokens
        if image_tensor is not None:
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device=device)
            vis_tokens, _vis_info = self.vision_tokenizer(image_tensor)
            if vis_tokens.shape[1] > max_vis:
                vis_tokens = vis_tokens[:, :max_vis, :]
            vis_mask = torch.ones(vis_tokens.shape[0], vis_tokens.shape[1], device=device)
            embeds_list.append(vis_tokens.to(dtype))
            mask_list.append(vis_mask)
            n_multimodal_prefix += vis_tokens.shape[1]

        # ── 3. Text embeddings LAST (labels target these) ──
        if text is not None:
            tokens = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=256)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            text_ids = tokens["input_ids"]
            text_mask = tokens["attention_mask"]
            embed_layer = self.model.get_input_embeddings()
            text_embeds = embed_layer(text_ids).to(dtype)
            embeds_list.append(text_embeds)
            mask_list.append(text_mask)
            text_ids_for_labels = text_ids

        # ── Combine ──
        if not embeds_list:
            raise ValueError("En az bir modalite (text, image, audio) gerekli")

        combined_embeds = torch.cat(embeds_list, dim=1)        # [1, total, D]
        combined_mask = torch.cat(mask_list, dim=1)            # [1, total]

        # ── Build labels ──
        # Labels: -100 for multimodal prefix positions, text token IDs for text positions.
        # The critical part: logits at the LAST multimodal position predict the FIRST
        # text token → gradient flows from text loss back through tokenizer weights!
        total_len = combined_embeds.shape[1]
        text_len = text_ids_for_labels.shape[1] if text_ids_for_labels is not None else 0

        labels = torch.full((1, total_len - 1), -100, dtype=torch.long, device=device)
        if text_len > 1:
            # text tokens start at position 'n_multimodal_prefix'
            # logits[i] predicts token at position i+1
            # We need labels for logits positions where the target is a text token
            text_start_in_labels = n_multimodal_prefix - 1  # logit at last mm position → first text token
            if text_start_in_labels < 0:
                text_start_in_labels = 0
            copy_len = min(text_len - 1, total_len - 1 - text_start_in_labels)
            if copy_len > 0:
                labels[0, text_start_in_labels:text_start_in_labels + copy_len] = \
                    text_ids_for_labels[0, 1:copy_len + 1]

        return combined_embeds, combined_mask, labels

    def _multimodal_learn_forward(
        self, text: str, image_tensor: torch.Tensor, waveform: torch.Tensor
    ) -> Dict[str, Any]:
        """Core multimodal forward: build embeds → forward → TTT update → return.

        Shared by learn_image / learn_audio / learn_multimodal.

        Key difference from text-only learn(): tokenizer projection weights
        (audio.input_proj, audio.output_proj, vision.input_proj, vision.output_proj)
        are ALSO included in the TTT gradient update. This means the tokenizers
        LEARN to map acoustic/visual features into the model's embedding space
        through repeated labeled examples.
        """
        embeds, attn_mask, labels = self._prepare_multimodal_embeds(
            text, image_tensor, waveform)

        self.model.train()
        outputs = self.model(inputs_embeds=embeds, attention_mask=attn_mask)
        logits = outputs.logits[:, :-1, :].contiguous()        # [1, total-1, V]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # ── Collect all trainable weights for TTT update ──
        # Layer 0..N-1: MLP fast-weights (always)
        # Extra: tokenizer projection layers (only when their modality is active)
        weights = [fw.layer.weight for fw in self.fast_weights]
        weight_labels = [f"mlp_l{i}" for i in range(len(self.fast_weights))]

        # Audio tokenizer projections — only if audio was provided
        has_audio = waveform is not None
        if has_audio:
            weights.append(self.audio_tokenizer.output_proj.weight)
            weight_labels.append("audio_out")
            weights.append(self.audio_tokenizer.input_proj.weight)
            weight_labels.append("audio_in")

        # Vision tokenizer projections — only if image was provided
        has_vision = image_tensor is not None
        if has_vision:
            weights.append(self.vision_tokenizer.output_proj.weight)
            weight_labels.append("vision_out")
            weights.append(self.vision_tokenizer.proj.weight)
            weight_labels.append("vision_in")

        # ── Single backward pass for all weights ──
        grads = torch.autograd.grad(loss, weights, retain_graph=False)
        surprise = loss.item()

        # ── Apply updates ──
        # MLP fast-weights (index 0..N-1): use FastWeight momentum + surprise-gate
        n_mlp = len(self.fast_weights)
        for i in range(n_mlp):
            fw = self.fast_weights[i]
            grad = grads[i]
            eff_lr = fw.effective_lr(surprise)
            fw.total_surprise += surprise
            fw.velocity = fw.momentum * fw.velocity - eff_lr * grad
            with torch.no_grad():
                fw.layer.weight.add_(fw.velocity)
            fw.update_count += 1
            fw._dirty = True

        # Tokenizer projections (index n_mlp..end): direct momentum update
        # These learn to map raw features into the embedding space.
        # Same lr and momentum as TTT, no surprise-gating (always learn from
        # multimodal signals to build the projection).
        tok_lr = self.config.ttt_lr * 0.5  # slightly lower LR for stability
        tok_momentum = self.config.ttt_momentum
        for i in range(n_mlp, len(weights)):
            grad = grads[i]
            label = weight_labels[i]
            # Get or create momentum buffer
            if not hasattr(self, '_tok_velocity'):
                self._tok_velocity = {}
            if label not in self._tok_velocity:
                self._tok_velocity[label] = torch.zeros_like(weights[i])
            vel = self._tok_velocity[label]
            vel = tok_momentum * vel - tok_lr * grad
            self._tok_velocity[label] = vel
            with torch.no_grad():
                weights[i].add_(vel)

        self.model.zero_grad()
        self.model.eval()

        # PER buffer (text tarafından)
        replay_text = text or ""
        self.replay_buffer.push(replay_text, surprise)
        self.step_counter += 1

        # Multi-frequency CMS konsolidasyonu
        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0:
                cms.consolidate()

        # PER replay
        if self.step_counter % 50 == 0:
            for sample_text in self.replay_buffer.sample(self.config.per_sample_size):
                if sample_text.strip():
                    self.learn(sample_text)

        # Otomatik kaydet
        if self.config.auto_save_every > 0 and self.step_counter % self.config.auto_save_every == 0:
            self.save(silent=True)

        return {"loss": loss.item(), "surprise": surprise, "step": self.step_counter}

    def learn_image(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Learn from an image using a default descriptive prompt.

        Args:
            image_tensor: [C, H, W] or [B, C, H, W] float tensor in model dtype.
        Returns:
            dict with loss, surprise, step.
        """
        return self._multimodal_learn_forward(
            text="Describe this image in detail:",
            image_tensor=image_tensor,
            waveform=None,
        )

    def learn_audio(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Learn from audio using a default descriptive prompt.

        Args:
            waveform: [samples] or [1, samples] float tensor.
        Returns:
            dict with loss, surprise, step.
        """
        return self._multimodal_learn_forward(
            text="Describe this audio in detail:",
            image_tensor=None,
            waveform=waveform,
        )

    def learn_multimodal(self, inputs: dict) -> Dict[str, Any]:
        """Learn from any combination of text, image, and audio.

        Args:
            inputs: {"text": str, "image": tensor, "audio": tensor}
                   All keys optional; at least one modality required.
        Returns:
            dict with loss, surprise, step.
        """
        return self._multimodal_learn_forward(
            text=inputs.get("text"),
            image_tensor=inputs.get("image"),
            waveform=inputs.get("audio"),
        )

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

    def chat(self, user_input, max_new: int = 128) -> str:
        """Chat: learn first, then respond.

        Accepts:
          - str: text-only input
          - dict: multimodal input, e.g. {"text": ..., "image": ..., "audio": ...}
                  At least "text" recommended for best results.
        """
        is_multimodal = isinstance(user_input, dict)
        text_part = user_input.get("text", "") if is_multimodal else user_input
        self.conversation_history.append(f"User: {text_part}")
        context = "\n".join(self.conversation_history[-6:])
        prompt = f"{context}\nAssistant:"

        if is_multimodal:
            info = self.learn_multimodal(user_input)
        else:
            info = self.learn(user_input)
        response = self.generate(prompt, max_new)
        assistant_part = response.split("Assistant:")[-1].strip()
        self.conversation_history.append(f"Assistant: {assistant_part}")
        return assistant_part

    # ============================================================
    # PERSISTENT MEMORY — save / load
    # ============================================================

    def save(self, path: str = None, silent: bool = False, incremental: bool = True):
        """
        Save learned knowledge to disk.

        With incremental=True (default): only saves CMS adapters and deltas
        that have changed since last save. Skips clean adapters to avoid
        redundant I/O (especially important for large models).

        Args:
            path: Save directory (default: config.save_dir)
            silent: Suppress print output
            incremental: Only save dirty/changed data
        """
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save CMS adapters (incremental: only dirty ones)
        cms_data = {}
        saved_cms = 0
        for i, cms in enumerate(self.cms_adapters):
            if incremental and not cms.is_dirty:
                continue
            sd = cms.state_dict()
            torch.save(sd, save_dir / f"cms_{i}.pt")
            cms_data[f"layer_{i}"] = {
                "rank": cms.rank,
                "alpha": cms.alpha,
                "frequency": cms.frequency,
                "shape_A": list(sd["A"].shape),
                "shape_B": list(sd["B"].shape),
                "weight_change": cms.fast.weight_change,
            }
            cms.mark_clean()
            saved_cms += 1

        # Save fast-weight deltas (incremental: only dirty ones)
        fw_data = {}
        saved_fw = 0
        for i, fw in enumerate(self.fast_weights):
            if incremental and not fw.is_dirty:
                # Preserve previous delta info from metadata
                prev_fw = self.memory.get("fast_weight_layers", {}).get(f"layer_{i}", {})
                if prev_fw:
                    fw_data[f"layer_{i}"] = prev_fw
                continue
            delta_path = save_dir / f"delta_{i}.pt"
            torch.save(fw.delta.cpu(), delta_path)
            fw_data[f"layer_{i}"] = {
                "updates": fw.update_count,
                "surprise": fw.total_surprise,
                "change_norm": fw.weight_change,
            }
            fw.mark_clean()
            saved_fw += 1

        # Metadata
        metadata = {
            "model_type": type(self.model).__name__,
            "steps": self.step_counter,
            "saved_at": datetime.datetime.now().isoformat(),
            "config": asdict(self.config),
            "cms_layers": cms_data,
            "fast_weight_layers": fw_data,
            "history_len": len(self.conversation_history),
            "memory_version": 2,  # v2: incremental save + multi-frequency + surprise-gate
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.memory.update(metadata)
        if not silent:
            verb = "artımsal" if incremental else "tam"
            print(f"💾 Kaydedildi ({verb}): {save_dir}/ ({saved_cms} CMS, {saved_fw} delta, "
                  f"{self.step_counter} adım, {len(self.conversation_history)} mesaj)")

    def save_pretrained(self, path: str = "prokopton_model"):
        """Merge all learned knowledge into base model and save with transformers.

        Consolidates all CMS adapters, applies deltas to base weights, and
        calls model.save_pretrained() + tokenizer.save_pretrained().

        The resulting model can be loaded with vanilla transformers —
        no Prokopton required. Knowledge is baked into the weights.

        Note: This rewrites the full model (~9-10 GB for Gemma 4), use
        save() for lightweight incremental checkpointing instead.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Consolidate and apply all CMS to base model
        for cms in self.cms_adapters:
            cms.consolidate()
            cms.apply_to_model()
            cms.mark_clean()

        for fw in self.fast_weights:
            fw.mark_clean()

        # Save trained tokenizer weights alongside the model
        tok_path = save_path / "prokopton_tokenizers.pt"
        tok_state = {
            "audio_input_proj": self.audio_tokenizer.input_proj.state_dict(),
            "audio_output_proj": self.audio_tokenizer.output_proj.state_dict(),
            "vision_input_proj": self.vision_tokenizer.proj.state_dict(),
            "vision_output_proj": self.vision_tokenizer.output_proj.state_dict(),
        }
        # Include momentum buffers if any training happened
        if hasattr(self, '_tok_velocity') and self._tok_velocity:
            tok_state["_tok_velocity"] = {k: v.cpu() for k, v in self._tok_velocity.items()}
        torch.save(tok_state, tok_path)

        # Save with transformers
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        print(f"💾 Değişmiş model kaydedildi: {save_path}/ "
              f"(transformers ile yüklenebilir: AutoModelForCausalLM.from_pretrained('{save_path}'))")
        print(f"   Tokenizer ağırlıkları: {tok_path}")

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
        for cms in self.cms_adapters:
            cms.mark_clean()
        # Clear tokenizer momentum buffers (trained projections reset)
        if hasattr(self, '_tok_velocity'):
            self._tok_velocity = {}
        self.replay_buffer = SurpriseBuffer(self.config.per_capacity)
        self.step_counter = 0
        self.conversation_history = []

    @property
    def stats(self) -> Dict[str, Any]:
        base = {
            "steps": self.step_counter,
            "updates": sum(fw.update_count for fw in self.fast_weights),
            "weight_change": sum(fw.weight_change for fw in self.fast_weights),
            "total_surprise": sum(fw.total_surprise for fw in self.fast_weights),
            "buffer_size": len(self.replay_buffer),
            "history_len": len(self.conversation_history),
            "saved": "metadata.json" in str(self.memory),
            "memory_version": self.memory.get("memory_version", 1),
        }
        # Per-layer stats for monitoring
        for i, fw in enumerate(self.fast_weights):
            base[f"layer_{i}_dw"] = f"{fw.weight_change:.4f}"
            base[f"layer_{i}_updates"] = fw.update_count
            base[f"layer_{i}_eff_lr"] = f"{fw.effective_lr(fw.total_surprise / max(1, fw.update_count)):.6f}"
        for i, cms in enumerate(self.cms_adapters):
            base[f"cms_{i}_freq"] = cms.frequency
            base[f"cms_{i}_dirty"] = cms.is_dirty
        # Tokenizer training stats
        if hasattr(self, '_tok_velocity') and self._tok_velocity:
            for label, vel in self._tok_velocity.items():
                base[f"tok_{label}_dW"] = f"{vel.norm().item():.6f}"
        return base
