"""
Prokopton — Self-Improving, Never-Forgetting LLM Framework

In-Place TTT + CMS + PER + Multimodal + PERSISTENT MEMORY.

Design invariants enforced by this module (see IMPROVEMENT_PLAN.md):

* ``W0`` is an immutable reference weight. ``layer.weight`` is always
  ``W0 + delta``. ``reset()`` restores ``W0``. Only the explicit one-way
  ``commit()`` mutates ``W0``.
* The learned delta is parameterised as a low-rank pair ``W = W0 + B @ A``
  (configurable back to raw updates), so optimizer state and checkpoints stay
  small and ``W0`` is preserved by construction.
* A hard trust region ``||delta|| / ||W0|| <= tau`` bounds drift; ``decay``
  pulls the delta toward zero. Together these make unbounded drift impossible.
* The surprise signal is a **gate** (z-score over a running mean/std), never an
  unbounded learning-rate multiplier.
* ``restore()`` is idempotent; ``commit()`` is explicit and one-way. Persistence
  therefore never compounds the learned delta.
"""
import datetime
import hashlib
import json
import math
import os
import re
import threading
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MEMORY_SCHEMA_VERSION = 3

# Module-name fragments that identify a non-language tower on a multimodal model.
# Selecting these would silently train the vision/audio encoder instead of the LM.
_NON_LANGUAGE_TOWER_HINTS = (
    "vision", "visual", "audio", "speech", "acoustic", "image_emb",
    "vision_tower", "audio_tower", "vision_model", "audio_model",
    "patch_embed", "merger", "rotary_pos_emb", "pos_embed",
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ProkoptonConfig:
    """Framework configuration."""
    # ── TTT optimiser ──
    # `ttt_lr` is calibrated for `parametrization="raw"` on Qwen3-VL-4B / MPS:
    # 40 repeats of one fact take the loss from 1.84 to 0.024 (98.7%) while
    # ‖ΔW‖/‖W0‖ stays at 0.0026 — well inside the 0.01 trust region. The
    # low-rank parameterisation needs roughly 10-100x this for the same effect
    # (see docs/memory-budget.md).
    ttt_lr: float = 1e-2
    ttt_momentum: float = 0.9
    ttt_n_layers: int = 5
    ttt_optimizer: str = "momentum"          # "momentum" | "adam" | "sgd"
    ttt_adam_beta1: float = 0.9
    ttt_adam_beta2: float = 0.999
    ttt_adam_eps: float = 1e-8
    ttt_grad_clip: float = 1.0               # global grad-norm clip (<=0 disables)

    # ── Surprise gate (z-score, NOT an LR multiplier) ──
    ttt_surprise_threshold: float = 0.0      # skip update when z < threshold
    ttt_surprise_warmup: int = 5             # ungated steps while stats form

    # ── Fast-weight parameterisation & stability ──
    # "raw" updates `down_proj` directly, exactly as the In-Place TTT paper
    # does, and is measurably the fastest learner. "lowrank" gives ~50x smaller
    # optimizer state and exact tiny checkpoints at the cost of a much higher LR.
    ttt_parametrization: str = "raw"         # "raw" | "lowrank"
    ttt_rank: int = 16                       # low-rank fast-weight rank
    ttt_trust_region: float = 0.01           # ||ΔW|| / ||W0|| <= tau
    ttt_decay: float = 0.0                   # per-step decay of Δ toward W0 (λ)

    # ── KL anchor regularisation ("never forgetting") ──
    kl_weight: float = 0.05
    kl_every: int = 1
    anchor_prompts: List[str] = field(default_factory=lambda: list(DEFAULT_ANCHOR_PROMPTS))

    # ── CMS consolidation ──
    # Checkpoints are ALWAYS low-rank CMS factors — never a full-rank delta.
    # `cms_rank` bounds the truncation error of save→load; 64 covers the
    # practical rank of a raw delta after ~60 updates.
    #
    # Consolidation is a full SVD of a `hidden x intermediate` matrix, so it is
    # scheduled rather than run on every step: layer i snapshots every
    # `cms_interval_base * 2**i` steps. `save()` always consolidates exactly
    # before writing, so the checkpoint is never staler than the last save.
    cms_frequencies: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    cms_interval_base: int = 32
    cms_rank: int = 64
    cms_alpha: Optional[float] = None        # None → equal to cms_rank (scale 1.0)

    # ── PER replay ──
    per_capacity: int = 128
    per_sample_size: int = 8
    per_replay_every: int = 50

    # ── Persistence ──
    save_dir: str = "prokopton_memory"
    auto_save_every: int = 100
    async_save: bool = True

    # ── Learning targets ──
    learn_from_assistant: bool = True        # chat() learns from its own reply
    max_target_length: int = 256

    # ── Multimodal (encoder-free path; deprecated but retained) ──
    vision_patch_size: int = 48
    vision_embed_dim: int = 1024
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
    audio_patch_frames: int = 4
    max_audio_tokens: int = 12
    max_visual_tokens: int = 100

    @property
    def cms_scale(self) -> float:
        """Multiplier applied when expanding a CMS adapter.

        Historically ``alpha / rank`` silently evaluated to 2.0 (32/16) and
        doubled every restored delta. It now defaults to exactly 1.0.
        """
        alpha = self.cms_rank if self.cms_alpha is None else self.cms_alpha
        return float(alpha) / float(self.cms_rank)


DEFAULT_ANCHOR_PROMPTS: Tuple[str, ...] = (
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The largest planet in the solar system is Jupiter.",
    "Shakespeare wrote Romeo and Juliet.",
    "Two plus two equals four.",
    "The chemical symbol for gold is Au.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Mount Everest is the tallest mountain above sea level.",
    "The speed of light is about 300,000 kilometres per second.",
    "Python is a programming language.",
    "The human heart has four chambers.",
    "Photosynthesis converts sunlight into chemical energy.",
)


# ============================================================
# Fast weights — immutable W0 + bounded delta
# ============================================================

class FastWeight:
    """Single layer's test-time updatable weight.

    The base weight ``W0`` is captured once and never mutated except by the
    explicit one-way :meth:`commit`. The learned delta is either a low-rank
    pair ``(B, A)`` (default) or a raw matrix, and is always kept inside a hard
    trust region ``||Δ|| / ||W0|| <= trust_region``.
    """

    def __init__(self, layer: nn.Linear, lr: float = 1e-3, momentum: float = 0.9,
                 surprise_threshold: float = 0.0, surprise_cap: float = 5.0,
                 rank: int = 16, parametrization: str = "lowrank",
                 trust_region: float = 0.01, decay: float = 0.0,
                 grad_clip: float = 1.0, optimizer: str = "momentum",
                 surprise_warmup: int = 5,
                 adam_beta1: float = 0.9, adam_beta2: float = 0.999,
                 adam_eps: float = 1e-8):
        self.layer = layer
        self.lr = float(lr)
        self.momentum = float(momentum)
        # NOTE: kept for API compatibility; the surprise cap no longer scales the LR.
        self.surprise_cap = float(surprise_cap)
        self.surprise_threshold = float(surprise_threshold)
        self.surprise_warmup = int(surprise_warmup)

        if parametrization not in ("lowrank", "raw"):
            raise ValueError(f"parametrization must be 'lowrank' or 'raw', got {parametrization!r}")
        if optimizer not in ("momentum", "adam", "sgd"):
            raise ValueError(f"optimizer must be 'momentum', 'adam' or 'sgd', got {optimizer!r}")
        self.parametrization = parametrization
        self.optimizer = optimizer
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        self.trust_region = float(trust_region) if trust_region and trust_region > 0 else 0.0
        self.decay = float(decay) if decay and decay > 0 else 0.0
        self.grad_clip = float(grad_clip) if grad_clip and grad_clip > 0 else 0.0

        out_dim, in_dim = layer.weight.shape
        self.out_dim, self.in_dim = out_dim, in_dim

        # ── Immutable reference (single full copy, held in the layer dtype) ──
        self.W0 = layer.weight.detach().clone()

        # ── State dtype ──
        # The delta and the low-rank factors live in float32 so that save → load
        # is bit-exact: keeping them in the model's bf16 re-rounds the delta on
        # every restore and the error compounds across reloads. Momentum is
        # stored in the layer dtype (bf16 on the target hardware) per the memory
        # budget — it is a smoothed direction, not a quantity that must survive
        # a round trip.
        self.state_dtype = torch.float32
        self.momentum_dtype = layer.weight.dtype if layer.weight.dtype in (
            torch.float32, torch.float64, torch.bfloat16, torch.float16) else torch.float32

        # ── Delta parameterisation ──
        # ``B`` starts at zero so the delta (and therefore the model output) is
        # unchanged at init, while ``A`` carries a small random basis so that
        # ``d/dB = g @ Aᵀ`` is non-zero — the standard LoRA initialisation. If
        # both factors started at zero the gradients would vanish and no update
        # would ever land.
        if parametrization == "lowrank":
            self.rank = max(1, min(int(rank), out_dim, in_dim))
            init_scale = 1.0 / math.sqrt(in_dim)
            self.A = (torch.randn(self.rank, in_dim, device=self.W0.device) * init_scale
                      ).to(self.state_dtype)
            self.B = torch.zeros(out_dim, self.rank, dtype=self.state_dtype,
                                 device=self.W0.device)
            self._m_A = torch.zeros(self.A.shape, dtype=self.momentum_dtype,
                                    device=self.W0.device)
            self._m_B = torch.zeros(self.B.shape, dtype=self.momentum_dtype,
                                    device=self.W0.device)
            self._v_A = torch.zeros(self.A.shape, dtype=torch.float32, device=self.W0.device)
            self._v_B = torch.zeros(self.B.shape, dtype=torch.float32, device=self.W0.device)
            self._raw = None
        else:
            self.rank = 0
            self.A = None
            self.B = None
            self._raw = torch.zeros(out_dim, in_dim, dtype=self.state_dtype,
                                    device=self.W0.device)
            self._m_raw = torch.zeros(out_dim, in_dim, dtype=self.momentum_dtype,
                                      device=self.W0.device)
            self._v_raw = torch.zeros(out_dim, in_dim, dtype=torch.float32,
                                      device=self.W0.device)

        self._opt_t: Dict[str, int] = {}
        self.update_count = 0
        self.skipped_count = 0
        self.total_surprise = 0.0
        self.running_surprise = 1.0
        self._surprise_mean = 0.0
        self._surprise_m2 = 0.0
        self._surprise_n = 0
        self._dirty = False
        self._last_saved_delta_norm = 0.0
        self._last_step_norm = 0.0

    # ── Backwards-compatible aliases ──────────────────────────
    @property
    def original_W(self) -> torch.Tensor:
        """Alias of the immutable reference ``W0``."""
        return self.W0

    @property
    def delta(self) -> torch.Tensor:
        """The learned delta ``W - W0``.

        Always read from the parameterisation (``B @ A`` in low-rank mode).
        Computing ``layer.weight - W0`` instead cancels catastrophically when
        the delta is small relative to ``W0`` — in float32 a delta of 1e-5 on
        weights of 1e-1 loses ~1% of its norm to rounding alone.

        ``layer.weight`` is owned by this object: mutate the delta through
        :meth:`apply_grad`, :meth:`restore` or :meth:`set_factors`.
        """
        return self.expand()

    @property
    def delta_norm(self) -> float:
        return self.delta.norm().item()

    @property
    def weight_change(self) -> float:
        return self.delta_norm

    @property
    def reference_norm(self) -> float:
        return self.W0.norm().item()

    @property
    def drift_ratio(self) -> float:
        """``||ΔW|| / ||W0||`` — the quantity the trust region bounds."""
        base = self.reference_norm
        return self.delta_norm / base if base > 0 else 0.0

    @property
    def last_effective_lr(self) -> float:
        """Learning rate applied by the last update (never momentum- or
        surprise-amplified)."""
        return getattr(self, "_last_effective_lr", self.lr)

    def effective_lr(self, surprise: float = 0.0) -> float:
        """Compatibility shim: the LR is no longer scaled by surprise."""
        return self.lr

    # ── Lifecycle ─────────────────────────────────────────────
    def reset(self):
        """Restore the immutable reference and drop all learned state."""
        with torch.no_grad():
            self.layer.weight.copy_(self.W0)
        self._zero_delta_state()
        self._opt_t: Dict[str, int] = {}
        self.update_count = 0
        self.skipped_count = 0
        self.total_surprise = 0.0
        self.running_surprise = 1.0
        self._surprise_mean = 0.0
        self._surprise_m2 = 0.0
        self._surprise_n = 0
        self._dirty = False
        self._last_saved_delta_norm = 0.0
        self._last_step_norm = 0.0

    def commit(self):
        """One-way: fold the current delta into ``W0`` and clear the delta."""
        with torch.no_grad():
            self.W0.copy_(self.layer.weight)
        self._zero_delta_state()
        self._last_saved_delta_norm = 0.0

    def restore(self, delta: Optional[torch.Tensor] = None):
        """Idempotently set ``layer.weight = W0 + delta``.

        With ``delta=None`` the currently held low-rank/raw factors are used.
        Calling this twice in a row is a no-op — this is the property that makes
        persistence non-compounding.
        """
        if delta is None:
            delta = self.expand()
            with torch.no_grad():
                self.layer.weight.copy_(self.W0 + delta.to(self.layer.weight.dtype))
            return
        delta = delta.to(self.state_dtype)
        with torch.no_grad():
            self.layer.weight.copy_(self.W0 + delta.to(self.layer.weight.dtype))
        self._sync_factors(delta)

    def set_factors(self, A: torch.Tensor, B: torch.Tensor):
        """Install low-rank factors directly (no SVD) and rewrite the weight.

        ``A`` is ``(rank_a, in_dim)``, ``B`` is ``(out_dim, rank_a)``. The pair
        is zero-padded/truncated to this fast weight's own rank.
        """
        if self.parametrization != "lowrank":
            raise ValueError("set_factors is only available in 'lowrank' mode")
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("A and B must be 2-D")
        if A.shape[1] != self.in_dim or B.shape[0] != self.out_dim or A.shape[0] != B.shape[1]:
            raise ValueError(
                f"factor shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)} "
                f"expected (*, {self.in_dim}) and ({self.out_dim}, *)")
        r = min(A.shape[0], self.rank)
        with torch.no_grad():
            self.B.zero_()
            self.B[:, :r].copy_(B[:, :r].to(self.state_dtype))
            if A.abs().max().item() == 0.0 and B.abs().max().item() == 0.0:
                # Keep the projection basis alive so d/dB stays non-zero.
                self.A.copy_(torch.randn_like(self.A) / math.sqrt(self.in_dim))
            else:
                self.A.zero_()
                self.A[:r].copy_(A[:r].to(self.state_dtype))
            delta = self.B @ self.A
            self.layer.weight.copy_(self.W0 + delta.to(self.layer.weight.dtype))
            self._m_A.zero_()
            self._m_B.zero_()
            self._v_A.zero_()
            self._v_B.zero_()

    def get_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return detached copies of the low-rank factors ``(A, B)``."""
        if self.parametrization != "lowrank":
            raise ValueError("get_factors is only available in 'lowrank' mode")
        return self.A.detach().clone(), self.B.detach().clone()


    def _zero_delta_state(self):
        """Drop the learned delta and the optimiser state.

        ``A`` is kept: it is the (fixed) projection basis that makes
        ``d/dB = g @ Aᵀ`` non-zero. Zeroing both factors would zero the
        gradients too and learning would silently stop.
        """
        if self.parametrization == "lowrank":
            self.B.zero_()
            self._m_A.zero_()
            self._m_B.zero_()
            self._v_A.zero_()
            self._v_B.zero_()
        else:
            self._raw.zero_()
            self._m_raw.zero_()
            self._v_raw.zero_()

    def _sync_factors(self, delta: torch.Tensor):
        """Make the parameterisation represent exactly ``delta``."""
        if self.parametrization == "lowrank":
            U, S, Vh = _safe_lowrank_svd(delta.float(), self.rank)
            with torch.no_grad():
                self.B.copy_((U * S).to(self.B.dtype))
                self.A.copy_(Vh.to(self.A.dtype))
                self._m_A.zero_()
                self._m_B.zero_()
                self._v_A.zero_()
                self._v_B.zero_()
        else:
            with torch.no_grad():
                self._raw.copy_(delta.to(self.state_dtype))
                self._m_raw.zero_()
                self._v_raw.zero_()

    def expand(self) -> torch.Tensor:
        """Full-rank delta currently represented by the factors (float32)."""
        if self.parametrization == "lowrank":
            return self.B.to(torch.float32) @ self.A.to(torch.float32)
        return self._raw.to(torch.float32)

    # ── Surprise gating ───────────────────────────────────────
    def surprise_z(self, surprise: float) -> float:
        """z-score of ``surprise`` against the running mean/std.

        The std is floored at a fraction of the mean so that a near-constant
        loss cannot make the z-score explode (and the gate flap).
        """
        if self._surprise_n < 2:
            return 0.0
        mean = self._surprise_mean / self._surprise_n
        var = max(self._surprise_m2 / self._surprise_n, 0.0)
        std = max(math.sqrt(var), 1e-3 * abs(mean), 1e-8)
        return (surprise - mean) / std

    def should_update(self, surprise: float) -> bool:
        """True when the surprise gate allows an update for this sample.

        The gate is a z-score test against a running mean/variance of observed
        loss. Below-threshold surprise means "nothing new here" → the step is
        skipped and counted in :attr:`skipped_count`.
        """
        if self._surprise_n < self.surprise_warmup:
            self._observe_surprise(surprise)
            return True
        z = self.surprise_z(surprise)
        self._observe_surprise(surprise)
        return z >= self.surprise_threshold

    def _observe_surprise(self, surprise: float):
        # Welford running mean / variance of observed loss.
        n = self._surprise_n + 1
        mean_prev = self._surprise_mean / self._surprise_n if self._surprise_n else 0.0
        mean_new = mean_prev + (surprise - mean_prev) / n
        self._surprise_m2 += (surprise - mean_prev) * (surprise - mean_new)
        self._surprise_mean = mean_new * n
        self._surprise_n = n
        self.running_surprise = 0.99 * self.running_surprise + 0.01 * surprise

    # ── Update ────────────────────────────────────────────────
    def apply_grad(self, grad: torch.Tensor, surprise: float) -> Dict[str, Any]:
        """Apply one gradient step to the delta under all stability constraints.

        Returns a small dict describing what happened (used for stats/tests).
        """
        if not self.should_update(surprise):
            self.skipped_count += 1
            self._last_step_norm = 0.0
            self._last_effective_lr = 0.0
            return {"skipped": True, "step_norm": 0.0, "drift": self.drift_ratio}

        grad = grad.to(torch.float32)
        if self.grad_clip:
            grad = _clip_grad_norm(grad, self.grad_clip)

        if self.parametrization == "lowrank":
            step_norm = self._apply_grad_lowrank(grad)
        else:
            step_norm = self._apply_grad_raw(grad)

        if self.decay:
            self._apply_decay()

        self._enforce_trust_region()

        self.update_count += 1
        self.total_surprise += surprise
        self._dirty = True
        self._last_step_norm = step_norm
        self._last_effective_lr = self.lr
        return {"skipped": False, "step_norm": step_norm, "drift": self.drift_ratio}

    def _momentum_update(self, name: str, m: torch.Tensor, v: torch.Tensor,
                         g: torch.Tensor):
        """Return ``(m, v, direction)`` for buffer ``name``.

        The direction is always **bias-corrected**, so the first step is exactly
        ``g`` and the steady-state step is ``g`` — never ``g / (1-β)`` and never
        the ``10×``-smaller warm-up value an uncorrected EMA would give. This is
        what keeps the effective learning rate equal to ``self.lr``.

        Each optimiser buffer has its own step counter so the A and B factors of
        a low-rank pair stay exactly in step with each other.
        """
        t = self._opt_t[name] = self._opt_t.get(name, 0) + 1
        g = g.to(torch.float32)
        mf = m.to(torch.float32)
        vf = v.to(torch.float32)
        if self.optimizer == "sgd":
            mf.mul_(self.momentum).add_(g)
            correction = 1.0 - self.momentum ** t
            direction = mf / correction if correction > 1e-12 else mf
        elif self.optimizer == "adam":
            mf.mul_(self.adam_beta1).add_(g, alpha=1 - self.adam_beta1)
            vf.mul_(self.adam_beta2).addcmul_(g, g, value=1 - self.adam_beta2)
            m_hat = mf / (1 - self.adam_beta1 ** t)
            v_hat = vf / (1 - self.adam_beta2 ** t)
            direction = m_hat / (v_hat.sqrt() + self.adam_eps)
        else:
            # "momentum": normalised EMA of the gradient, bias-corrected.
            mf.mul_(self.momentum).add_(g, alpha=1 - self.momentum)
            correction = 1.0 - self.momentum ** t
            direction = mf / correction
        m.copy_(mf.to(m.dtype))
        v.copy_(vf.to(v.dtype))
        return m, v, direction

    def _apply_grad_lowrank(self, gW: torch.Tensor) -> float:
        # d/dB (g·BA) = g Aᵀ ; d/dA = Bᵀ g
        gB = gW @ self.A.to(torch.float32).t()
        gA = self.B.to(torch.float32).t() @ gW
        self._m_B, self._v_B, dirB = self._momentum_update("B", self._m_B, self._v_B, gB)
        self._m_A, self._v_A, dirA = self._momentum_update("A", self._m_A, self._v_A, gA)
        with torch.no_grad():
            self.B.add_(dirB.to(self.state_dtype), alpha=-self.lr)
            self.A.add_(dirA.to(self.state_dtype), alpha=-self.lr)
            delta = self.B @ self.A
            self.layer.weight.copy_(self.W0 + delta.to(self.layer.weight.dtype))
        return float((dirA.norm() + dirB.norm()).item()) * self.lr

    def _apply_grad_raw(self, gW: torch.Tensor) -> float:
        self._m_raw, self._v_raw, direction = self._momentum_update(
            "raw", self._m_raw, self._v_raw, gW)
        with torch.no_grad():
            self._raw.add_(direction.to(self.state_dtype), alpha=-self.lr)
            self.layer.weight.copy_(self.W0 + self._raw.to(self.layer.weight.dtype))
        return float(direction.norm().item()) * self.lr

    def _apply_decay(self):
        """Scale the delta by ``(1 - λ)``; structurally bounds drift."""
        factor = 1.0 - self.decay
        with torch.no_grad():
            if self.parametrization == "lowrank":
                self.B.mul_(factor)
                self.layer.weight.copy_(self.W0 + (self.B @ self.A).to(self.layer.weight.dtype))
            else:
                self._raw.mul_(factor)
                self.layer.weight.copy_(self.W0 + self._raw.to(self.layer.weight.dtype))

    def _enforce_trust_region(self):
        """Rescale (never truncate) the delta so ``||Δ||/||W0|| ≤ tau``."""
        if not self.trust_region:
            return
        delta = self.delta.to(torch.float32)
        d_norm = delta.norm().item()
        base = self.W0.float().norm().item()
        if base <= 0 or d_norm <= 0:
            return
        allowed = self.trust_region * base
        if d_norm <= allowed:
            return
        scale = allowed / d_norm
        with torch.no_grad():
            if self.parametrization == "lowrank":
                self.B.mul_(scale)
                self.layer.weight.copy_(self.W0 + (self.B @ self.A).to(self.layer.weight.dtype))
            else:
                self._raw.mul_(scale)
                self.layer.weight.copy_(self.W0 + self._raw.to(self.layer.weight.dtype))

    # ── Persistence ───────────────────────────────────────────
    @property
    def is_dirty(self) -> bool:
        if not self._dirty:
            return False
        return abs(self.weight_change - self._last_saved_delta_norm) > max(
            1e-8, self._last_saved_delta_norm * 0.01)

    def mark_clean(self):
        self._last_saved_delta_norm = self.weight_change
        self._dirty = False

    def state_dict(self) -> Dict[str, Any]:
        if self.parametrization == "lowrank":
            factors = {"A": self.A.detach().cpu(), "B": self.B.detach().cpu()}
        else:
            factors = {"delta": self._raw.detach().cpu()}
        return {
            "parametrization": self.parametrization,
            "rank": self.rank,
            "factors": factors,
            "update_count": self.update_count,
            "skipped_count": self.skipped_count,
            "total_surprise": self.total_surprise,
            "running_surprise": self.running_surprise,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        if sd.get("parametrization", self.parametrization) != self.parametrization:
            raise ValueError(
                f"parametrization mismatch: checkpoint={sd.get('parametrization')!r} "
                f"live={self.parametrization!r}")
        factors = sd["factors"]
        if self.parametrization == "lowrank":
            self.set_factors(factors["A"], factors["B"])
        else:
            with torch.no_grad():
                self._raw.copy_(factors["delta"].to(self._raw))
                self.layer.weight.copy_(self.W0 + self._raw.to(self.layer.weight.dtype))
            self._m_raw.zero_()
            self._v_raw.zero_()
        self.update_count = int(sd.get("update_count", 0))
        self.skipped_count = int(sd.get("skipped_count", 0))
        self.total_surprise = float(sd.get("total_surprise", 0.0))
        self.running_surprise = float(sd.get("running_surprise", 1.0))


# ============================================================
# CMS — consolidation + persistence
# ============================================================

class CMSAdapter:
    """Distills the fast-weight delta into rank-``r`` factors via SVD.

    The adapter owns the checkpoint representation of a layer's delta. Its
    semantics are deliberately split:

    * :meth:`consolidate` — snapshot ``fast.delta`` as ``B @ A`` (rank-``r``).
    * :meth:`apply_to_model` — **restore**: ``W = W0 + Δ``. Idempotent.
    * :meth:`commit` — **one-way**: ``W0 ← W`` then clear Δ.
    """

    def __init__(self, fast_weight: 'FastWeight', rank: int = 16, alpha: float = None,
                 frequency: int = 1):
        self.fast = fast_weight
        device = fast_weight.layer.weight.device
        dtype = fast_weight.layer.weight.dtype
        in_dim = fast_weight.layer.weight.shape[1]
        out_dim = fast_weight.layer.weight.shape[0]
        self.rank = max(1, min(int(rank), out_dim, in_dim))
        # Factors are float32 so save → load is bit-exact (see FastWeight).
        state_dtype = getattr(fast_weight, "state_dtype", torch.float32)
        self.A = nn.Parameter(torch.zeros(self.rank, in_dim, device=device, dtype=state_dtype))
        self.B = nn.Parameter(torch.zeros(out_dim, self.rank, device=device, dtype=state_dtype))
        # alpha == rank → scale exactly 1.0 (the historical 32/16 = 2.0 silently
        # doubled every restored delta).
        self.alpha = float(self.rank if alpha is None else alpha)
        self.frequency = max(1, int(frequency))
        self._dirty = False

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def consolidate(self):
        """Snapshot the fast-weight delta as rank-``r`` factors ``A``/``B``.

        Always an SVD truncation of ``fast.delta``, so the result is a genuine
        rank-``r`` approximation and is correct even when ``layer.weight`` was
        changed outside the fast-weight update path. The SVD is deterministic
        and device-safe: ``torch.svd_lowrank`` is attempted first and the delta
        is moved to CPU for a full ``torch.linalg.svd`` whenever that kernel is
        unavailable (it is missing on some accelerator/dtype combinations).
        """
        scale = self.scale or 1.0
        fw = self.fast
        delta = fw.delta.detach()

        # Fast path: the low-rank factors *are* the delta, so copying them is
        # exact and avoids an unnecessary SVD.
        if fw.parametrization == "lowrank" and fw.rank <= self.rank:
            with torch.no_grad():
                self.A.zero_()
                self.B.zero_()
                r = fw.rank
                self.A[:r].copy_(fw.A[:r].to(self.A))
                self.B[:, :r].copy_(fw.B[:, :r].to(self.B) / scale)
            self._dirty = True
            return

        U, S, Vh = _safe_lowrank_svd(delta.float(), self.rank)
        with torch.no_grad():
            self.B.copy_((U * S).to(torch.float32) / scale)
            self.A.copy_(Vh.to(torch.float32))
        self._dirty = True

    def expand(self) -> torch.Tensor:
        """Expand ``A``/``B`` to the full delta matrix (float32)."""
        return (self.B.to(torch.float32) @ self.A.to(torch.float32)) * self.scale

    def apply_to_model(self):
        """Restore: ``W = W0 + Δ``. Idempotent — safe to call repeatedly."""
        fw = self.fast
        scale = self.scale or 1.0
        if fw.parametrization == "lowrank" and self.rank <= fw.rank:
            # Exact factor transfer — no SVD, no compounding.
            fw.set_factors(self.A.data * scale, self.B.data)
            return
        fw.restore(self.expand())


    def commit(self):
        """One-way: fold Δ into the immutable reference and clear it."""
        self.fast.commit()
        with torch.no_grad():
            self.A.zero_()
            self.B.zero_()
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"A": self.A.data.detach().cpu(), "B": self.B.data.detach().cpu()}

    def load_state_dict(self, d: Dict[str, torch.Tensor]):
        if tuple(d["A"].shape) != tuple(self.A.shape) or tuple(d["B"].shape) != tuple(self.B.shape):
            raise ValueError(
                f"CMS shape mismatch: checkpoint A={tuple(d['A'].shape)} B={tuple(d['B'].shape)} "
                f"live A={tuple(self.A.shape)} B={tuple(self.B.shape)}")
        self.A.data.copy_(d["A"].to(self.A.device, self.A.dtype))
        self.B.data.copy_(d["B"].to(self.B.device, self.B.dtype))
        self._dirty = True


# ============================================================
# Numerics helpers
# ============================================================

def _clip_grad_norm(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    total = grad.norm().item()
    if total > max_norm and total > 0:
        return grad * (max_norm / total)
    return grad


def _safe_lowrank_svd(delta: torch.Tensor, rank: int):
    """Rank-``r`` SVD factors ``(U, S, Vh)`` of ``delta``, deterministic & safe.

    ``torch.linalg.svd`` is used as the primary path because it is
    deterministic and agrees exactly across CPU/MPS, which is what the
    cross-device acceptance test requires. ``torch.svd_lowrank`` is *randomized*
    and is only a fallback — it is also missing on some accelerator builds.
    Consolidation is infrequent, so the extra cost of the exact SVD is
    acceptable and buys reproducibility.
    """
    rank = max(1, min(int(rank), min(delta.shape)))
    work = delta.detach().float()

    # MPS cannot stage large matrices in threadgroup memory and silently falls
    # back to CPU anyway; go straight there for anything big enough to matter.
    candidates = [work] if work.numel() <= 4096 else [work.cpu()]
    candidates.append(work.cpu() if candidates[0] is not work else work.cpu())
    for candidate in candidates:
        try:
            U, S, Vh = torch.linalg.svd(candidate, full_matrices=False)
            device = delta.device
            return (U[:, :rank].to(device), S[:rank].to(device), Vh[:rank].to(device))
        except (RuntimeError, NotImplementedError):
            continue

    try:
        U, S, V = torch.svd_lowrank(work, q=rank)
        return U, S, V.t()
    except (RuntimeError, NotImplementedError):
        pass

    # Absolute last resort: eigen-decomposition of the Gram matrix.
    gram = work.cpu() @ work.cpu().t()
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)[:rank]
    S = evals[order].clamp_min(0).sqrt()
    U = evecs[:, order]
    Vh = (U.t() @ work.cpu()) / S.clamp_min(1e-12).unsqueeze(1)
    device = delta.device
    return U.to(device), S.to(device), Vh.to(device)


# ============================================================
# PER replay buffer
# ============================================================

class SurpriseBuffer:
    """Surprise-prioritized replay buffer (PER)."""
    def __init__(self, capacity: int = 128):
        self.items: deque = deque(maxlen=capacity)
        self.surprises: deque = deque(maxlen=capacity)

    def push(self, text: str, surprise: float):
        self.items.append(text)
        self.surprises.append(surprise)

    def sample(self, k: int = 8) -> List[str]:
        if not self.items:
            return []
        k = min(k, len(self.items))
        probs = torch.tensor(list(self.surprises), dtype=torch.float)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if float(probs.sum()) <= 0:
            probs = torch.ones_like(probs)
        probs = probs / probs.sum()
        idxs = torch.multinomial(probs, k).tolist()
        return [self.items[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.items)


# ============================================================
# Multimodal tokenizers (encoder-free path — DEPRECATED)
# ============================================================

class VisualTokenizer(nn.Module):
    """.. deprecated:: 0.4.1
        Encoder-free vision tokenizers are frozen. Prefer the native VLM path
        (:class:`prokopton.vlm.ProkoptonVL`), which uses the model's own vision
        tower and needs no separate projection to be trained.
    """

    def __init__(self, patch_size=48, embed_dim=1024, output_dim=2560, max_grid=64):
        super().__init__()
        warnings.warn(
            "VisualTokenizer is deprecated and frozen; use prokopton.vlm.ProkoptonVL "
            "with a native multimodal model instead.", DeprecationWarning, stacklevel=2)
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
        if c.shape[-1] < D // 2:
            c = nn.functional.pad(c, (0, D // 2 - c.shape[-1]))
            s = nn.functional.pad(s, (0, D // 2 - s.shape[-1]))
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        x = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)
        return self.output_proj(x), {"num_tokens": gh * gw, "grid": (gh, gw)}


class AudioTokenizer(nn.Module):
    """.. deprecated:: 0.4.1
        Encoder-free audio tokenizers are frozen. See :class:`VisualTokenizer`.
    """

    def __init__(self, sample_rate=16000, n_mels=80, patch_frames=4, embed_dim=1024, output_dim=2560):
        super().__init__()
        warnings.warn(
            "AudioTokenizer is deprecated and frozen; use prokopton.vlm.ProkoptonVL "
            "with a native multimodal model instead.", DeprecationWarning, stacklevel=2)
        self.sr, self.n_mels, self.pf = sample_rate, n_mels, patch_frames
        self.n_fft, self.hop = 400, 160
        self.input_proj = nn.Linear(patch_frames * n_mels, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        n_freq_bins = self.n_fft // 2 + 1
        mel_f = torch.linspace(0, 2595 * math.log10(1 + (sample_rate // 2) / 700), n_mels + 2)
        hz_f = 700 * (10 ** (mel_f / 2595) - 1)
        fft_bins = torch.floor((self.n_fft + 1) * hz_f / sample_rate).long()

        mel_weights = torch.zeros(n_freq_bins, n_mels)
        for m in range(n_mels):
            s, e = fft_bins[m].item(), fft_bins[m + 2].item()
            s = max(0, s)
            e = min(e, n_freq_bins)
            if e > s:
                mel_weights[s:e, m] = 1.0 / (e - s)
        self.register_buffer('mel_weights', mel_weights)
        self.register_buffer('hann_window', torch.hann_window(self.n_fft))

        pe = torch.zeros(4096, embed_dim)
        pos = torch.arange(4096).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, waveform):
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

            frames = w.unfold(0, self.n_fft, self.hop)[:n_frames]
            frames = frames * self.hann_window.to(device)

            spec = torch.abs(torch.fft.rfft(frames))[:, :self.n_fft // 2 + 1] ** 2

            mel = spec @ self.mel_weights.to(device)
            mel = torch.log(mel + 1e-10)
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            npatch = n_frames // self.pf
            if npatch == 0:
                all_tokens.append(torch.zeros(1, self.output_proj.out_features, device=device))
                all_info.append({"num_tokens": 1, "duration_ms": 0})
                continue

            patches = mel[:npatch * self.pf].reshape(npatch, self.pf * self.n_mels)

            x = self.input_proj(patches) + self.pe[:npatch].to(device)
            all_tokens.append(self.output_proj(x))

            dur = (n_frames * self.hop / self.sr) * 1000
            all_info.append({"num_tokens": npatch, "duration_ms": dur})

        return all_tokens, all_info


# ============================================================
# Layer selection
# ============================================================

_LAYER_RE = re.compile(r"(?:^|\.)layers\.\d+\.mlp\.down_proj$")


def _is_non_language_tower(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _NON_LANGUAGE_TOWER_HINTS)


def select_ttt_layers(model: nn.Module, n_layers: int) -> List[Tuple[str, nn.Module]]:
    """Select the language model's ``mlp.down_proj`` modules for TTT.

    Selection is **scoped to the language model**: only names matching
    ``...layers.<N>.mlp.down_proj`` are eligible, and anything belonging to a
    vision/audio tower is rejected. There is no "last N Linear" fallback — that
    heuristic can silently attach TTT to a vision encoder.

    Raises ``ValueError`` with the discovered candidate names when nothing
    matches, instead of guessing.
    """
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _LAYER_RE.search(name):
            continue
        if _is_non_language_tower(name):
            continue
        if getattr(module, "weight", None) is None or module.weight.dim() != 2:
            continue
        candidates.append((name, module))

    if not candidates:
        raise ValueError(
            "No language-model `mlp.down_proj` layers found for TTT. "
            "Refusing to fall back to arbitrary Linear modules (that can attach "
            "TTT to a vision/audio tower). Either wrap the language model so its "
            "layers are named `...layers.<N>.mlp.down_proj`, or pass an explicit "
            "layer list. For reference, the model exposes these Linear modules: "
            + ", ".join(n for n, m in model.named_modules() if isinstance(m, nn.Linear))[:500]
        )

    # Prefer names under a `language_model` root when both kinds exist.
    lm_scoped = [c for c in candidates if "language_model" in c[0]]
    pool = lm_scoped or candidates

    if n_layers and n_layers > 0:
        pool = pool[-min(n_layers, len(pool)):]
    return pool


# ============================================================
# Prokopton framework
# ============================================================

class Prokopton:
    """Prokopton: self-improving, never-forgetting LLM framework.

    Usage::

        prok = Prokopton(model, tokenizer)
        prok.learn("Zephyria's capital is Aethel.")
        prok.save()                      # persist to disk
        prok2 = Prokopton(model2, tokenizer2)
        prok2.load()                     # restore previous knowledge (idempotent)
        prok2.chat("What is the capital of Zephyria?")
    """

    def __init__(self, model, tokenizer, config: ProkoptonConfig = None,
                 ttt_layers: Optional[Sequence[Tuple[str, nn.Module]]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ProkoptonConfig()

        self.fast_weights: List[FastWeight] = []
        self.cms_adapters: List[CMSAdapter] = []
        self.replay_buffer = SurpriseBuffer(self.config.per_capacity)
        self.step_counter = 0          # gradient updates (incl. replay)
        self.interaction_counter = 0   # user interactions / explicit learn() calls
        self.skipped_steps = 0
        self.conversation_history: List[str] = []
        self.memory: Dict[str, Any] = {}

        self._setup_ttt(ttt_layers)
        self._setup_multimodal()
        self._setup_anchor_kl()

    # ── Setup ─────────────────────────────────────────────────
    def _setup_ttt(self, ttt_layers=None):
        selected = list(ttt_layers) if ttt_layers is not None else \
            select_ttt_layers(self.model, self.config.ttt_n_layers)

        n_layers = len(selected)
        if len(self.config.cms_frequencies) < n_layers:
            base = max(1, int(self.config.cms_interval_base))
            auto_freqs = [base * 2 ** i for i in range(n_layers)]
        else:
            auto_freqs = list(self.config.cms_frequencies[:n_layers])

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
                adam_beta1=self.config.ttt_adam_beta1,
                adam_beta2=self.config.ttt_adam_beta2,
                adam_eps=self.config.ttt_adam_eps,
            )
            fw.layer_name = name
            self.fast_weights.append(fw)
            cms = CMSAdapter(fw, self.config.cms_rank, self.config.cms_alpha,
                             frequency=auto_freqs[i])
            self.cms_adapters.append(cms)

        self.ttt_layer_names = [name for name, _ in selected]

    def _setup_multimodal(self):
        """Initialize multimodal tokenizers + detect model capabilities."""
        try:
            output_dim = self.model.config.hidden_size
        except (AttributeError, KeyError):
            try:
                output_dim = self.model.config.text_config.hidden_size
            except (AttributeError, KeyError):
                output_dim = 2560

        try:
            model_dtype = self.model.dtype
        except AttributeError:
            model_dtype = next(self.model.parameters()).dtype
        try:
            model_device = self.model.device
        except AttributeError:
            model_device = next(self.model.parameters()).device

        self._model_info = self._detect_model_type()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.vision_tokenizer = VisualTokenizer(
                self.config.vision_patch_size, self.config.vision_embed_dim, output_dim)
            self.audio_tokenizer = AudioTokenizer(
                self.config.audio_sample_rate, self.config.audio_n_mels,
                self.config.audio_patch_frames, self.config.vision_embed_dim, output_dim)

        self.vision_tokenizer = self.vision_tokenizer.to(device=model_device)
        self.audio_tokenizer = self.audio_tokenizer.to(device=model_device)

    def _detect_model_type(self) -> Dict[str, Any]:
        config = getattr(self.model, "config", None)
        info = {
            "is_gemma4": False,
            "is_multimodal": False,
            "has_text_config": False,
            "model_name": getattr(config, 'model_type', 'unknown'),
            "oom_risk_multimodal": False,
        }
        if config is None:
            return info

        model_type = getattr(config, 'model_type', '') or ''
        if 'gemma4' in model_type or 'gemma_4' in model_type:
            info["is_gemma4"] = True
        if hasattr(config, 'text_config'):
            info["has_text_config"] = True
        has_vision = hasattr(config, 'vision_config') or hasattr(config, 'visual_config')
        has_audio = hasattr(config, 'audio_config') or hasattr(config, 'speech_config')
        if has_vision or has_audio:
            info["is_multimodal"] = True
        if info["is_gemma4"] and not info["is_multimodal"]:
            info["oom_risk_multimodal"] = True
        return info

    # ── Model forward helper ──────────────────────────────────
    @staticmethod
    def _model_forward(model, **kwargs):
        """Call ``model(**kwargs)``, silently dropping kwargs it does not accept.

        Prokopton wraps arbitrary user models; not every model takes
        ``attention_mask``/``position_ids``.
        """
        try:
            return model(**kwargs)
        except TypeError:
            import inspect
            try:
                sig = inspect.signature(model.forward)
                accepted = set(sig.parameters)
                if any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values()):
                    raise
                dropped = {k: v for k, v in kwargs.items() if k not in accepted}
                kept = {k: v for k, v in kwargs.items() if k in accepted}
                if not dropped:
                    raise
                return model(**kept)
            except Exception:
                raise

    # ── KL anchor ("never forgetting") ────────────────────────
    def _setup_anchor_kl(self):
        self._anchor_input_ids = None
        self._anchor_attention = None
        self._anchor_teacher_logprobs = None
        if not self.config.kl_weight or not self.config.anchor_prompts:
            return
        self._refresh_anchor_teacher()

    def _refresh_anchor_teacher(self):
        """Cache teacher log-probs of the frozen reference model on anchors."""
        self._anchor_teacher_logprobs = None
        if not self.config.kl_weight or not self.config.anchor_prompts:
            return
        try:
            try:
                device = self.model.device
            except AttributeError:
                device = next(self.model.parameters()).device

            prompts = list(self.config.anchor_prompts)
            try:
                enc = self.tokenizer(prompts, return_tensors="pt", truncation=True,
                                     max_length=64, padding=True)
            except TypeError:
                ids = [self.tokenizer(p, return_tensors="pt", truncation=True,
                                      max_length=64)["input_ids"] for p in prompts]
                enc = {"input_ids": torch.cat(ids, dim=0)}
            self._anchor_input_ids = enc["input_ids"].to(device)
            self._anchor_attention = enc.get(
                "attention_mask", torch.ones_like(self._anchor_input_ids)).to(device)

            backups = [fw.layer.weight.detach().clone() for fw in self.fast_weights]
            try:
                for fw in self.fast_weights:
                    with torch.no_grad():
                        fw.layer.weight.copy_(fw.W0)
                with torch.no_grad():
                    out = self._model_forward(
                        self.model, input_ids=self._anchor_input_ids,
                        attention_mask=self._anchor_attention)
                    self._anchor_teacher_logprobs = F.log_softmax(out.logits.float(), dim=-1)
            finally:
                for fw, w in zip(self.fast_weights, backups):
                    with torch.no_grad():
                        fw.layer.weight.copy_(w)
        except Exception as exc:  # pragma: no cover - model-specific forward issues
            warnings.warn(f"KL anchor teacher unavailable: {exc}", RuntimeWarning)
            self._anchor_input_ids = None
            self._anchor_attention = None
            self._anchor_teacher_logprobs = None

    def _kl_anchor_loss(self) -> torch.Tensor:
        if self._anchor_teacher_logprobs is None or self._anchor_input_ids is None:
            return torch.zeros(())
        out = self._model_forward(self.model, input_ids=self._anchor_input_ids,
                                  attention_mask=self._anchor_attention)
        student = F.log_softmax(out.logits.float(), dim=-1)
        teacher = self._anchor_teacher_logprobs
        kl = F.kl_div(student[:, :-1], teacher[:, :-1].exp(),
                      reduction="batchmean", log_target=False)
        return kl

    # ── Learning ──────────────────────────────────────────────
    def learn(self, text: str, target: Optional[str] = None) -> Dict[str, Any]:
        """Learn from text.

        Two modes:

        * ``learn(fact)`` — the string is an explicit fact to be memorised. The
          whole string is the learning target (the documented API).
        * ``learn(prompt, target=reply)`` — only ``target`` tokens carry loss;
          the prompt is masked out. Use this for anything that may contain
          questions, typos, false premises or boilerplate.
        """
        self.interaction_counter += 1
        return self._learn_step(text, target=target, allow_replay=True)

    def _learn_step(self, text: str, target: Optional[str] = None,
                    allow_replay: bool = True) -> Dict[str, Any]:
        seq, labels = self._build_supervised_sequence(text, target)
        loss, surprise = self._forward_and_loss(seq, labels)

        grads = torch.autograd.grad(loss, [fw.layer.weight for fw in self.fast_weights],
                                    retain_graph=False, allow_unused=True)

        skipped = 0
        step_norms = []
        for fw, grad in zip(self.fast_weights, grads):
            if grad is None:
                grad = torch.zeros_like(fw.layer.weight)
            info = fw.apply_grad(grad, surprise)
            if info["skipped"]:
                skipped += 1
            else:
                step_norms.append(info["step_norm"])

        self.model.zero_grad()
        self.model.eval()

        if skipped == len(self.fast_weights):
            self.skipped_steps += 1

        self.replay_buffer.push(text if target is None else f"{text}\n{target}", surprise)
        self.step_counter += 1

        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0 and cms.fast.is_dirty:
                cms.consolidate()

        # Schedules run off the interaction counter so that replay re-hearsals
        # (which are not new experience) cannot cascade into further replay.
        if allow_replay and self.config.per_replay_every > 0 and \
                self.interaction_counter % self.config.per_replay_every == 0:
            self._run_replay()

        if self.config.auto_save_every > 0 and \
                self.interaction_counter % self.config.auto_save_every == 0:
            self.save(silent=True)

        return {
            "loss": float(loss.item()),
            "surprise": float(surprise),
            "step": self.step_counter,
            "skipped": skipped,
            "drift": max((fw.drift_ratio for fw in self.fast_weights), default=0.0),
        }

    def _run_replay(self):
        """Bounded, non-recursive PER replay (Phase 5.2)."""
        samples = self.replay_buffer.sample(self.config.per_sample_size)
        for sample_text in samples:
            if sample_text and sample_text.strip():
                self._learn_step(sample_text, target=None, allow_replay=False)

    def _build_supervised_sequence(self, text: str, target: Optional[str]):
        """Return ``(input_ids, labels)`` with the prompt masked out of the loss."""
        device = self.model.device if hasattr(self.model, "device") else \
            next(self.model.parameters()).device
        max_len = self.config.max_target_length

        if target is None:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            ids = enc["input_ids"].to(device)
            labels = ids.clone()
            return {"input_ids": ids,
                    "attention_mask": enc.get("attention_mask", torch.ones_like(ids)).to(device)}, labels

        prompt_ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=max_len)["input_ids"].to(device)
        full_ids = self.tokenizer(text + target, return_tensors="pt", truncation=True,
                                  max_length=max_len)["input_ids"].to(device)
        prompt_len = int(prompt_ids.shape[1])
        labels = torch.full_like(full_ids, -100)
        labels[0, prompt_len:] = full_ids[0, prompt_len:]
        return {"input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids, device=device)}, labels

    def _forward_and_loss(self, seq: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.model.train()
        outputs = self._model_forward(self.model, **seq)
        logits = outputs.logits[:, :-1].contiguous()
        target = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100)

        if self.config.kl_weight > 0 and self.config.kl_every > 0 and \
                self.step_counter % self.config.kl_every == 0 and \
                self._anchor_teacher_logprobs is not None:
            kl = self._kl_anchor_loss()
            loss = loss + self.config.kl_weight * kl.to(loss.device)

        return loss, float(loss.item())

    # ── Multimodal learning ───────────────────────────────────
    def _prepare_multimodal_embeds(
        self, text: str = None, image_tensor: torch.Tensor = None,
        waveform: torch.Tensor = None
    ):
        """Build combined embeddings, attention mask, and labels.

        Layout: [AUDIO...] [VISUAL...] [TEXT...]
        Labels are built by placing text ids at their absolute positions in a
        full-length label row and then shifting by one, which makes
        ``labels[j] == combined_ids[j+1]`` true by construction and leaves every
        multimodal position at ``-100``.
        """
        device = self.model.device
        dtype = self.model.dtype
        embeds_list, mask_list = [], []
        text_ids_for_labels = None
        n_multimodal_prefix = 0

        max_aud = self.config.max_audio_tokens
        if self._model_info["oom_risk_multimodal"]:
            max_aud = min(max_aud, 15)

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
                warnings.warn(
                    f"Audio tokens trimmed by {n_trimmed} (→ {max_aud} max).",
                    RuntimeWarning)

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

        if text is not None:
            tokens = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=self.config.max_target_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            text_ids = tokens["input_ids"]
            text_mask = tokens["attention_mask"]
            embed_layer = self.model.get_input_embeddings()
            text_embeds = embed_layer(text_ids).to(dtype)
            embeds_list.append(text_embeds)
            mask_list.append(text_mask)
            text_ids_for_labels = text_ids

        if not embeds_list:
            raise ValueError("At least one modality (text, image, audio) is required")

        combined_embeds = torch.cat(embeds_list, dim=1)
        combined_mask = torch.cat(mask_list, dim=1)

        total_len = combined_embeds.shape[1]
        device = combined_embeds.device
        full_labels = torch.full((1, total_len), -100, dtype=torch.long, device=device)
        if text_ids_for_labels is not None:
            text_len = text_ids_for_labels.shape[1]
            start = n_multimodal_prefix
            end = min(start + text_len, total_len)
            if end > start:
                full_labels[0, start:end] = text_ids_for_labels[0, :end - start]
        labels = full_labels[:, 1:].contiguous()

        return combined_embeds, combined_mask, labels

    def _multimodal_learn_forward(
        self, text: str, image_tensor: torch.Tensor, waveform: torch.Tensor
    ) -> Dict[str, Any]:
        embeds, attn_mask, labels = self._prepare_multimodal_embeds(
            text, image_tensor, waveform)

        self.model.train()
        outputs = self._model_forward(self.model, inputs_embeds=embeds, attention_mask=attn_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
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

        self.replay_buffer.push(text or "", surprise)
        self.step_counter += 1

        for cms in self.cms_adapters:
            if self.step_counter % cms.frequency == 0 and cms.fast.is_dirty:
                cms.consolidate()

        return {"loss": surprise, "surprise": surprise, "step": self.step_counter,
                "skipped": skipped}

    def learn_image(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        return self._multimodal_learn_forward(
            text="Describe this image in detail:",
            image_tensor=image_tensor,
            waveform=None,
        )

    def learn_audio(self, waveform: torch.Tensor) -> Dict[str, Any]:
        return self._multimodal_learn_forward(
            text="Describe this audio in detail:",
            image_tensor=None,
            waveform=waveform,
        )

    def learn_multimodal(self, inputs: dict) -> Dict[str, Any]:
        return self._multimodal_learn_forward(
            text=inputs.get("text"),
            image_tensor=inputs.get("image"),
            waveform=inputs.get("audio"),
        )

    # ── Generation ────────────────────────────────────────────
    def generate_completion(self, prompt: str, max_new: int = 128) -> str:
        """Raw generation: returns the prompt *plus* the completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                temperature=1.0, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate(self, prompt: str, max_new: int = 128) -> str:
        """Generate only the new tokens (the prompt is never echoed)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                temperature=1.0, pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _chat_prompt(self, history: List[str], user_text: str) -> str:
        """Build a prompt, preferring the tokenizer's chat template (Phase 2.8)."""
        messages = []
        for turn in history:
            if turn.startswith("User: "):
                messages.append({"role": "user", "content": turn[len("User: "):]})
            elif turn.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": turn[len("Assistant: "):]})
        messages.append({"role": "user", "content": user_text})

        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        context = "\n".join(history[-6:])
        return f"{context}\nUser: {user_text}\nAssistant:" if context else \
            f"User: {user_text}\nAssistant:"

    def chat(self, user_input, max_new: int = 128) -> str:
        """Chat: respond first, then learn from the assistant reply.

        Accepts ``str`` or a multimodal dict ``{"text", "image", "audio"}``.

        The raw user utterance is **never** a learning target — it may contain
        questions, typos or false premises (Phase 2.9). Learning targets are the
        assistant response (when ``config.learn_from_assistant``), facts passed
        explicitly to :meth:`learn`, or content confirmed via
        :meth:`confirm_learning`.
        """
        is_multimodal = isinstance(user_input, dict)
        text_part = user_input.get("text", "") if is_multimodal else user_input

        prompt = self._chat_prompt(self.conversation_history, text_part)
        response = self.generate(prompt, max_new)
        for prefix in ("Assistant:", "assistant:", "assistant\n"):
            if response.startswith(prefix):
                response = response[len(prefix):]
        response = response.strip()

        info: Dict[str, Any] = {}
        self.interaction_counter += 1
        if self.config.learn_from_assistant:
            if is_multimodal:
                mm = dict(user_input)
                mm["text"] = response
                info = self.learn_multimodal(mm)
            else:
                info = self._learn_step(prompt, target=response, allow_replay=True)

        self.conversation_history.append(f"User: {text_part}")
        self.conversation_history.append(f"Assistant: {response}")
        self._last_chat_info = info
        return response

    def confirm_learning(self, content: str) -> Dict[str, Any]:
        """Learn explicitly confirmed content (Phase 2.9 option c)."""
        return self.learn(content)

    # ── Persistence ───────────────────────────────────────────
    def model_fingerprint(self) -> str:
        """Stable architecture + shape fingerprint of the TTT-tracked layers."""
        parts = []
        for name, fw in zip(self.ttt_layer_names, self.fast_weights):
            parts.append(f"{name}:{tuple(fw.W0.shape)}")
        raw = "|".join(parts).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _environment_metadata(self) -> Dict[str, Any]:
        meta = {}
        try:
            meta["torch"] = torch.__version__
        except Exception:
            pass
        try:
            import transformers
            meta["transformers"] = transformers.__version__
        except Exception:
            pass
        try:
            meta["device"] = str(next(self.model.parameters()).device)
        except Exception:
            pass
        try:
            meta["python"] = __import__("sys").version.split()[0]
        except Exception:
            pass
        return meta

    def save(self, path: str = None, silent: bool = False, incremental: bool = True):
        """Save learned knowledge to disk as low-rank CMS adapters.

        Only rank-``r`` factors are written — never a full-rank delta. Writes
        are atomic (tmp file + rename) and, when ``config.async_save``, happen on
        a background thread.

        ``incremental`` is retained for API compatibility: with genuinely
        low-rank checkpoints the whole state is a few hundred KB, so every layer
        is always refreshed. The flag only skips layers whose delta is provably
        unchanged **and** whose file already exists.
        """
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = self._build_save_payload(incremental=incremental, save_dir=save_dir)

        if self.config.async_save and payload["tensors"]:
            thread = threading.Thread(target=_save_tensors,
                                      args=(save_dir, payload["tensors"]), daemon=True)
            thread.start()
            thread.join(timeout=30.0)
            if thread.is_alive():
                threading.Thread(target=_save_tensors,
                                 args=(save_dir, payload["tensors"]), daemon=True).start()
        else:
            _save_tensors(save_dir, payload["tensors"])

        _atomic_write_json(save_dir / "metadata.json", payload["metadata"])

        self.memory.update(payload["metadata"])
        for cms in self.cms_adapters:
            cms.mark_clean()
        for fw in self.fast_weights:
            fw.mark_clean()

        if not silent:
            print(f"💾 Saved ({'incremental' if incremental else 'full'}): {save_dir}/ "
                  f"({len(payload['tensors'])} adapters, {self.step_counter} steps, "
                  f"{len(self.conversation_history)} messages)")

    def _build_save_payload(self, incremental: bool = True,
                            save_dir: Optional[Path] = None) -> Dict[str, Any]:
        tensors: Dict[str, Any] = {}
        cms_data: Dict[str, Any] = {}
        for i, cms in enumerate(self.cms_adapters):
            target = f"cms_{i}.pt"
            if incremental and not cms.is_dirty and save_dir is not None and \
                    (save_dir / target).exists():
                prev = self.memory.get("cms_layers", {}).get(f"layer_{i}")
                if prev:
                    cms_data[f"layer_{i}"] = prev
                continue
            cms.consolidate()
            sd = cms.state_dict()
            tensors[target] = sd
            cms_data[f"layer_{i}"] = {
                "rank": cms.rank,
                "alpha": cms.alpha,
                "frequency": cms.frequency,
                "shape_A": list(sd["A"].shape),
                "shape_B": list(sd["B"].shape),
                "weight_change": cms.fast.weight_change,
                "drift_ratio": cms.fast.drift_ratio,
                "name": getattr(cms.fast, "layer_name", f"layer_{i}"),
            }

        fw_data: Dict[str, Any] = {}
        for i, fw in enumerate(self.fast_weights):
            fw_data[f"layer_{i}"] = {
                "updates": fw.update_count,
                "skipped": fw.skipped_count,
                "surprise": fw.total_surprise,
                "change_norm": fw.weight_change,
                "drift_ratio": fw.drift_ratio,
                "parametrization": fw.parametrization,
                "rank": fw.rank,
            }

        metadata = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "model_type": type(self.model).__name__,
            "model_fingerprint": self.model_fingerprint(),
            "ttt_layer_names": list(self.ttt_layer_names),
            "steps": self.step_counter,
            "interactions": self.interaction_counter,
            "skipped_steps": self.skipped_steps,
            "saved_at": datetime.datetime.now().isoformat(),
            "config": asdict(self.config),
            "cms_layers": cms_data,
            "fast_weight_layers": fw_data,
            "conversation_history": list(self.conversation_history),
            "history_len": len(self.conversation_history),
            "environment": self._environment_metadata(),
            "memory_version": MEMORY_SCHEMA_VERSION,
        }
        return {"tensors": tensors, "metadata": metadata}

    def save_pretrained(self, path: str = "prokopton_model"):
        """Merge learned knowledge into the base model and save with transformers.

        Uses :meth:`commit` so the merge is explicit and one-way.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        for cms in self.cms_adapters:
            cms.consolidate()
            cms.commit()

        for fw in self.fast_weights:
            fw.mark_clean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tok_path = save_path / "prokopton_tokenizers.pt"
            tok_state = {
                "audio_input_proj": self.audio_tokenizer.input_proj.state_dict(),
                "audio_output_proj": self.audio_tokenizer.output_proj.state_dict(),
                "vision_input_proj": self.vision_tokenizer.proj.state_dict(),
                "vision_output_proj": self.vision_tokenizer.output_proj.state_dict(),
            }
        torch.save(tok_state, tok_path)

        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        print(f"💾 Merged model saved: {save_path}/")
        print(f"   Tokenizer weights: {tok_path}")

    def load(self, path: str = None, silent_on_missing: bool = False) -> bool:
        """Restore previously saved knowledge.

        Restore is **idempotent**: it always sets ``W = W0 + Δ_saved`` rather
        than accumulating, so repeated load cycles cannot compound the delta.
        """
        save_dir = Path(path or self.config.save_dir)
        if not save_dir.exists():
            if not silent_on_missing:
                print(f"⚠ Folder not found: {save_dir}")
            return False

        meta_path = save_dir / "metadata.json"
        if not meta_path.exists():
            print("⚠ metadata.json not found")
            return False

        with open(meta_path) as f:
            metadata = json.load(f)

        schema = int(metadata.get("schema_version", metadata.get("memory_version", 1)))
        if schema > MEMORY_SCHEMA_VERSION:
            raise ValueError(
                f"Memory schema v{schema} is newer than this build (v{MEMORY_SCHEMA_VERSION}).")

        fingerprint = metadata.get("model_fingerprint")
        if fingerprint and fingerprint != self.model_fingerprint():
            raise ValueError(
                "Refusing to load memory into a mismatched model.\n"
                f"  checkpoint fingerprint: {fingerprint}\n"
                f"  current model fingerprint: {self.model_fingerprint()}\n"
                "Use save_pretrained() to bake knowledge into a specific model, or "
                "load with the same architecture/layer shapes.")

        loaded = 0
        for i in range(len(self.cms_adapters)):
            cms_path = save_dir / f"cms_{i}.pt"
            if cms_path.exists():
                sd = torch.load(cms_path, map_location=self.model.device, weights_only=True)
                self.cms_adapters[i].load_state_dict(sd)
                self.cms_adapters[i].apply_to_model()
                loaded += 1

        self.step_counter = int(metadata.get("steps", 0))
        self.interaction_counter = int(metadata.get("interactions", self.step_counter))
        self.skipped_steps = int(metadata.get("skipped_steps", 0))
        self.conversation_history = list(metadata.get("conversation_history", []))
        self.memory = metadata

        # Teacher must follow the restored W0.
        self._refresh_anchor_teacher()

        print(f"📂 Loaded: {save_dir}/ ({loaded} layers, "
              f"{metadata.get('steps', 0)} steps, "
              f"{len(self.conversation_history)} messages)")
        return True

    # ── Utility ───────────────────────────────────────────────
    def reset(self):
        """Reset all learned knowledge (RAM only). ``W0`` is untouched."""
        for fw in self.fast_weights:
            fw.reset()
        for cms in self.cms_adapters:
            cms.mark_clean()
        self.replay_buffer = SurpriseBuffer(self.config.per_capacity)
        self.step_counter = 0
        self.interaction_counter = 0
        self.skipped_steps = 0
        self.conversation_history = []
        self._refresh_anchor_teacher()

    @property
    def stats(self) -> Dict[str, Any]:
        base = {
            "steps": self.step_counter,
            "interactions": self.interaction_counter,
            "skipped_steps": self.skipped_steps,
            "updates": sum(fw.update_count for fw in self.fast_weights),
            "weight_change": sum(fw.weight_change for fw in self.fast_weights),
            "max_drift_ratio": max((fw.drift_ratio for fw in self.fast_weights), default=0.0),
            "trust_region": self.config.ttt_trust_region,
            "total_surprise": sum(fw.total_surprise for fw in self.fast_weights),
            "buffer_size": len(self.replay_buffer),
            "history_len": len(self.conversation_history),
            "memory_version": self.memory.get("memory_version", 1),
        }
        for i, fw in enumerate(self.fast_weights):
            base[f"layer_{i}_dw"] = f"{fw.weight_change:.4f}"
            base[f"layer_{i}_drift"] = f"{fw.drift_ratio:.6f}"
            base[f"layer_{i}_updates"] = fw.update_count
            base[f"layer_{i}_skipped"] = fw.skipped_count
            base[f"layer_{i}_eff_lr"] = f"{fw.last_effective_lr:.6f}"
        for i, cms in enumerate(self.cms_adapters):
            base[f"cms_{i}_freq"] = cms.frequency
            base[f"cms_{i}_dirty"] = cms.is_dirty
            base[f"cms_{i}_scale"] = cms.scale
        return base


# ============================================================
# Atomic I/O helpers
# ============================================================

def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _save_tensors(save_dir: Path, tensors: Dict[str, Dict[str, torch.Tensor]]):
    for name, sd in tensors.items():
        target = save_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(sd, tmp)
        os.replace(tmp, target)
