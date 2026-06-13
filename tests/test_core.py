"""Unit tests for Prokopton core module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import pytest

from prokopton.core import (
    Prokopton,
    ProkoptonConfig,
    FastWeight,
    CMSAdapter,
    SurpriseBuffer,
    VisualTokenizer,
    AudioTokenizer,
)


class TestProkoptonConfig:
    def test_defaults(self):
        cfg = ProkoptonConfig()
        assert cfg.ttt_lr == 1e-3
        assert cfg.ttt_momentum == 0.9
        assert cfg.ttt_n_layers == 5
        assert cfg.cms_rank == 16
        assert cfg.per_capacity == 128

    def test_custom(self):
        cfg = ProkoptonConfig(ttt_lr=0.01, ttt_n_layers=3, cms_rank=8)
        assert cfg.ttt_lr == 0.01
        assert cfg.ttt_n_layers == 3
        assert cfg.cms_rank == 8


class TestFastWeight:
    def test_init(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        assert fw.update_count == 0
        assert torch.allclose(fw.delta, torch.zeros_like(lin.weight))
        assert torch.allclose(fw.original_W, lin.weight)

    def test_reset(self):
        lin = nn.Linear(8, 8)
        original = lin.weight.clone()
        fw = FastWeight(lin)
        with torch.no_grad():
            lin.weight.add_(torch.randn_like(lin.weight) * 0.1)
        fw.reset()
        assert torch.allclose(lin.weight, original)

    def test_delta(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        with torch.no_grad():
            lin.weight.add_(0.5)
        assert not torch.allclose(fw.delta, torch.zeros_like(lin.weight))


class TestCMSAdapter:
    def test_init(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        assert cms.rank == 4
        assert cms.A.shape == (4, 8)
        assert cms.B.shape == (8, 4)

    def test_consolidate_and_expand(self):
        lin = nn.Linear(16, 16)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        with torch.no_grad():
            lin.weight.add_(torch.randn_like(lin.weight) * 0.5)
        cms.consolidate()
        delta = cms.expand()
        assert delta.shape == (16, 16)

    def test_save_load_state(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        sd = cms.state_dict()
        cms2 = CMSAdapter(FastWeight(nn.Linear(8, 8)), rank=4)
        cms2.load_state_dict(sd)
        assert torch.allclose(cms.A, cms2.A)
        assert torch.allclose(cms.B, cms2.B)

    def test_apply_to_model(self):
        lin = nn.Linear(8, 8)
        fw = FastWeight(lin)
        cms = CMSAdapter(fw, rank=4)
        orig = fw.original_W.clone()
        with torch.no_grad():
            lin.weight.add_(0.3)
        cms.consolidate()
        cms.apply_to_model()
        assert not torch.allclose(fw.original_W, orig)


class TestSurpriseBuffer:
    def test_push_and_len(self):
        buf = SurpriseBuffer(10)
        assert len(buf) == 0
        buf.push("a", 0.5)
        buf.push("b", 0.9)
        assert len(buf) == 2

    def test_sample(self):
        buf = SurpriseBuffer(20)
        for i in range(10):
            buf.push(f"item{i}", 0.5 + i * 0.05)
        sample = buf.sample(3)
        assert len(sample) == 3
        # The highest-surprise items should be more likely
        # (probabilistic, so just check format)
        assert all(isinstance(s, str) for s in sample)

    def test_capacity_limit(self):
        buf = SurpriseBuffer(5)
        for i in range(10):
            buf.push(f"item{i}", 0.1)
        assert len(buf) == 5

    def test_empty_sample(self):
        buf = SurpriseBuffer(10)
        assert buf.sample(5) == []


class TestVisualTokenizer:
    def test_output_shape(self):
        vt = VisualTokenizer(patch_size=48, embed_dim=128, output_dim=256)
        img = torch.randn(1, 3, 96, 96)
        tokens, info = vt(img)
        assert tokens.ndim == 3
        assert tokens.shape[-1] == 256

    def test_multiple_images(self):
        vt = VisualTokenizer(patch_size=32, embed_dim=64, output_dim=128)
        img = torch.randn(2, 3, 64, 64)
        tokens, info = vt(img)
        assert tokens.shape[0] == 2

    def test_non_divisible(self):
        vt = VisualTokenizer(patch_size=48, embed_dim=128, output_dim=256)
        img = torch.randn(1, 3, 100, 100)  # not divisible by 48
        tokens, info = vt(img)
        assert tokens.ndim == 3  # should pad and work


class TestAudioTokenizer:
    def test_output_shape(self):
        import math
        at = AudioTokenizer(sample_rate=16000, n_mels=40, patch_frames=4,
                           embed_dim=128, output_dim=256)
        audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, 16000))
        tokens, info = at(audio)
        assert len(tokens) == 1
        assert tokens[0].shape[-1] == 256

    def test_short_audio(self):
        import math
        at = AudioTokenizer(sample_rate=16000, n_mels=40, patch_frames=4,
                           embed_dim=128, output_dim=256)
        audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 0.01, 160))  # very short
        tokens, info = at(audio)
        assert len(tokens) == 1
        assert tokens[0].shape[-1] == 256


class TestProkoptonCore:
    """Integration test with a small dummy model."""

    @pytest.fixture
    def dummy_model_and_tokenizer(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': 64})()
                self.embed = nn.Embedding(100, 64)
                self.ln1 = nn.Linear(64, 64)
                self.ln2 = nn.Linear(64, 64)
                self.ln3 = nn.Linear(64, 100)
                self._dev = torch.device('cpu')

            def forward(self, input_ids):
                B, T = input_ids.shape
                h = self.embed(input_ids)
                h = torch.relu(self.ln1(h))
                h = torch.relu(self.ln2(h))
                logits = self.ln3(h)
                return type('obj', (object,), {'logits': logits})()

            def generate(self, **kw):
                return torch.tensor([[1, 2, 3]])

            @property
            def device(self):
                return self._dev

            def train(self): self.training = True
            def eval(self): self.training = False
            def zero_grad(self):
                for p in self.parameters():
                    if p.grad is not None: p.grad = None

        class SimpleTok:
            eos_token = "</s>"
            eos_token_id = 0

            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                ids = [min(ord(c) % 100 + 1, 99) for c in text[:max_length or 256]]
                return {"input_ids": torch.tensor([ids])}

            def decode(self, ids, skip_special_tokens=True):
                return "simple response"

            @property
            def pad_token(self):
                return self.eos_token

            @pad_token.setter
            def pad_token(self, v):
                pass

        return SimpleModel(), SimpleTok()

    def test_prokopton_create(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=2, auto_save_every=0, per_capacity=8)
        prok = Prokopton(model, tok, cfg)
        assert len(prok.fast_weights) > 0
        assert len(prok.cms_adapters) > 0

    def test_learn(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=2, auto_save_every=0, per_capacity=8)
        prok = Prokopton(model, tok, cfg)
        result = prok.learn("hello world test")
        assert "loss" in result
        assert "step" in result
        assert result["step"] == 1

    def test_learn_multiple(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=4)
        prok = Prokopton(model, tok, cfg)
        for i in range(5):
            r = prok.learn(f"test message {i}")
            assert r["step"] == i + 1

    def test_stats(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=4)
        prok = Prokopton(model, tok, cfg)
        prok.learn("test")
        s = prok.stats
        assert s["steps"] == 1
        assert s["updates"] >= 1

    def test_save_load_reset(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=4)
        prok = Prokopton(model, tok, cfg)

        prok.learn("important fact about Zephyria")

        import tempfile, os
        tmp = tempfile.mkdtemp()
        try:
            prok.save(tmp)
            assert os.path.exists(os.path.join(tmp, "metadata.json"))

            prok.reset()
            assert prok.step_counter == 0

            loaded = prok.load(tmp)
            assert loaded
        finally:
            import shutil
            shutil.rmtree(tmp)

    def test_chat(self, dummy_model_and_tokenizer):
        model, tok = dummy_model_and_tokenizer
        cfg = ProkoptonConfig(ttt_n_layers=1, auto_save_every=0, per_capacity=4)
        prok = Prokopton(model, tok, cfg)
        resp = prok.chat("hello", max_new=5)
        assert isinstance(resp, str)
        assert len(resp) > 0
