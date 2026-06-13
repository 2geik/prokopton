"""Unit tests for Prokopton config module."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from pathlib import Path

from prokopton.config import ProkoptonCLIConfig, load_config, save_config


class TestProkoptonCLIConfig:
    def test_defaults(self):
        cfg = ProkoptonCLIConfig()
        assert cfg.lr == 1e-3
        assert cfg.n_layers == 5
        assert cfg.backend == ""
        assert cfg.model == ""

    def test_save_load_roundtrip(self):
        cfg = ProkoptonCLIConfig(
            model="test/model",
            backend="cuda",
            lr=0.005,
            n_layers=3,
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            cfg.save(tmp_path)
            assert tmp_path.exists()

            loaded = ProkoptonCLIConfig.load(tmp_path)
            assert loaded.model == "test/model"
            assert loaded.backend == "cuda"
            assert loaded.lr == 0.005
            assert loaded.n_layers == 3
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_missing_file(self):
        cfg = ProkoptonCLIConfig.load(Path("/nonexistent/path/config.yaml"))
        assert cfg.lr == 1e-3  # defaults

    def test_merge_cli_args(self):
        cfg = ProkoptonCLIConfig(lr=0.001, n_layers=5)
        cfg.merge_cli_args(lr=0.01, backend="cuda")
        assert cfg.lr == 0.01
        assert cfg.backend == "cuda"
        assert cfg.n_layers == 5  # unchanged

    def test_merge_none_doesnt_override(self):
        cfg = ProkoptonCLIConfig(model="e2b")
        cfg.merge_cli_args(model=None)
        assert cfg.model == "e2b"

    def test_to_prokopton_config(self):
        cfg = ProkoptonCLIConfig(lr=0.005, n_layers=3, cms_rank=8, save_dir="test_mem")
        pc = cfg.to_prokopton_config()
        assert pc.ttt_lr == 0.005
        assert pc.ttt_n_layers == 3
        assert pc.cms_rank == 8
        assert pc.save_dir == "test_mem"

    def test_verbose_quiet(self):
        cfg = ProkoptonCLIConfig(verbose=True, quiet=True)
        assert cfg.verbose
        assert cfg.quiet


def test_load_config_default():
    """load_config() without args returns defaults (no file)."""
    cfg = load_config(Path("/nonexistent/path.yaml"))
    assert isinstance(cfg, ProkoptonCLIConfig)
