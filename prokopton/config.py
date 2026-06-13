"""
Prokopton Config — YAML-based persistent configuration.

Reads/writes ~/.prokopton.yaml for default settings.
CLI flags override config file values.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


DEFAULT_CONFIG_PATH = Path.home() / ".prokopton.yaml"


@dataclass
class ProkoptonCLIConfig:
    """Persistent CLI configuration."""

    # Model
    model: str = ""
    backend: str = ""  # "", "rocm", "cuda", "mps", "mlx", "cpu"

    # TTT
    lr: float = 1e-3
    n_layers: int = 5

    # Storage
    save_dir: str = "prokopton_memory"
    models_dir: str = "models"

    # Display
    verbose: bool = False
    quiet: bool = False

    # CMS
    cms_rank: int = 16

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ProkoptonCLIConfig":
        """Load config from YAML file, return defaults if not found."""
        config_path = path or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return cls()

        # Filter to known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: Optional[Path] = None):
        """Save config to YAML file."""
        config_path = path or DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def merge_cli_args(self, **kwargs) -> "ProkoptonCLIConfig":
        """Override config fields with CLI arguments (non-None values win)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_prokopton_config(self):
        """Convert to ProkoptonConfig for the framework."""
        # Import here to avoid circular dependency
        from prokopton.core import ProkoptonConfig
        return ProkoptonConfig(
            ttt_lr=self.lr,
            ttt_n_layers=self.n_layers,
            cms_rank=self.cms_rank,
            save_dir=self.save_dir,
        )


def load_config(path: Optional[Path] = None) -> ProkoptonCLIConfig:
    """Load Prokopton CLI config from default location."""
    return ProkoptonCLIConfig.load(path)


def save_config(config: ProkoptonCLIConfig, path: Optional[Path] = None):
    """Save Prokopton CLI config to default location."""
    config.save(path)
