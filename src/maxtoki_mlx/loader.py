"""Model loading helpers.

Wraps mlx_lm.utils.load_model with MaxToki-specific path defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple

from mlx_lm.utils import load_model as _mlx_load_model


Variant = Literal["217m", "1b"]


_DEFAULT_PATHS: dict[str, Path] = {
    "217m": Path("maxtoki-217m-mlx"),
    "1b": Path("maxtoki-1b-mlx"),
}


def resolve_path(variant_or_path: str | Path) -> Path:
    """Resolve a variant name or path to a concrete directory."""
    if isinstance(variant_or_path, Path):
        return variant_or_path
    key = variant_or_path.lower()
    if key in _DEFAULT_PATHS:
        # Check env override
        env_key = f"MAXTOKI_MLX_{key.upper()}_PATH"
        override = os.environ.get(env_key)
        if override:
            return Path(override)
        return _DEFAULT_PATHS[key]
    return Path(variant_or_path)


def load_model(variant_or_path: str | Path = "217m"):
    """Load a MaxToki MLX model.

    Args:
        variant_or_path: Either "217m", "1b", or a filesystem path to an MLX model dir.

    Returns:
        (model, config) tuple. Model is an mlx.nn.Module.
    """
    path = resolve_path(variant_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MLX model directory not found: {path}. "
            f"Run `mlx_lm.convert --hf-path theodoris-lab/MaxToki/MaxToki-217M-HF "
            f"--mlx-path maxtoki-217m-mlx --dtype float32` first."
        )
    model, config = _mlx_load_model(path)
    model.eval()
    return model, config
