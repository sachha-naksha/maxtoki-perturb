"""Shared pytest fixtures.

``maxtoki_mlx`` is MLX-only and ships on the x86 build. Importing it at module
scope breaks collection on DeltaAI (ARM/CUDA container has no MLX). We defer
the import into the fixtures that need it so pure-torch tests (NextCell
pipeline, bionemo path) collect and run without the MLX dep.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


_MLX_217M_PATH = Path(__file__).resolve().parents[2] / "maxtoki-217m-mlx"


def _model_path() -> Path:
    override = os.environ.get("MAXTOKI_MLX_217M_PATH")
    if override:
        return Path(override)
    return _MLX_217M_PATH


@pytest.fixture(scope="session")
def tokenizer():
    try:
        from maxtoki_mlx import CellTokenizer
    except ImportError as e:
        pytest.skip(f"maxtoki_mlx not installed: {e}")
    return CellTokenizer()


@pytest.fixture(scope="session")
def model_and_config():
    try:
        from maxtoki_mlx import load_model
    except ImportError as e:
        pytest.skip(f"maxtoki_mlx not installed: {e}")
    path = _model_path()
    if not path.exists():
        pytest.skip(f"MLX 217M model not found at {path}")
    return load_model(path)


@pytest.fixture(scope="session")
def model(model_and_config):
    return model_and_config[0]
