"""Shared pytest fixtures for maxtoki-mlx."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from maxtoki_mlx import CellTokenizer, load_model


_MLX_217M_PATH = Path(__file__).resolve().parents[2] / "maxtoki-217m-mlx"


def _model_path() -> Path:
    override = os.environ.get("MAXTOKI_MLX_217M_PATH")
    if override:
        return Path(override)
    return _MLX_217M_PATH


@pytest.fixture(scope="session")
def tokenizer() -> CellTokenizer:
    return CellTokenizer()


@pytest.fixture(scope="session")
def model_and_config():
    path = _model_path()
    if not path.exists():
        pytest.skip(f"MLX 217M model not found at {path}")
    return load_model(path)


@pytest.fixture(scope="session")
def model(model_and_config):
    return model_and_config[0]
