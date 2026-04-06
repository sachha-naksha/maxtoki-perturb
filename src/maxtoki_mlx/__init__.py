"""maxtoki-mlx: Apple Silicon port of MaxToki.

Public API:
    load_model(variant)          - load MLX weights for "217m" or "1b"
    CellTokenizer                - rank-value cell tokenizer
    predict_cell_embedding(...)  - forward pass, returns hidden state at EOS
    in_silico_perturb(...)       - delete / overexpress a gene, return score
"""
from __future__ import annotations

from .loader import load_model
from .tokenizer import CellTokenizer
from .inference import (
    predict_cell_embedding,
    run_forward,
    run_backbone,
    embedding_distance,
)
from .perturbation import in_silico_perturb, perturb_tokens

__all__ = [
    "load_model",
    "CellTokenizer",
    "predict_cell_embedding",
    "run_forward",
    "run_backbone",
    "embedding_distance",
    "in_silico_perturb",
    "perturb_tokens",
]

__version__ = "0.1.0"
