"""Inference primitives.

Pretraining model forward passes + hidden state extraction.

Note: the published HF weights are pretraining-only (no temporal head, no time tokens).
They do rank-value next-gene prediction. To get a usable "biological age delta" signal
we extract representations at the <eos> position and use them as cell embeddings.
"""
from __future__ import annotations

from typing import Iterable

import mlx.core as mx
import numpy as np


def _to_batched_input(token_ids: Iterable[int]) -> mx.array:
    arr = np.asarray(list(token_ids), dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"token_ids must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("token_ids must be non-empty")
    return mx.array(arr[None, :])


def run_forward(model, token_ids: Iterable[int]) -> mx.array:
    """Run a single forward pass returning full-vocab logits.

    Args:
        model: An mlx_lm Llama model (top-level Model with .lm_head)
        token_ids: 1D sequence of token IDs (includes BOS/EOS)

    Returns:
        logits: mx.array of shape (1, L, vocab_size)
    """
    batched = _to_batched_input(token_ids)
    logits = model(batched)
    mx.eval(logits)
    return logits


def run_backbone(model, token_ids: Iterable[int]) -> mx.array:
    """Run the backbone (everything before lm_head) returning hidden states.

    This is dramatically cheaper than a full forward when you only need the
    per-cell embedding: we skip the (hidden_size x vocab_size) LM head matmul.

    Args:
        model: An mlx_lm Llama model
        token_ids: 1D sequence of token IDs

    Returns:
        hidden: mx.array of shape (1, L, hidden_size), post final RMSNorm
    """
    batched = _to_batched_input(token_ids)
    hidden = model.model(batched)
    mx.eval(hidden)
    return hidden


def predict_cell_embedding(
    model,
    token_ids: Iterable[int],
    layer: str = "last_hidden",
) -> np.ndarray:
    """Extract a cell embedding from a tokenized cell.

    Args:
        model: An mlx_lm Llama model
        token_ids: 1D list of token IDs (should start with <bos> and end with <eos>)
        layer: "last_hidden" (default) returns the final-layer hidden state at the
               <eos> position (shape hidden_size). This is the cleanest dense
               representation the pretraining model can provide.
               "logits_eos" returns the vocab-sized logit distribution at <eos>.

    Returns:
        numpy array.
    """
    if layer == "last_hidden":
        hidden = run_backbone(model, token_ids)
        return np.asarray(hidden[0, -1])
    elif layer == "logits_eos":
        logits = run_forward(model, token_ids)
        return np.asarray(logits[0, -1])
    else:
        raise ValueError(f"unknown layer: {layer!r}. Must be 'last_hidden' or 'logits_eos'.")


def embedding_distance(emb_a: np.ndarray, emb_b: np.ndarray, metric: str = "cosine") -> float:
    """Compute distance between two cell embeddings.

    Args:
        emb_a, emb_b: 1D arrays of equal length.
        metric: "cosine" (default), "l2", or "l1".

    Returns:
        Non-negative scalar. For cosine, 0.0 = identical direction,
        1.0 = orthogonal (or either vector is zero).
    """
    emb_a = np.asarray(emb_a).ravel()
    emb_b = np.asarray(emb_b).ravel()
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"shape mismatch: {emb_a.shape} vs {emb_b.shape}")
    if metric == "cosine":
        dot = float(np.dot(emb_a, emb_b))
        na = float(np.linalg.norm(emb_a))
        nb = float(np.linalg.norm(emb_b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)
    elif metric == "l2":
        return float(np.linalg.norm(emb_a - emb_b))
    elif metric == "l1":
        return float(np.abs(emb_a - emb_b).sum())
    else:
        raise ValueError(f"unknown metric: {metric!r}. Must be 'cosine', 'l2', or 'l1'.")
