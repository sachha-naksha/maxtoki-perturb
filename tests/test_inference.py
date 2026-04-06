"""Tests for model inference primitives."""
from __future__ import annotations

import numpy as np
import pytest

from maxtoki_mlx import (
    CellTokenizer,
    embedding_distance,
    predict_cell_embedding,
    run_backbone,
    run_forward,
)


def _make_tokens(tokenizer: CellTokenizer, n_genes: int = 20, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    gene_list = list(tokenizer._gene_ids)
    picked = rng.choice(len(gene_list), size=n_genes, replace=False)
    ensembl_ids = [gene_list[i] for i in picked]
    counts = np.ones(n_genes) * 10.0
    return tokenizer.tokenize_expression(ensembl_ids, counts)


def test_run_forward_returns_logit_shape(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=20)
    logits = run_forward(model, tokens)
    assert logits.shape == (1, len(tokens), 20275)


def test_run_backbone_returns_hidden_shape(model, tokenizer, model_and_config):
    _, config = model_and_config
    tokens = _make_tokens(tokenizer, n_genes=20)
    hidden = run_backbone(model, tokens)
    assert hidden.shape == (1, len(tokens), config["hidden_size"])


def test_forward_no_nan_or_inf(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=50)
    logits = run_forward(model, tokens)
    arr = np.asarray(logits)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_embedding_is_deterministic(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=30, seed=7)
    e1 = predict_cell_embedding(model, tokens)
    e2 = predict_cell_embedding(model, tokens)
    assert np.array_equal(e1, e2)


def test_embedding_layers(model, tokenizer, model_and_config):
    _, config = model_and_config
    tokens = _make_tokens(tokenizer, n_genes=10)
    last_hidden = predict_cell_embedding(model, tokens, layer="last_hidden")
    logits_eos = predict_cell_embedding(model, tokens, layer="logits_eos")
    assert last_hidden.shape == (config["hidden_size"],)
    assert logits_eos.shape == (20275,)


def test_unknown_layer_raises(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=5)
    with pytest.raises(ValueError, match="unknown layer"):
        predict_cell_embedding(model, tokens, layer="fake_layer")


def test_embedding_distance_cosine():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert embedding_distance(a, b, metric="cosine") == pytest.approx(0.0, abs=1e-7)

    c = np.array([0.0, 1.0, 0.0])
    assert embedding_distance(a, c, metric="cosine") == pytest.approx(1.0, abs=1e-7)


def test_embedding_distance_l2():
    a = np.array([1.0, 0.0])
    b = np.array([4.0, 4.0])
    assert embedding_distance(a, b, metric="l2") == pytest.approx(5.0)


def test_embedding_distance_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        embedding_distance(np.array([1.0]), np.array([1.0, 2.0]))


def test_embedding_distance_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        embedding_distance(np.array([1.0]), np.array([2.0]), metric="bogus")


def test_embedding_distance_zero_vector_safe():
    zero = np.zeros(5)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Zero vector → cosine returns 1.0 (max distance) by convention
    assert embedding_distance(zero, a) == pytest.approx(1.0)
    assert embedding_distance(a, zero) == pytest.approx(1.0)


def test_empty_tokens_raises(model):
    with pytest.raises(ValueError, match="non-empty"):
        run_forward(model, [])
