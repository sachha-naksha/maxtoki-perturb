"""Tests for in-silico gene perturbation."""
from __future__ import annotations

import numpy as np
import pytest

from maxtoki_mlx import (
    CellTokenizer,
    in_silico_perturb,
    perturb_tokens,
)


def _make_tokens(tokenizer: CellTokenizer, n_genes: int = 20, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    gene_list = list(tokenizer._gene_ids)
    picked = rng.choice(len(gene_list), size=n_genes, replace=False)
    ensembl_ids = [gene_list[i] for i in picked]
    counts = np.arange(1, n_genes + 1, dtype=float)
    return tokenizer.tokenize_expression(ensembl_ids, counts)


# --- perturb_tokens (pure function, no model) ---


def test_perturb_tokens_delete_present_gene(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10)
    gene = tokens[3]
    out = perturb_tokens(tokens, gene, "delete", tokenizer.bos_id, tokenizer.eos_id)
    assert gene not in out
    assert len(out) == len(tokens) - 1
    assert out[0] == tokenizer.bos_id
    assert out[-1] == tokenizer.eos_id


def test_perturb_tokens_delete_absent_gene_is_noop(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10)
    # Find a gene that's NOT in the sequence
    absent_gene = None
    for tid in tokenizer._gene_ids.values():
        if tid not in tokens:
            absent_gene = tid
            break
    out = perturb_tokens(tokens, absent_gene, "delete", tokenizer.bos_id, tokenizer.eos_id)
    assert out == tokens


def test_perturb_tokens_overexpress_moves_to_front(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10)
    gene = tokens[5]  # somewhere in the middle
    out = perturb_tokens(tokens, gene, "overexpress", tokenizer.bos_id, tokenizer.eos_id)
    assert out[0] == tokenizer.bos_id
    assert out[1] == gene
    # Gene appears exactly once
    assert out.count(gene) == 1
    assert len(out) == len(tokens)


def test_perturb_tokens_overexpress_absent_gene_inserts(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10)
    absent_gene = None
    for tid in tokenizer._gene_ids.values():
        if tid not in tokens:
            absent_gene = tid
            break
    out = perturb_tokens(tokens, absent_gene, "overexpress", tokenizer.bos_id, tokenizer.eos_id)
    assert out[0] == tokenizer.bos_id
    assert out[1] == absent_gene
    assert len(out) == len(tokens) + 1


def test_perturb_tokens_inhibit_moves_to_back(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10)
    gene = tokens[2]
    out = perturb_tokens(tokens, gene, "inhibit", tokenizer.bos_id, tokenizer.eos_id)
    assert out[-1] == tokenizer.eos_id
    assert out[-2] == gene
    assert out.count(gene) == 1


def test_perturb_tokens_rejects_unwrapped_input(tokenizer):
    with pytest.raises(ValueError, match="BOS"):
        perturb_tokens([100, 200, 300], 100, "delete", tokenizer.bos_id, tokenizer.eos_id)


def test_perturb_tokens_unknown_direction(tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=5)
    with pytest.raises(ValueError, match="unknown direction"):
        perturb_tokens(tokens, tokens[1], "explode", tokenizer.bos_id, tokenizer.eos_id)


def test_perturb_tokens_empty_input(tokenizer):
    out = perturb_tokens([], 100, "delete", tokenizer.bos_id, tokenizer.eos_id)
    assert out == []


# --- in_silico_perturb (with model) ---


def test_in_silico_perturb_delete_changes_embedding(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=15, seed=1)
    gene = tokens[3]
    result = in_silico_perturb(
        model, tokens, gene, "delete", tokenizer.bos_id, tokenizer.eos_id
    )
    assert result["gene_present"] is True
    assert result["tokens_before"] == 15
    assert result["tokens_after"] == 14
    assert result["distance"] > 1e-5
    assert result["direction"] == "delete"


def test_in_silico_perturb_absent_gene_zero_distance(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=15, seed=2)
    absent_gene = None
    for tid in tokenizer._gene_ids.values():
        if tid not in tokens:
            absent_gene = tid
            break
    result = in_silico_perturb(
        model, tokens, absent_gene, "delete", tokenizer.bos_id, tokenizer.eos_id
    )
    assert result["gene_present"] is False
    assert result["distance"] < 1e-6


def test_in_silico_perturb_deterministic(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10, seed=3)
    r1 = in_silico_perturb(model, tokens, tokens[1], "delete", tokenizer.bos_id, tokenizer.eos_id)
    r2 = in_silico_perturb(model, tokens, tokens[1], "delete", tokenizer.bos_id, tokenizer.eos_id)
    assert r1["distance"] == r2["distance"]


def test_in_silico_perturb_all_directions(model, tokenizer):
    tokens = _make_tokens(tokenizer, n_genes=10, seed=4)
    gene = tokens[5]
    results = {
        d: in_silico_perturb(model, tokens, gene, d, tokenizer.bos_id, tokenizer.eos_id)
        for d in ["delete", "overexpress", "inhibit"]
    }
    # All three should have matching metadata
    for d, r in results.items():
        assert r["direction"] == d
        assert r["gene_present"] is True
        assert "baseline_embedding" in r
        assert "perturbed_embedding" in r
