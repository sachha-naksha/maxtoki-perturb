"""Tests for the rank-value cell tokenizer."""
from __future__ import annotations

import numpy as np
import pytest

from maxtoki_mlx import CellTokenizer
from maxtoki_mlx.tokenizer import MODEL_INPUT_SIZE, TARGET_SUM


def test_tokenizer_loads_with_expected_vocab(tokenizer: CellTokenizer):
    assert len(tokenizer) == 20277  # 4 specials + 20271 genes + 2 query tokens
    assert tokenizer.num_genes == 20271
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.pad_id == 0


def test_every_gene_has_a_median(tokenizer: CellTokenizer):
    # Sanity: no gene in the vocab should be missing a median
    for ens in tokenizer._gene_ids:
        assert ens in tokenizer.gene_median, f"missing median for {ens}"


def test_empty_cell_returns_bos_eos_only(tokenizer: CellTokenizer):
    tokens = tokenizer.tokenize_expression(
        ensembl_ids=["ENSG00000109906"],
        expression=np.array([0.0]),  # zero expression
        n_counts=1.0,
    )
    assert tokens == [tokenizer.bos_id, tokenizer.eos_id]


def test_unknown_genes_are_silently_dropped(tokenizer: CellTokenizer):
    # Mix real and fake genes - fake ones should be ignored
    real_gene = next(iter(tokenizer._gene_ids))
    tokens = tokenizer.tokenize_expression(
        ensembl_ids=[real_gene, "ENSG99999999999", "NOT_A_GENE"],
        expression=np.array([10.0, 10.0, 10.0]),
        n_counts=30.0,
    )
    # Only the real gene should survive
    assert len(tokens) == 3  # BOS + 1 gene + EOS
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id
    assert tokens[1] == tokenizer._gene_ids[real_gene]


def test_rank_ordering_is_descending(tokenizer: CellTokenizer):
    """After median normalization, tokens should be ordered by descending normalized expression."""
    gene_a, gene_b, gene_c = list(tokenizer._gene_ids)[:3]
    med_a = tokenizer.gene_median[gene_a]
    med_b = tokenizer.gene_median[gene_b]
    med_c = tokenizer.gene_median[gene_c]

    # Construct expressions such that after dividing by median, order is b > a > c
    n_counts = 1000.0
    expr_a = (2.0 * med_a) * n_counts / TARGET_SUM
    expr_b = (5.0 * med_b) * n_counts / TARGET_SUM
    expr_c = (1.0 * med_c) * n_counts / TARGET_SUM

    tokens = tokenizer.tokenize_expression(
        ensembl_ids=[gene_a, gene_b, gene_c],
        expression=np.array([expr_a, expr_b, expr_c]),
        n_counts=n_counts,
    )
    # Expect: BOS, b, a, c, EOS
    expected = [
        tokenizer.bos_id,
        tokenizer._gene_ids[gene_b],
        tokenizer._gene_ids[gene_a],
        tokenizer._gene_ids[gene_c],
        tokenizer.eos_id,
    ]
    assert tokens == expected


def test_length_mismatch_raises(tokenizer: CellTokenizer):
    with pytest.raises(ValueError, match="length"):
        tokenizer.tokenize_expression(
            ensembl_ids=["ENSG00000109906"],
            expression=np.array([1.0, 2.0]),
        )


def test_zero_n_counts_raises(tokenizer: CellTokenizer):
    with pytest.raises(ValueError, match="n_counts"):
        tokenizer.tokenize_expression(
            ensembl_ids=["ENSG00000109906"],
            expression=np.array([1.0]),
            n_counts=0.0,
        )


def test_truncation_to_max_len(tokenizer: CellTokenizer):
    # Build a cell with more than max_len-2 expressed genes
    gene_list = list(tokenizer._gene_ids)[:100]
    expression = np.ones(100)
    tokens = tokenizer.tokenize_expression(
        gene_list, expression, n_counts=100.0, max_len=10
    )
    assert len(tokens) == 10
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id


def test_decode_gene_roundtrip(tokenizer: CellTokenizer):
    ensembl = next(iter(tokenizer._gene_ids))
    token_id = tokenizer._gene_ids[ensembl]
    assert tokenizer.decode_gene(token_id) == ensembl
    # Non-gene tokens should return None
    assert tokenizer.decode_gene(tokenizer.bos_id) is None
    assert tokenizer.decode_gene(99999999) is None


def test_zero_expression_genes_dropped(tokenizer: CellTokenizer):
    """Genes with zero expression should not appear in the output."""
    gene_a, gene_b, gene_c = list(tokenizer._gene_ids)[:3]
    tokens = tokenizer.tokenize_expression(
        ensembl_ids=[gene_a, gene_b, gene_c],
        expression=np.array([5.0, 0.0, 3.0]),
        n_counts=8.0,
    )
    # Only gene_a and gene_c should appear
    inner = tokens[1:-1]
    assert len(inner) == 2
    assert tokenizer._gene_ids[gene_b] not in inner
