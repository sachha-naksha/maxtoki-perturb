"""In-silico gene perturbation.

Semantics follow Geneformer's InSilicoPerturber:
    - "delete": remove the gene's token entirely from the rank sequence
    - "overexpress": move the gene's token to the front (highest rank)
    - "inhibit": move the gene's token to the back (lowest rank, but still present)
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .inference import predict_cell_embedding, embedding_distance


Direction = Literal["delete", "overexpress", "inhibit"]


def perturb_tokens(
    token_ids: list[int],
    gene_token: int,
    direction: Direction,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """Apply a perturbation to a tokenized cell.

    Args:
        token_ids: Full token list including BOS and EOS
        gene_token: The token ID of the gene to perturb
        direction: "delete" | "overexpress" | "inhibit"
        bos_id: BOS token id (to skip during search)
        eos_id: EOS token id (to skip during search)
    """
    # Strip BOS/EOS, work on gene portion
    if not token_ids:
        return list(token_ids)
    if token_ids[0] != bos_id or token_ids[-1] != eos_id:
        raise ValueError("token_ids must start with BOS and end with EOS")

    genes = token_ids[1:-1]

    # Locate the gene
    if gene_token not in genes:
        # Gene not in this cell's expression profile - no-op
        # For overexpress, we still insert at front; for delete/inhibit, no change
        if direction == "overexpress":
            return [bos_id, gene_token] + genes + [eos_id]
        return [bos_id] + genes + [eos_id]

    # Remove current occurrence
    genes_without = [g for g in genes if g != gene_token]

    if direction == "delete":
        return [bos_id] + genes_without + [eos_id]
    elif direction == "overexpress":
        return [bos_id, gene_token] + genes_without + [eos_id]
    elif direction == "inhibit":
        return [bos_id] + genes_without + [gene_token, eos_id]
    else:
        raise ValueError(f"unknown direction: {direction}")


def in_silico_perturb(
    model,
    token_ids: list[int],
    gene_token: int,
    direction: Direction,
    bos_id: int,
    eos_id: int,
    metric: str = "cosine",
) -> dict:
    """Perturb one gene in a cell and quantify the representation shift.

    Args:
        model: loaded MaxToki MLX model
        token_ids: original tokenized cell [<bos>, ..., <eos>]
        gene_token: token id of the gene to perturb
        direction: perturbation direction
        bos_id, eos_id: special tokens for safety-checking the sequence
        metric: "cosine" | "l2" | "l1"

    Returns:
        dict with:
            baseline_embedding: ndarray
            perturbed_embedding: ndarray
            distance: float (representation shift magnitude)
            gene_present: bool (was the gene expressed in the input cell?)
            tokens_before: int (number of gene tokens in original)
            tokens_after: int (number in perturbed)
    """
    gene_present = gene_token in token_ids
    baseline = predict_cell_embedding(model, token_ids)
    perturbed_tokens = perturb_tokens(token_ids, gene_token, direction, bos_id, eos_id)
    perturbed = predict_cell_embedding(model, perturbed_tokens)
    dist = embedding_distance(baseline, perturbed, metric=metric)
    return {
        "baseline_embedding": baseline,
        "perturbed_embedding": perturbed,
        "distance": dist,
        "gene_present": bool(gene_present),
        "tokens_before": len(token_ids) - 2,
        "tokens_after": len(perturbed_tokens) - 2,
        "direction": direction,
    }
