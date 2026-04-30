"""Token-level gene perturbation (delete / overexpress / inhibit).

Mirrors ``maxtoki_mlx.perturbation.perturb_tokens`` so the score we compute
on the BioNeMo PyTorch model is directly comparable to the MLX path.
"""
from __future__ import annotations

from typing import Literal

Direction = Literal["delete", "overexpress", "inhibit"]


def perturb_tokens(
    token_ids: list[int],
    gene_token: int,
    direction: Direction,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    if not token_ids:
        return list(token_ids)
    if token_ids[0] != bos_id or token_ids[-1] != eos_id:
        raise ValueError("token_ids must start with BOS and end with EOS")

    genes = token_ids[1:-1]
    if gene_token not in genes:
        if direction == "overexpress":
            return [bos_id, gene_token] + genes + [eos_id]
        return [bos_id] + genes + [eos_id]

    genes_without = [g for g in genes if g != gene_token]
    if direction == "delete":
        return [bos_id] + genes_without + [eos_id]
    if direction == "overexpress":
        return [bos_id, gene_token] + genes_without + [eos_id]
    if direction == "inhibit":
        return [bos_id] + genes_without + [gene_token, eos_id]
    raise ValueError(f"unknown direction: {direction}")
