"""Zero-shot temporal-MSE scoring for in-silico gene inhibition.

For each cell c and target gene g:
    baseline_t   = temporal_head(backbone(tokens(c)))
    perturbed_t  = temporal_head(backbone(perturb(tokens(c), g, "inhibit")))
    delta_t(c,g) = perturbed_t - baseline_t                # signed
    mse(c,g)     = mean( delta_t(c,g) ** 2 )               # always >= 0

We report:
    * dataset-level mean MSE (the paper's "temporal MSE" summary score)
    * per-cell delta_t (kept signed so direction is recoverable)

All forwards are batched and run under ``torch.inference_mode``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import MaxTokiTemporal
from .perturbation import perturb_tokens, Direction


@dataclass
class CellResult:
    cell_index: int
    cell_id: str
    gene_present: bool
    baseline: np.ndarray       # (T,)
    perturbed: np.ndarray      # (T,)
    delta: np.ndarray          # (T,)
    mse: float                 # scalar


def _pad_batch(
    seqs: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    n = len(seqs)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.inference_mode()
def temporal_batch(
    model: MaxTokiTemporal,
    seqs: list[list[int]],
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute temporal-head outputs for a list of token sequences. Returns (N, T)."""
    device = next(model.parameters()).device
    pad_id = model.config.pad_id
    outs: list[torch.Tensor] = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        ids, mask = _pad_batch(chunk, pad_id, device)
        outs.append(model.temporal(ids, mask).float().cpu())
    return torch.cat(outs, dim=0)


def score_inhibition(
    model: MaxTokiTemporal,
    cells: list,           # list[TokenizedCell] from data.iter_tokenized_cells
    gene_token: int,
    direction: Direction = "inhibit",
    batch_size: int = 16,
) -> tuple[list[CellResult], dict]:
    """Score one perturbation across the whole dataset.

    Returns (per-cell results, dataset-level summary).
    """
    bos_id = model.config.bos_id
    eos_id = model.config.eos_id

    baselines = [c.token_ids for c in cells]
    perturbed = [
        perturb_tokens(c.token_ids, gene_token, direction, bos_id, eos_id)
        for c in cells
    ]

    base_t = temporal_batch(model, baselines, batch_size=batch_size).numpy()
    pert_t = temporal_batch(model, perturbed, batch_size=batch_size).numpy()
    delta = pert_t - base_t  # (N, T)

    # per-cell MSE across temporal-head output dims
    per_cell_mse = (delta ** 2).mean(axis=1)

    results: list[CellResult] = []
    for i, c in enumerate(cells):
        results.append(
            CellResult(
                cell_index=c.cell_index,
                cell_id=c.cell_id,
                gene_present=gene_token in c.token_ids,
                baseline=base_t[i],
                perturbed=pert_t[i],
                delta=delta[i],
                mse=float(per_cell_mse[i]),
            )
        )

    n_present = sum(r.gene_present for r in results)
    summary = {
        "n_cells": len(results),
        "n_cells_with_gene": n_present,
        "mean_mse": float(per_cell_mse.mean()),
        "mean_mse_present": (
            float(per_cell_mse[[r.gene_present for r in results]].mean())
            if n_present > 0
            else float("nan")
        ),
        "mean_delta": delta.mean(axis=0).tolist(),
        "direction": direction,
    }
    return results, summary
