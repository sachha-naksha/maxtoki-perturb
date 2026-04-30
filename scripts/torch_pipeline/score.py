"""Read PredictionWriter outputs and compute zero-shot temporal MSE.

Upstream ``PredictionWriter`` writes one ``predictions__rank_{R}.pt`` per
rank when ``write_interval="epoch"``. Each file is a torch-pickled dict
holding (at least):

    regression_preds:      tensor[1, B]   - predicted timestep per row
    timelapse_token_preds: tensor[1, B]   - argmax numeric token per row

We assume baseline and perturbed datasets are emitted in identical row
order (``dataset_prep.build_paired_dataset`` guarantees this) so we can
join on row index. Per-cell:

    delta_t(c) = pert_t[c] - base_t[c]
    mse(c)     = delta_t(c)^2

Dataset-level summary: mean(mse), mean(delta_t), n_cells_with_gene.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _load_rank_files(predictions_dir: Path) -> dict[int, dict]:
    files = sorted(predictions_dir.glob("predictions__rank_*.pt"))
    if not files:
        raise FileNotFoundError(f"no predictions__rank_*.pt under {predictions_dir}")
    out = {}
    for f in files:
        # parse rank from filename: predictions__rank_<R>.pt or .._batch_<B>.pt
        stem = f.stem
        try:
            rank = int(stem.split("rank_")[1].split("__")[0])
        except (IndexError, ValueError):
            continue
        out[rank] = torch.load(f, map_location="cpu", weights_only=False)
    return out


def _flatten_regression(pred_dict: dict) -> np.ndarray:
    """Return a 1D numpy array of regression predictions, in original row
    order. Handles both per-batch list-of-dicts and the epoch-aggregated dict
    that PredictionWriter produces."""
    if "regression_preds" not in pred_dict:
        # epoch-mode wraps a list of per-batch dicts under "predictions"
        if "predictions" in pred_dict and isinstance(pred_dict["predictions"], list):
            arrs = [_flatten_regression(p) for p in pred_dict["predictions"]]
            return np.concatenate(arrs) if arrs else np.array([])
        raise KeyError(f"no 'regression_preds' in dict (keys={list(pred_dict)})")
    pred = pred_dict["regression_preds"]
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().float().cpu().numpy()
    return np.asarray(pred).reshape(-1)


def load_predictions(predictions_dir: str | Path) -> np.ndarray:
    """Load all ranks and return concatenated regression preds in row order."""
    rank_to_dict = _load_rank_files(Path(predictions_dir))
    parts = [_flatten_regression(rank_to_dict[r]) for r in sorted(rank_to_dict)]
    return np.concatenate(parts) if parts else np.array([])


@dataclass
class ScoreSummary:
    """Dataset-level summary of one zero-shot perturbation run.

    Attributes:
        n_cells: Total number of scored rows (queries).
        n_cells_with_gene: Subset where the perturbed gene was actually
            present in the query cell's expression profile.
        mean_mse: ``mean((perturbed_t - baseline_t) ** 2)`` across all rows.
        mean_mse_present: Same, restricted to ``gene_present`` rows.
        mean_delta_t: Signed mean of ``perturbed_t - baseline_t``.
            Negative -> model thinks the perturbation rejuvenates;
            positive -> ages.
        mean_delta_t_present: Signed mean restricted to ``gene_present``.
        abs_mean_delta_t: Mean of absolute ``delta_t``.
        direction: ``inhibit`` / ``delete`` / ``overexpress``.
        gene_ensembl: Resolved Ensembl ID of the perturbed gene.
        gene_token: Token ID in the model's vocabulary.
    """
    n_cells: int
    n_cells_with_gene: int
    mean_mse: float
    mean_mse_present: float
    mean_delta_t: float
    mean_delta_t_present: float
    abs_mean_delta_t: float
    direction: str
    gene_ensembl: str
    gene_token: int


def score(
    baseline_dir: str | Path,
    perturbed_dir: str | Path,
    paired_metadata: Optional[dict] = None,
    out_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ScoreSummary]:
    """Read both prediction dirs, compute per-cell MSE, return arrays + summary."""
    base = load_predictions(baseline_dir)
    pert = load_predictions(perturbed_dir)
    if base.shape != pert.shape:
        raise ValueError(
            f"baseline / perturbed prediction count mismatch: "
            f"{base.shape} vs {pert.shape}"
        )
    delta = pert - base
    per_cell_mse = delta ** 2

    meta = paired_metadata or {}
    gene_present_mask = np.asarray(meta.get("gene_present", [True] * len(delta)), dtype=bool)
    n_present = int(gene_present_mask.sum())

    summary = ScoreSummary(
        n_cells=int(len(delta)),
        n_cells_with_gene=n_present,
        mean_mse=float(per_cell_mse.mean()),
        mean_mse_present=(
            float(per_cell_mse[gene_present_mask].mean()) if n_present else float("nan")
        ),
        mean_delta_t=float(delta.mean()),
        mean_delta_t_present=(
            float(delta[gene_present_mask].mean()) if n_present else float("nan")
        ),
        abs_mean_delta_t=float(np.abs(delta).mean()),
        direction=str(meta.get("direction", "")),
        gene_ensembl=str(meta.get("gene_ensembl", "")),
        gene_token=int(meta.get("gene_token", -1)),
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            baseline=base,
            perturbed=pert,
            delta_t=delta,
            per_cell_mse=per_cell_mse,
            gene_present=gene_present_mask,
            summary=np.array(json.dumps(asdict(summary))),
        )

    return base, pert, delta, summary
