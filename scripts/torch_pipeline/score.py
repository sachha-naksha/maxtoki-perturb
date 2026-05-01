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
        stem = f.stem
        try:
            rank = int(stem.split("rank_")[1].split("__")[0])
        except (IndexError, ValueError):
            continue
        try:
            out[rank] = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[score] skipping {f.name}: load error: {e}")
    return out


def _flatten_regression(pred_dict: dict) -> np.ndarray:
    if pred_dict is None:
        return np.array([])
    if "regression_preds" not in pred_dict:
        if "predictions" in pred_dict and isinstance(pred_dict["predictions"], list):
            arrs = [_flatten_regression(p) for p in pred_dict["predictions"]]
            return np.concatenate(arrs) if arrs else np.array([])
        return np.array([])
    pred = pred_dict["regression_preds"]
    if pred is None:
        return np.array([])
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().float().cpu().numpy()
    return np.asarray(pred).reshape(-1)


def sink_predictions(
    predictions_dir: str | Path,
    expected_n: int | None = None,
) -> np.ndarray:
    """Gather predictions from all ranks into a single 1-D array in row order.

    Layout:
      - tp-only / pp-only (dp_size==1): only rank 0 has non-empty preds; rank>0
        files are MP partial stubs. We drop empty arrays silently.
      - dp>1: each rank holds contiguous rows. Concatenating in numeric order
        recovers original sequence (Lightning's DistributedSampler shuffle=False).
    """
    rank_to_dict = _load_rank_files(Path(predictions_dir))
    parts: list[np.ndarray] = []
    for r in sorted(rank_to_dict):
        arr = _flatten_regression(rank_to_dict[r])
        if arr.size == 0:
            continue
        parts.append(arr)
    out = np.concatenate(parts) if parts else np.array([])
    if expected_n is not None and out.size != expected_n:
        print(f"[sink] WARNING: gathered {out.size} predictions, expected {expected_n}; "
              f"per-rank sizes={[p.size for p in parts]}")
    return out


def load_predictions(predictions_dir: str | Path) -> np.ndarray:
    """Backwards-compatible alias for sink_predictions."""
    return sink_predictions(predictions_dir)


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
    expected_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ScoreSummary]:
    """Read both prediction dirs, compute per-cell MSE, return arrays + summary."""
    base = load_predictions(baseline_dir, expected_n=expected_n)
    pert = load_predictions(perturbed_dir, expected_n=expected_n)
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
