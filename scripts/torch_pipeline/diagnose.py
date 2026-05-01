"""Diagnose Δt=0 in a perturbation run.

Verifies the entire pipeline up to the prediction tensors:
  - input_ids actually differ between baseline and perturbed where they should
  - the gene token actually appears in baseline cells (gene_present sanity)
  - the raw regression_preds tensors actually differ between runs
  - the spread / quantization of the model's numeric-token output

Prints a verdict aligned with four hypotheses: input bug, bf16 quantization,
217M decision-lock, or a real but small effect.

Run on a finished --out-dir:
    python scripts/torch_pipeline/diagnose.py --out-dir ./out/tp53_217m_smoketest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_pred_dir(d: Path) -> list[dict]:
    files = sorted(Path(d).glob("predictions__rank_*.pt"))
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def _flatten(preds: list[dict], key: str) -> np.ndarray:
    out = []
    for p in preds:
        if key in p and p[key] is not None:
            out.append(p[key].float().cpu().numpy().reshape(-1))
        elif "predictions" in p and isinstance(p["predictions"], list):
            for q in p["predictions"]:
                if key in q and q[key] is not None:
                    out.append(q[key].float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sample-rows", type=int, default=50,
                   help="how many rows to spot-check input_ids on")
    args = p.parse_args()

    out = args.out_dir
    summary = json.loads((out / "summary.json").read_text())
    gene_token = int(summary["gene_token"])
    print(f"[diag] gene={summary['gene_ensembl']} token={gene_token} "
          f"variant={summary['variant']}")

    # 1. input_ids really differ between baseline + perturbed
    import datasets
    base_ds = datasets.load_from_disk(str(out / "baseline.dataset"))
    pert_ds = datasets.load_from_disk(str(out / "perturbed.dataset"))
    n_check = min(len(base_ds), args.sample_rows)
    n_diff = 0
    n_present_in_query = 0
    pos_shifts = []
    for i in range(n_check):
        b_ids = base_ds[i]["input_ids"]
        p_ids = pert_ds[i]["input_ids"]
        if b_ids != p_ids:
            n_diff += 1
        if base_ds[i]["gene_present_in_query"]:
            n_present_in_query += 1
            b_pos = [j for j, t in enumerate(b_ids) if t == gene_token]
            p_pos = [j for j, t in enumerate(p_ids) if t == gene_token]
            if b_pos and p_pos:
                pos_shifts.append(p_pos[-1] - b_pos[-1])
    print(f"[diag] sampled {n_check} rows: input_ids differ in {n_diff}, "
          f"gene_present_in_query in {n_present_in_query}")
    if pos_shifts:
        ps = np.array(pos_shifts)
        print(f"[diag] gene token last-position shift (perturb - base): "
              f"min={ps.min()} max={ps.max()} mean={ps.mean():.1f}")

    if n_diff == 0:
        print("[diag] *** BUG *** input_ids identical across all sampled rows. "
              "perturbation is a no-op. check perturb_tokens / dataset_prep.")

    # 2. prediction tensors really differ
    base_preds = _load_pred_dir(out / "baseline_predictions")
    pert_preds = _load_pred_dir(out / "perturbed_predictions")
    print(f"[diag] prediction files: baseline={len(base_preds)}, "
          f"perturbed={len(pert_preds)}")

    br = _flatten(base_preds, "regression_preds")
    pr = _flatten(pert_preds, "regression_preds")
    print(f"[diag] baseline regression_preds: shape={br.shape}, "
          f"min={br.min():.4f} max={br.max():.4f} "
          f"distinct(round 4dp)={len(np.unique(br.round(4)))}")
    print(f"[diag] perturbed regression_preds: shape={pr.shape}, "
          f"min={pr.min():.4f} max={pr.max():.4f} "
          f"distinct(round 4dp)={len(np.unique(pr.round(4)))}")

    if len(br) == len(pr):
        delta = pr - br
        n_zero = int(np.count_nonzero(delta == 0))
        n_total = len(delta)
        print(f"[diag] delta_t: min={delta.min():.6f} max={delta.max():.6f} "
              f"|mean|={np.abs(delta).mean():.6f} "
              f"exactly_zero={n_zero}/{n_total}")
        bit_eq = bool(np.array_equal(br, pr))
        print(f"[diag] raw regression_preds bit-identical: {bit_eq}")
    else:
        bit_eq = None

    bt = _flatten(base_preds, "timelapse_token_preds")
    pt = _flatten(pert_preds, "timelapse_token_preds")
    if len(bt) > 0 and len(pt) > 0 and len(bt) == len(pt):
        n_argmax_diff = int((bt != pt).sum())
        print(f"[diag] timelapse_token_preds (argmax) changed in "
              f"{n_argmax_diff}/{len(bt)} cells")

    # Verdict
    print("\n[diag] === verdict ===")
    if n_diff == 0:
        print("  -> input pipeline bug; perturbation never reaches the model")
    elif bit_eq is True:
        print("  -> raw regression tensors identical despite different inputs.")
        print("     primary suspect: bf16 precision quantization erasing logit deltas.")
        print("     fix: re-run with --precision 32-true (or 16-mixed). "
              "if 1B is available, also worth trying.")
    elif bit_eq is False and n_total > 0 and n_zero == n_total:
        print("  -> raw tensors differ but every per-cell pair rounds to identical.")
        print("     suggests downstream casting/quantization in score.py or PredictionWriter.")
    elif len(np.unique(br.round(2))) <= 3:
        print("  -> baseline regression_preds cluster on <= 3 distinct values.")
        print("     headless temporal head is decision-locked; perturbation cannot move it.")
        print("     try the 1B variant.")
    else:
        print("  -> looks healthy: predictions vary across cells AND between "
              "baseline/perturbed. if delta_t magnitude is just small, that may be the "
              "real biological signal for this gene/cohort. compare against a "
              "non-expressed-gene control to establish noise floor.")


if __name__ == "__main__":
    main()