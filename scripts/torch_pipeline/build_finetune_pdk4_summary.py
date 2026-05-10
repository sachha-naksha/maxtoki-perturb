"""Aggregate PDK4 predict runs from the Stage-2 finetuning setup.

Walks `out/finetune_skm_*pdk4*` (and the legacy `finetune_skm_predict*`),
pulls headline metrics from each summary.json + per-cell delta_t from
scores.npz, writes:

  out/finetune_pdk4_predict_summary.csv  one row per predict run
  out/finetune_pdk4_predict_summary.png  per-cell delta_t distributions
                                         + bar chart of run means

The 0-shot biologist-validated PDK4 inhibition signal (Δt = -37.4 over
2000 cells) lives on a different magnitude and is referenced in the title
rather than overlaid on the per-cell axis.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("out")

# (label, dir-under-out, note)
RUNS = [
    ("50step_smoke",     "finetune_skm_predict_pdk4",     "50-step smoke, n=16"),
    ("v2",               "finetune_skm_v2_pdk4",          "v2 weights, n=2000"),
    ("v2_persample",     "finetune_skm_v2_pdk4_persample","v2 weights w/ per-sample report"),
    ("v4_persample",     "finetune_skm_v4_pdk4_persample","v4 weights w/ per-sample report"),
    ("1k_eval",          "finetune_skm_1k_pdk4_eval",     "1k weights, n=32 eval"),
    ("1k_full",          "finetune_skm_1k_pdk4_full",     "1k weights, n=2000 full"),
]


def _flatten_summary(label: str, note: str, run_dir: Path) -> dict | None:
    sp = run_dir / "summary.json"
    if not sp.exists():
        print(f"[skip] {run_dir} has no summary.json")
        return None
    s = json.loads(sp.read_text())
    p = s.get("perturbation") or {}
    b = s.get("baseline") or {}
    per = p.get("per_sample") or {}

    n = int(s.get("n_queries", 0))
    n_neg = int(p.get("n_negative", 0))
    row = {
        "run":                   label,
        "note":                  note,
        "run_dir":               run_dir.name,
        "finetune_dir":          Path(s.get("finetune_dir", "")).name,
        "n_queries":             n,
        "seq_length":            s.get("seq_length"),
        "K":                     s.get("K"),
        "context_sample":        s.get("context_sample"),
        "T_MEAN":                round(float(s.get("T_MEAN", float("nan"))), 4),
        "T_STD":                 round(float(s.get("T_STD",  float("nan"))), 4),
        "baseline_t_pred_mean":  round(float(b.get("t_pred_mean", float("nan"))), 4),
        "baseline_t_pred_std":   round(float(b.get("t_pred_std",  float("nan"))), 4),
        "baseline_pearson_r":    round(float(b.get("pearson_r_to_target", float("nan"))), 4),
        "gene":                  p.get("gene", ""),
        "direction":             p.get("direction", ""),
        "delta_t_mean":          round(float(p.get("delta_t_mean",   float("nan"))), 5),
        "delta_t_std":           round(float(p.get("delta_t_std",    float("nan"))), 5),
        "delta_t_median":        round(float(p.get("delta_t_median", float("nan"))), 5),
        "n_negative":            n_neg,
        "frac_negative":         round(n_neg / n, 4) if n else float("nan"),
        "OM6_n":                 (per.get("OM6") or {}).get("n", ""),
        "OM6_delta_t_mean":      round(float((per.get("OM6") or {}).get("delta_t_mean", float("nan"))), 5)
                                   if "OM6" in per else "",
        "OM6_n_negative":        (per.get("OM6") or {}).get("n_negative", ""),
        "OM9_n":                 (per.get("OM9") or {}).get("n", ""),
        "OM9_delta_t_mean":      round(float((per.get("OM9") or {}).get("delta_t_mean", float("nan"))), 5)
                                   if "OM9" in per else "",
        "OM9_n_negative":        (per.get("OM9") or {}).get("n_negative", ""),
    }
    return row


def _load_delta_t(run_dir: Path) -> np.ndarray | None:
    sp = run_dir / "scores.npz"
    if not sp.exists():
        return None
    d = np.load(sp, allow_pickle=False)
    if "delta_t" not in d.files:
        return None
    return np.asarray(d["delta_t"], dtype=np.float64)


def main() -> None:
    rows: list[dict] = []
    deltas: dict[str, np.ndarray] = {}
    for label, run_name, note in RUNS:
        run_dir = OUT_DIR / run_name
        row = _flatten_summary(label, note, run_dir)
        if row is None:
            continue
        rows.append(row)
        dt = _load_delta_t(run_dir)
        if dt is not None and dt.size:
            deltas[label] = dt

    if not rows:
        raise SystemExit("no PDK4 finetune-predict runs found under out/")

    # mark "best" (most-negative delta_t mean among full eval runs, n>=1000)
    full_rows = [r for r in rows if (r["n_queries"] or 0) >= 1000]
    if full_rows:
        best = min(full_rows, key=lambda r: r["delta_t_mean"])
        for r in rows:
            r["is_best_full_eval"] = (r is best)
    else:
        for r in rows:
            r["is_best_full_eval"] = False

    out_csv = OUT_DIR / "finetune_pdk4_predict_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")

    print("\n=== headline ===")
    for r in rows:
        marker = " *" if r.get("is_best_full_eval") else "  "
        print(
            f"{marker}{r['run']:14s} ft={r['finetune_dir']:22s}  "
            f"n={r['n_queries']:>5d}  "
            f"Δt mean={r['delta_t_mean']:+8.4f}  std={r['delta_t_std']:.4f}  "
            f"med={r['delta_t_median']:+.4f}  "
            f"n<0={r['n_negative']}/{r['n_queries']} ({r['frac_negative']:.2%})"
        )

    # ------- viz -------
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # Top: per-run delta_t histograms, stacked with horizontal offsets
    ax = axes[0]
    labels_with_data = [r["run"] for r in rows if r["run"] in deltas]
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(labels_with_data):
        dt = deltas[lab]
        # symmetric clip for visual sanity (mostly small values, but smoke
        # runs sometimes have a few large negatives)
        ax.hist(
            dt, bins=60, alpha=0.55, color=cmap(i % 10),
            label=f"{lab}  n={len(dt)}  mean={dt.mean():+.3f}",
        )
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"per-cell $\Delta t$  (perturbed − baseline, finetune-head t_pred)")
    ax.set_ylabel("count")
    ax.set_title(
        "Stage-2 finetune PDK4-inhibit Δt per query cell\n"
        "0-shot 217M (analytic head) reference: Δt = −37.4 over 2000 cells "
        "(off-axis; finetune scale ≪ 0-shot)"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Bottom: bar of mean delta_t per run, ordered as in RUNS
    ax = axes[1]
    bar_labels = [r["run"] for r in rows]
    means = [r["delta_t_mean"] for r in rows]
    stds = [r["delta_t_std"] for r in rows]
    ns = [r["n_queries"] for r in rows]
    sems = [(s / max(np.sqrt(n), 1.0)) for s, n in zip(stds, ns)]
    colors = ["tab:red" if r.get("is_best_full_eval") else "tab:blue" for r in rows]
    xs = np.arange(len(bar_labels))
    ax.bar(xs, means, yerr=sems, color=colors, alpha=0.85, capsize=3)
    for x, m, n in zip(xs, means, ns):
        ax.text(x, m, f"  {m:+.3f}\n  n={n}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, rotation=20, ha="right")
    ax.set_ylabel(r"mean $\Delta t$ (per-cell, ± SEM)")
    ax.set_title("Mean Δt by run (red = best full-eval run, n≥1000)")

    fig.tight_layout()
    out_png = OUT_DIR / "finetune_pdk4_predict_summary.png"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".svg"))
    print(f"wrote {out_png}")
    print(f"wrote {out_png.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
