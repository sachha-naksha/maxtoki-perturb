"""Paper-style visualizations for zero-shot perturbation results.

Aligned with MaxToki paper Figure 3 (B, C):
  - 3B: predicted timestep vs ground-truth pseudotime correlation (Pearson + Spearman)
  - 3C: model |error| vs most-common-pseudotime baseline (Wilcoxon rank-sums)

Plus perturbation-specific plots:
  - delta_t distribution (KDE), split by gene_present
  - |delta_t| vs pseudotime
  - Per-donor delta_t boxplot (sanity check)

Logs all figures to a W&B run if provided; saves PNGs to ``out_dir/viz/``
either way. Also emits ``viz_stats.json`` with the headline numbers
(Pearson r, Wilcoxon p, model MAE, baseline MAE).

Run as a script (re-attaches to a wandb run by id):

    python scripts/torch_pipeline/viz.py \\
        --out-dir ./out/tp53_217m_smoketest \\
        --wandb-project maxtoki-perturb \\
        --wandb-run-id upj1gkhv

Or import ``run_viz`` and call from a notebook.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _is_main_rank() -> bool:
    """Return True only on global rank 0; matches the helper in run_inhibit_temporal_mse.py."""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "PMI_RANK"):
        v = os.environ.get(var)
        if v is not None:
            return int(v) == 0
    return True


def _load_scores(scores_npz: Path) -> dict:
    data = np.load(scores_npz, allow_pickle=True)
    return {
        "baseline": np.asarray(data["baseline"]).ravel(),
        "perturbed": np.asarray(data["perturbed"]).ravel(),
        "delta_t": np.asarray(data["delta_t"]).ravel(),
        "per_cell_mse": np.asarray(data["per_cell_mse"]).ravel(),
        "gene_present": np.asarray(data["gene_present"]).astype(bool),
    }


def _load_metadata(base_ds_dir: Path) -> dict:
    import datasets
    ds = datasets.load_from_disk(str(base_ds_dir))
    return {
        "cell_id": [r["cell_id"] for r in ds],
        "query_pseudotime": np.array(
            [float(r["query_pseudotime"]) if r["query_pseudotime"] is not None else np.nan
             for r in ds]
        ),
        "group": [str(r["group"]) for r in ds],
    }


def plot_pseudotime_correlation(
    predicted_t: np.ndarray,
    true_pseudotime: np.ndarray,
    out_path: Path,
    title: str = "Predicted vs ground-truth pseudotime",
    groups: Optional[list[str]] = None,
) -> dict:
    """Figure 3B style: scatter + regression + Pearson r / Spearman ρ."""
    from scipy import stats

    mask = ~np.isnan(true_pseudotime)
    x = true_pseudotime[mask]
    y = predicted_t[mask]
    g = (np.asarray(groups)[mask] if groups is not None else None)

    if len(x) < 2:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"), "n": int(len(x))}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    if g is not None:
        for i, gv in enumerate(sorted(set(g))):
            sel = g == gv
            ax.scatter(x[sel], y[sel], alpha=0.5, s=10, label=f"{gv} (n={sel.sum()})", color=f"C{i}")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.scatter(x, y, alpha=0.4, s=8, color="C0")

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="C3", linestyle="--", linewidth=1.5,
            label=f"y = {slope:.3f} x + {intercept:.2f}")

    ax.set_xlabel("Ground-truth pseudotime")
    ax.set_ylabel("Predicted timestep (model)")
    ax.set_title(
        f"{title}\nPearson r = {pearson_r:.3f}  (p = {pearson_p:.2e}),  "
        f"Spearman ρ = {spearman_r:.3f},  n = {len(x)}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "n": int(len(x)),
    }


def plot_baseline_comparison(
    predicted_t: np.ndarray,
    true_pseudotime: np.ndarray,
    out_path: Path,
    cell_type_label: str = "skeletal muscle",
) -> dict:
    """Figure 3C style: model |error| vs most-common-pseudotime baseline."""
    from scipy import stats

    mask = ~np.isnan(true_pseudotime)
    pred = predicted_t[mask]
    true = true_pseudotime[mask]

    if len(true) < 2:
        return {"wilcoxon_p": float("nan"), "model_mae": float("nan"), "baseline_mae": float("nan")}

    baseline_pred = np.full_like(true, np.median(true))
    model_err = np.abs(pred - true)
    baseline_err = np.abs(baseline_pred - true)

    stat, p_value = stats.ranksums(model_err, baseline_err)

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(
        [model_err, baseline_err],
        labels=["MaxToki", f"baseline\n(median {cell_type_label})"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C7"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    sig = "***" if p_value < 1e-3 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
    ax.set_ylabel("|predicted - true| pseudotime")
    ax.set_title(
        f"Pseudo-timelapse prediction error  ({sig})\n"
        f"Wilcoxon rank-sums p = {p_value:.2e},  n = {len(pred)}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "wilcoxon_p": float(p_value),
        "model_mae": float(model_err.mean()),
        "baseline_mae": float(baseline_err.mean()),
        "n": int(len(pred)),
    }


def plot_delta_t_distribution(
    delta_t: np.ndarray,
    gene_present: np.ndarray,
    out_path: Path,
    gene_label: str = "",
) -> dict:
    """Δt KDE / histogram split by gene_present, with Wilcoxon p vs 0."""
    from scipy import stats as _stats

    p_present = delta_t[gene_present]
    p_absent = delta_t[~gene_present]

    fig, ax = plt.subplots(figsize=(7, 4))
    lo = float(min(delta_t.min(), -1.0))
    hi = float(max(delta_t.max(), 1.0))
    bins = np.linspace(lo, hi, 30)

    if len(p_present):
        ax.hist(p_present, bins=bins, alpha=0.6, label=f"gene present (n={len(p_present)})", color="C0")
    if len(p_absent):
        ax.hist(p_absent, bins=bins, alpha=0.6, label=f"gene absent (n={len(p_absent)})", color="C7")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    p_value = float("nan")
    if len(p_present) > 1 and not np.allclose(p_present, 0):
        try:
            _, p_value = _stats.wilcoxon(p_present, alternative="two-sided")
        except ValueError:
            p_value = float("nan")
    sig = "***" if p_value < 1e-3 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
    ax.set_title(
        f"Δt distribution: {gene_label}\n"
        f"Wilcoxon (gene-present cells vs 0): p = {p_value:.2e} ({sig})"
    )
    ax.set_xlabel("Δt = perturbed - baseline (timestep units)")
    ax.set_ylabel("Number of cells")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"wilcoxon_present_vs_zero_p": p_value}


def plot_delta_t_vs_pseudotime(
    delta_t: np.ndarray,
    pseudotime: np.ndarray,
    gene_present: np.ndarray,
    group: list[str],
    out_path: Path,
    gene_label: str = "",
) -> None:
    """Δt vs query pseudotime, colored by donor."""
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = ~np.isnan(pseudotime)
    pt = pseudotime[mask]
    dt = delta_t[mask]
    gp = np.asarray(group)[mask]

    for i, g in enumerate(sorted(set(gp))):
        sel = gp == g
        ax.scatter(pt[sel], dt[sel], alpha=0.5, s=10, label=f"{g} (n={sel.sum()})", color=f"C{i}")

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Query cell pseudotime")
    ax.set_ylabel("Δt = perturbed - baseline")
    ax.set_title(f"{gene_label}: perturbation effect across pseudotime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_donor_delta(delta_t: np.ndarray, group: list[str], out_path: Path, gene_label: str = "") -> dict:
    """Boxplot of Δt by donor with pairwise Mann-Whitney."""
    from scipy import stats as _stats

    g_arr = np.asarray(group)
    unique = sorted(set(g_arr))
    data_by = [delta_t[g_arr == g] for g in unique]

    pairwise = {}
    if len(unique) >= 2:
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = data_by[i], data_by[j]
                if len(a) > 0 and len(b) > 0 and not (np.allclose(a, 0) and np.allclose(b, 0)):
                    try:
                        _, p = _stats.mannwhitneyu(a, b)
                        pairwise[f"{unique[i]}_vs_{unique[j]}"] = float(p)
                    except ValueError:
                        pass

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data_by, labels=unique, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(f"C{i}")
        patch.set_alpha(0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Δt = perturbed - baseline")
    title = f"{gene_label}: Δt by donor"
    if pairwise:
        first_p = next(iter(pairwise.values()))
        title += f"  (first pair p = {first_p:.2e}, Mann-Whitney)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"pairwise_mannwhitney_p": pairwise}


def run_viz(out_dir: Path, wandb_run=None, cell_type_label: str = "skeletal muscle") -> dict:
    """Master entrypoint. Reads ``out_dir/scores.npz`` + ``out_dir/baseline.dataset``,
    produces all plots, writes them under ``out_dir/viz/``, optionally logs to wandb.

    Skips silently on non-main ranks (defense in depth; main() should already
    have early-returned by the time it would call this).
    """
    if not _is_main_rank():
        return {}

    out_dir = Path(out_dir)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    scores = _load_scores(out_dir / "scores.npz")
    meta = _load_metadata(out_dir / "baseline.dataset")
    summary = json.loads((out_dir / "summary.json").read_text())
    gene_label = f"{summary['gene_ensembl']} ({summary['direction']})"
    variant = summary.get("variant", "?")

    corr_stats = plot_pseudotime_correlation(
        scores["baseline"], meta["query_pseudotime"],
        viz_dir / "fig3b_pseudotime_correlation.png",
        title=f"Pseudotime correlation - {variant}",
        groups=meta["group"],
    )

    baseline_stats = plot_baseline_comparison(
        scores["baseline"], meta["query_pseudotime"],
        viz_dir / "fig3c_baseline_comparison.png",
        cell_type_label=cell_type_label,
    )

    delta_stats = plot_delta_t_distribution(
        scores["delta_t"], scores["gene_present"],
        viz_dir / "delta_t_distribution.png",
        gene_label=gene_label,
    )

    plot_delta_t_vs_pseudotime(
        scores["delta_t"], meta["query_pseudotime"],
        scores["gene_present"], meta["group"],
        viz_dir / "delta_t_vs_pseudotime.png",
        gene_label=gene_label,
    )

    donor_stats = plot_per_donor_delta(
        scores["delta_t"], meta["group"],
        viz_dir / "per_donor_delta.png",
        gene_label=gene_label,
    )

    all_stats = {
        **corr_stats, **baseline_stats, **delta_stats,
        "pairwise_mannwhitney_p": donor_stats.get("pairwise_mannwhitney_p", {}),
        "gene": summary["gene_ensembl"],
        "direction": summary["direction"],
        "variant": variant,
    }
    (viz_dir / "viz_stats.json").write_text(json.dumps(all_stats, indent=2))

    if wandb_run is not None:
        import wandb
        for png in sorted(viz_dir.glob("*.png")):
            wandb_run.log({f"viz/{png.stem}": wandb.Image(str(png))})
        flat_summary = {f"viz/{k}": v for k, v in all_stats.items() if isinstance(v, (int, float))}
        wandb_run.summary.update(flat_summary)

    return all_stats


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="Output dir from run_inhibit_temporal_mse.py")
    p.add_argument("--wandb-project", default=None,
                   help="If set with --wandb-run-id, append plots to that run")
    p.add_argument("--wandb-run-id", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--cell-type-label", default="skeletal muscle")
    args = p.parse_args()

    wandb_run = None
    if args.wandb_run_id and args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
        )
    stats = run_viz(Path(args.out_dir), wandb_run=wandb_run, cell_type_label=args.cell_type_label)
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()