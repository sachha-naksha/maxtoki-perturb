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


def plot_rejuvenation_effect(
    baseline_t: np.ndarray,
    perturbed_t: np.ndarray,
    out_path: Path,
    *,
    pseudotime: Optional[np.ndarray] = None,
    group: Optional[list[str]] = None,
    gene_present: Optional[np.ndarray] = None,
    bin_by: str = "donor",          # "donor" | "pseudotime" | "gene_present" | "single"
    n_bins: int = 6,
    gene_label: str = "",
    units_label: str = "Δ predicted timestep (model units)",
    power: float = 0.4,
    figsize: tuple = (9.0, 5.5),
    cmap_pro: str = "Reds",
    cmap_rej: str = "Blues",
    bar_height: float = 0.62,
    title_fontsize: int = 13,
    label_fontsize: int = 10,
    show_value: bool = True,
    show_n: bool = True,
    sort_by: str = "value",         # "value" | "category"
) -> dict:
    """Divergent bar chart of perturbation effect, oriented by ageing direction.

    Numbers come from the predicted-timestep arrays (the y-axis of the
    pseudotime-correlation plot, fig3b). For each category the row's value is
    mean(perturbed_t - baseline_t).

    Sign convention matches the wet-lab interpretation:
      - mean Δt < 0  →  perturbation predicts cells as *younger* → rejuvenating
                        → bar extends LEFT, Blues colormap.
      - mean Δt > 0  →  perturbation predicts cells as *older* → pro-ageing
                        → bar extends RIGHT, Reds colormap.

    Color *saturation* is power-normed (γ=power, default 0.4) per side, so
    small-magnitude rows still read distinctly — same trick as plot_drivers.
    """
    import matplotlib.colors as mcolors
    from matplotlib import cm

    baseline_t = np.asarray(baseline_t).ravel().astype(float)
    perturbed_t = np.asarray(perturbed_t).ravel().astype(float)
    delta = perturbed_t - baseline_t

    # ---------- build (label, mask) categories ----------
    cats: list[tuple[str, np.ndarray]] = []
    if bin_by == "single":
        cats.append(("all cells", np.ones_like(delta, dtype=bool)))
    elif bin_by == "donor":
        if group is None:
            raise ValueError("bin_by='donor' requires `group`")
        g_arr = np.asarray(group)
        for gv in sorted(set(g_arr)):
            cats.append((str(gv), g_arr == gv))
    elif bin_by == "gene_present":
        if gene_present is None:
            raise ValueError("bin_by='gene_present' requires `gene_present`")
        gp = np.asarray(gene_present, dtype=bool)
        cats.append(("gene present", gp))
        cats.append(("gene absent", ~gp))
    elif bin_by == "pseudotime":
        if pseudotime is None:
            raise ValueError("bin_by='pseudotime' requires `pseudotime`")
        pt = np.asarray(pseudotime, dtype=float)
        valid = ~np.isnan(pt)
        edges = np.quantile(pt[valid], np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        for i in range(n_bins):
            mask = valid & (pt > edges[i]) & (pt <= edges[i + 1])
            cats.append((f"pt [{edges[i]:.2f}, {edges[i+1]:.2f}]", mask))
    else:
        raise ValueError(f"unknown bin_by={bin_by!r}")

    # ---------- aggregate ----------
    rows = []
    for label, mask in cats:
        d = delta[mask]
        if d.size == 0:
            continue
        m = float(np.nanmean(d))
        sem = float(np.nanstd(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        rows.append({"label": label, "mean": m, "sem": sem, "n": int(d.size)})
    if not rows:
        raise ValueError("no non-empty categories")

    if sort_by == "value":
        rows.sort(key=lambda r: r["mean"])  # most negative (rejuvenating) at top
    # else: keep insertion order (already category-sorted for donor/pseudotime)

    means = np.array([r["mean"] for r in rows])
    sems = np.array([r["sem"] for r in rows])
    labels = [r["label"] for r in rows]
    ns = [r["n"] for r in rows]
    y = np.arange(len(rows))

    # ---------- color: power-norm of |mean|, separate per side ----------
    pos_mask = means > 0
    neg_mask = means < 0
    abs_pos = np.abs(means[pos_mask])
    abs_neg = np.abs(means[neg_mask])
    pos_norm = mcolors.PowerNorm(
        gamma=power,
        vmin=abs_pos.min() if abs_pos.size else 0.0,
        vmax=abs_pos.max() if abs_pos.size else 1.0,
    )
    neg_norm = mcolors.PowerNorm(
        gamma=power,
        vmin=abs_neg.min() if abs_neg.size else 0.0,
        vmax=abs_neg.max() if abs_neg.size else 1.0,
    )
    cmap_p = cm.get_cmap(cmap_pro)
    cmap_n = cm.get_cmap(cmap_rej)
    colors = np.empty((len(rows), 4))
    if pos_mask.any():
        colors[pos_mask] = cmap_p(0.35 + 0.6 * pos_norm(abs_pos))
    if neg_mask.any():
        colors[neg_mask] = cmap_n(0.35 + 0.6 * neg_norm(abs_neg))
    colors[means == 0] = (0.7, 0.7, 0.7, 1.0)

    # ---------- draw ----------
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    ax.barh(
        y, means, height=bar_height, color=colors,
        edgecolor="white", linewidth=0.5,
        xerr=sems, error_kw=dict(ecolor="#444444", lw=0.8, capsize=2.5, alpha=0.7),
    )

    ax.axvline(0, color="#888888", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel(units_label, fontsize=label_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", colors="#555555")
    ax.grid(False)

    # symmetric x-limits so 0 sits in the middle; pad enough room for
    # outside-the-bar value labels without colliding with the y-tick labels.
    span = float(np.max(np.abs(means) + sems)) if len(means) else 1.0
    span *= 1.45
    ax.set_xlim(-span, span)

    # Value/n annotations placed on the *zero-side* of each bar tip — i.e.
    # always pointing back toward the 0-axis — so they never overlap with
    # the y-tick labels at the outer edge of the plot.
    if show_value or show_n:
        offset = span * 0.012
        for yi, (m, s, n) in enumerate(zip(means, sems, ns)):
            txt_parts = []
            if show_value:
                txt_parts.append(f"{m:+.1f}" + (f" ± {s:.1f}" if s else ""))
            if show_n:
                txt_parts.append(f"n={n}")
            txt = "  ".join(txt_parts)
            if m >= 0:
                # positive bar extends right; place label just left of 0
                ax.text(-offset, yi, txt, va="center", ha="right",
                        fontsize=label_fontsize - 1, color="#333333")
            else:
                # negative bar extends left; place label just right of 0
                ax.text(offset, yi, txt, va="center", ha="left",
                        fontsize=label_fontsize - 1, color="#333333")

    # Side captions in axes-relative coords so they stay anchored to the
    # top corners regardless of how many rows there are.
    ax.text(
        0.0, 1.04, "← REJUVENATING  (anti-ageing)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=label_fontsize, color=cmap_n(0.85), fontweight="semibold",
    )
    ax.text(
        1.0, 1.04, "PRO-AGEING →",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=label_fontsize, color=cmap_p(0.85), fontweight="semibold",
    )

    title = "Perturbation rejuvenation profile"
    if gene_label:
        title = f"{gene_label}: " + title
    sub = f"split by {bin_by}" + (f" ({n_bins} quantile bins)" if bin_by == "pseudotime" else "")
    ax.set_title(f"{title}\n{sub}", fontsize=title_fontsize, pad=28, fontweight="semibold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "rows": rows,
        "overall_mean_delta_t": float(np.nanmean(delta)),
        "overall_n": int(delta.size),
    }


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

    rejuv_donor = plot_rejuvenation_effect(
        scores["baseline"], scores["perturbed"],
        viz_dir / "rejuvenation_by_donor.png",
        group=meta["group"], bin_by="donor", gene_label=gene_label,
    )
    rejuv_pt = plot_rejuvenation_effect(
        scores["baseline"], scores["perturbed"],
        viz_dir / "rejuvenation_by_pseudotime.png",
        pseudotime=meta["query_pseudotime"], bin_by="pseudotime", n_bins=6,
        gene_label=gene_label,
    )
    rejuv_gp = plot_rejuvenation_effect(
        scores["baseline"], scores["perturbed"],
        viz_dir / "rejuvenation_by_gene_present.png",
        gene_present=scores["gene_present"], bin_by="gene_present",
        gene_label=gene_label,
    )

    all_stats = {
        **corr_stats, **baseline_stats, **delta_stats,
        "pairwise_mannwhitney_p": donor_stats.get("pairwise_mannwhitney_p", {}),
        "rejuvenation_by_donor": rejuv_donor["rows"],
        "rejuvenation_by_pseudotime": rejuv_pt["rows"],
        "rejuvenation_by_gene_present": rejuv_gp["rows"],
        "overall_mean_delta_t": rejuv_donor["overall_mean_delta_t"],
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