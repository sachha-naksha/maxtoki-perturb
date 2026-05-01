"""Negative-control panel for zero-shot perturbation health checks.

Without a noise floor, a single |Δt|=0.5 from one gene tells you nothing.
This script gives you N control genes whose perturbation Δt distribution
is the empirical null, then plots your target gene's distribution against
that null with a significance test.

Three subcommands:

    pick      pick N control genes per strategy, write to JSON
    run       run the same perturbation pipeline on each control
              (shells out to run_inhibit_temporal_mse.py)
    compare   aggregate target + controls, plot null-vs-target,
              compute Mann-Whitney p, log to W&B

Strategies for ``pick``:

    random        N random ENSGs from the model vocabulary (seeded)
    unexpressed   N genes with bottom-decile mean expression across the
                  query cells - true negatives: perturb_tokens is largely
                  a no-op for absent genes, so any non-zero Δt here is
                  pure model noise. Best control type for most cases.
    housekeeping  fixed list of well-known housekeeping genes (ACTB, GAPDH,
                  B2M, RPL13, RPL19, RPS18, HPRT1, TBP). Highly expressed
                  but biologically irrelevant for aging trajectories;
                  Δt should be near zero if the model is well-calibrated.

Run order:

    # 1. pick controls
    python scripts/torch_pipeline/negative_control.py pick \\
        --strategy unexpressed --n 8 \\
        --spec scripts/torch_pipeline/configs/asachan_young34_old80.yaml \\
        --tokenizer-path /weights/MaxToki-217M-bionemo/context/token_dictionary.json \\
        --h5ad ./data/preprocessed.h5ad \\
        --out ./out/controls.json

    # 2. run them (sequentially shown; SLURM array-friendly)
    python scripts/torch_pipeline/negative_control.py run \\
        --controls ./out/controls.json \\
        --target-cmd "python scripts/torch_pipeline/run_inhibit_temporal_mse.py \\
            --spec scripts/torch_pipeline/configs/asachan_young34_old80.yaml \\
            --h5ad ./data/preprocessed.h5ad \\
            --ckpt-dir /weights/MaxToki-217M-bionemo \\
            --tokenizer-path /weights/MaxToki-217M-bionemo/context/token_dictionary.json \\
            --variant 217m --direction inhibit \\
            --wandb-project maxtoki-perturb --wandb-tags negative_control" \\
        --out-dir-prefix ./out/control_

    # 3. compare against your target run
    python scripts/torch_pipeline/negative_control.py compare \\
        --target-out-dir ./out/tp53_217m_smoketest \\
        --control-out-dirs ./out/control_*_217m \\
        --out-dir ./out/control_comparison \\
        --wandb-project maxtoki-perturb \\
        --wandb-run-id <target run id>      # appends to the existing run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Bootstrap so sibling .py files import when run as `python file.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .spec import load_spec
    from .tokenizer import CellTokenizer
except ImportError:
    from spec import load_spec
    from tokenizer import CellTokenizer


HOUSEKEEPING_GENES = {
    "ACTB":   "ENSG00000075624",
    "GAPDH":  "ENSG00000111640",
    "B2M":    "ENSG00000166710",
    "RPL13":  "ENSG00000167526",
    "RPL19":  "ENSG00000108298",
    "RPS18":  "ENSG00000231500",
    "HPRT1":  "ENSG00000165704",
    "TBP":    "ENSG00000112592",
    "PPIA":   "ENSG00000196262",
    "PGK1":   "ENSG00000102144",
}


# ---------------------------------------------------------------------------
# pick subcommand
# ---------------------------------------------------------------------------


def _filter_query_cells(adata, spec) -> np.ndarray:
    """Return boolean mask over adata.n_obs of cells matching query.filter_obs."""
    n = adata.n_obs
    keep = np.ones(n, dtype=bool)
    for col, allowed in (spec.query.filter_obs or {}).items():
        if col not in adata.obs.columns:
            continue
        col_vals = adata.obs[col].values
        allowed_set = set(allowed)
        keep &= np.array([v in allowed_set for v in col_vals], dtype=bool)
    return keep


def pick_random(tokenizer: CellTokenizer, n: int, seed: int = 0,
                exclude: Optional[set[str]] = None) -> list[tuple[str, Optional[str]]]:
    rng = random.Random(seed)
    pool = [k for k in tokenizer.token_dict if k.startswith("ENSG")]
    if exclude:
        pool = [g for g in pool if g not in exclude]
    chosen = rng.sample(pool, min(n, len(pool)))
    return [(g, None) for g in chosen]


def pick_unexpressed(adata, tokenizer: CellTokenizer, query_mask: np.ndarray,
                     n: int, seed: int = 0,
                     exclude: Optional[set[str]] = None) -> list[tuple[str, Optional[str]]]:
    """Pick N genes from the bottom expression decile in query cells."""
    import scipy.sparse as sp

    if "ensembl_id" in adata.var.columns:
        var_ensgs = adata.var["ensembl_id"].astype(str).values
    elif "feature_id" in adata.var.columns:
        var_ensgs = adata.var["feature_id"].astype(str).values
    else:
        var_ensgs = np.asarray(adata.var_names.astype(str).values)

    X = adata.raw.X if adata.raw is not None else adata.X
    X_q = X[query_mask] if query_mask.any() else X
    if sp.issparse(X_q):
        mean_expr = np.asarray(X_q.mean(axis=0)).ravel()
    else:
        mean_expr = X_q.mean(axis=0)

    in_vocab = np.array([
        ensg in tokenizer.token_dict and ensg.startswith("ENSG")
        for ensg in var_ensgs
    ], dtype=bool)
    if exclude:
        excl_arr = np.array([ensg in exclude for ensg in var_ensgs], dtype=bool)
        in_vocab &= ~excl_arr

    if not in_vocab.any():
        sys.exit("no in-vocab genes available for unexpressed selection")

    candidate_idxs = np.where(in_vocab)[0]
    candidate_means = mean_expr[candidate_idxs]
    cutoff = np.quantile(candidate_means, 0.10)
    bottom = candidate_idxs[candidate_means <= cutoff]

    rng = random.Random(seed)
    pick_n = min(n, len(bottom))
    chosen_idx = rng.sample(list(bottom), pick_n)
    return [(str(var_ensgs[i]), None) for i in chosen_idx]


def pick_housekeeping(tokenizer: CellTokenizer, n: int) -> list[tuple[str, Optional[str]]]:
    out: list[tuple[str, Optional[str]]] = []
    for sym, ensg in HOUSEKEEPING_GENES.items():
        if ensg in tokenizer.token_dict:
            out.append((ensg, sym))
        if len(out) >= n:
            break
    return out


def cmd_pick(args) -> None:
    os.environ.setdefault("MAXTOKI_TOKEN_DICT", args.tokenizer_path)
    tokenizer = CellTokenizer()
    if not tokenizer.has_temporal_tokens:
        sys.exit("token dict missing temporal tokens; pass full BioNeMo dict via --tokenizer-path")

    spec = load_spec(args.spec)
    exclude = {args.exclude_target} if args.exclude_target else set()

    if args.strategy == "random":
        chosen = pick_random(tokenizer, args.n, seed=args.seed, exclude=exclude)
    elif args.strategy == "unexpressed":
        import anndata as ad
        adata = ad.read_h5ad(args.h5ad)
        query_mask = _filter_query_cells(adata, spec)
        chosen = pick_unexpressed(adata, tokenizer, query_mask, args.n,
                                  seed=args.seed, exclude=exclude)
    elif args.strategy == "housekeeping":
        chosen = pick_housekeeping(tokenizer, args.n)
    else:
        sys.exit(f"unknown strategy: {args.strategy}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    serialized = [{"ensembl": g, "symbol": s, "strategy": args.strategy} for g, s in chosen]
    Path(args.out).write_text(json.dumps(serialized, indent=2))
    print(f"[pick] wrote {len(serialized)} controls to {args.out}")
    for r in serialized:
        print(f"  {r['ensembl']}  {r['symbol'] or ''}")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    controls = json.loads(Path(args.controls).read_text())
    if not controls:
        sys.exit(f"no controls in {args.controls}")

    base_cmd = args.target_cmd.strip()
    failed = []
    for i, ctrl in enumerate(controls):
        ensg = ctrl["ensembl"]
        out_dir = f"{args.out_dir_prefix}{ensg}"
        cmd = (f"{base_cmd} --gene {ensg} --out-dir {out_dir} "
               f"--wandb-name control_{ensg}")
        print(f"\n[run {i+1}/{len(controls)}] {ensg}")
        print(f"  $ {cmd}")
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            print(f"[run] FAILED: {ensg}")
            failed.append(ensg)
    if failed:
        print(f"\n[run] {len(failed)} controls failed: {failed}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------


def _load_run(out_dir: Path) -> dict:
    scores = np.load(out_dir / "scores.npz", allow_pickle=True)
    summary = json.loads((out_dir / "summary.json").read_text())
    return {
        "delta_t": np.asarray(scores["delta_t"]).ravel(),
        "abs_delta_t": np.abs(np.asarray(scores["delta_t"]).ravel()),
        "gene_present": np.asarray(scores["gene_present"]).astype(bool),
        "summary": summary,
    }


def cmd_compare(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    target = _load_run(Path(args.target_out_dir))
    target_label = f"{target['summary']['gene_ensembl']} (target)"

    controls = []
    for d in args.control_out_dirs:
        p = Path(d)
        if not (p / "scores.npz").exists():
            print(f"[compare] skip {p} (no scores.npz)")
            continue
        controls.append(_load_run(p))
    if not controls:
        sys.exit("no usable control runs")
    print(f"[compare] target={target['summary']['gene_ensembl']}, "
          f"controls={len(controls)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    null_delta = np.concatenate([c["delta_t"] for c in controls])
    null_abs = np.abs(null_delta)
    target_abs = target["abs_delta_t"]

    stat, p_mw = stats.mannwhitneyu(target_abs, null_abs, alternative="greater")

    target_mean = float(target_abs.mean())
    null_means = np.array([np.abs(c["delta_t"]).mean() for c in controls])
    rank = (null_means >= target_mean).sum() + 1
    empirical_p = rank / (len(null_means) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(target_abs.max(), null_abs.max(), 1e-3), 40)
    ax.hist(null_abs, bins=bins, alpha=0.5, color="C7",
            label=f"null (pooled controls, n={len(null_abs)})", density=True)
    ax.hist(target_abs, bins=bins, alpha=0.6, color="C3",
            label=f"{target_label} (n={len(target_abs)})", density=True)
    ax.axvline(np.median(null_abs), color="C7", linestyle="--", linewidth=1)
    ax.axvline(np.median(target_abs), color="C3", linestyle="--", linewidth=1)
    ax.set_xlabel("|Δt| = |perturbed - baseline|")
    ax.set_ylabel("density")
    sig = "***" if p_mw < 1e-3 else ("**" if p_mw < 0.01 else ("*" if p_mw < 0.05 else "ns"))
    ax.set_title(f"Per-cell |Δt|: target vs control null  ({sig})\n"
                 f"Mann-Whitney one-sided p = {p_mw:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(viz_dir / "target_vs_null_per_cell.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_means, bins=max(8, len(null_means) // 2 + 1),
            alpha=0.6, color="C7", label="control genes (mean |Δt|)")
    ax.axvline(target_mean, color="C3", linewidth=2,
               label=f"{target['summary']['gene_ensembl']} = {target_mean:.4f}")
    ax.set_xlabel("mean |Δt| across cells")
    ax.set_ylabel("number of genes")
    ax.set_title(f"Per-gene effect: target vs control panel\n"
                 f"empirical p = {empirical_p:.3f} "
                 f"(target ranked {rank}/{len(null_means) + 1})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(viz_dir / "target_vs_null_per_gene.png", dpi=150)
    plt.close(fig)

    stats_out = {
        "target_gene": target["summary"]["gene_ensembl"],
        "direction": target["summary"]["direction"],
        "n_target_cells": int(len(target_abs)),
        "n_control_genes": int(len(controls)),
        "n_pooled_control_cells": int(len(null_abs)),
        "target_mean_abs_delta_t": target_mean,
        "control_mean_abs_delta_t_mean": float(null_means.mean()),
        "control_mean_abs_delta_t_median": float(np.median(null_means)),
        "mannwhitney_p_one_sided": float(p_mw),
        "empirical_per_gene_p": float(empirical_p),
        "target_rank": int(rank),
        "control_genes": [c["summary"]["gene_ensembl"] for c in controls],
    }
    (out_dir / "comparison_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))

    if args.wandb_project and args.wandb_run_id:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="must",
            )
            for png in sorted(viz_dir.glob("*.png")):
                run.log({f"control/{png.stem}": wandb.Image(str(png))})
            run.summary.update({
                f"control/{k}": v for k, v in stats_out.items()
                if isinstance(v, (int, float))
            })
            art = wandb.Artifact(
                f"control_panel_{stats_out['target_gene']}_{stats_out['direction']}",
                type="control_comparison",
            )
            for fname in ("comparison_stats.json",):
                art.add_file(str(out_dir / fname))
            for png in viz_dir.glob("*.png"):
                art.add_file(str(png))
            run.log_artifact(art)
            run.finish()
            print(f"[compare] logged plots + artifact to wandb run {args.wandb_run_id}")
        except Exception as e:
            print(f"[compare] wandb attach skipped: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pick", help="select N control genes")
    pp.add_argument("--strategy", required=True,
                    choices=["random", "unexpressed", "housekeeping"])
    pp.add_argument("--n", type=int, default=8)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--spec", required=True, help="ExperimentSpec YAML/JSON")
    pp.add_argument("--tokenizer-path", required=True)
    pp.add_argument("--h5ad", help="required for strategy=unexpressed")
    pp.add_argument("--exclude-target", help="ENSG of your target gene; never picked as control")
    pp.add_argument("--out", required=True, help="output JSON listing controls")

    pr = sub.add_parser("run", help="invoke the runner once per control gene")
    pr.add_argument("--controls", required=True, help="JSON from `pick`")
    pr.add_argument("--target-cmd", required=True,
                    help="full base command without --gene / --out-dir")
    pr.add_argument("--out-dir-prefix", required=True,
                    help="each control's --out-dir is <prefix><ensembl>")
    pr.add_argument("--dry-run", action="store_true")

    pc = sub.add_parser("compare", help="aggregate target + controls into plots + stats")
    pc.add_argument("--target-out-dir", required=True)
    pc.add_argument("--control-out-dirs", nargs="+", required=True)
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--wandb-project", default=None,
                    help="if set with --wandb-run-id, appends to the target's wandb run")
    pc.add_argument("--wandb-run-id", default=None)
    pc.add_argument("--wandb-entity", default=None)

    args = p.parse_args()
    if args.cmd == "pick":
        cmd_pick(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()