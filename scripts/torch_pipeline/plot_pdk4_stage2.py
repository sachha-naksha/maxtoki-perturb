"""Summarize paired gene-inhibition predictions with donor-conditional cell CIs.

CLI execution, plotting and tests belong in Slurm compute allocations.
New runs contain rows.json (schema below) and optional run.json. Historical
runs use scores.npz plus baseline.dataset, preserving the original estimates.
Intervals quantify cell sampling within the observed donors, not uncertainty
across donors, checkpoint training runs, or biological interventions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

DONORS = ("OM6", "OM9")
ROW_FIELDS = ("row", "query_cell_id", "donor", "gene_present",
              "control_prediction", "perturbed_prediction")
CI_METHOD = ("95% percentile bootstrap of query-cell clusters within each donor; "
             "donor row weights fixed; conditional on these observed donors")
CAVEAT = ("Predicted prompt-edit response, not measured perturbation biology. "
          "OM6 was used in Stage 2 training; OM9 was used for validation and "
          "checkpoint selection. Historical output used a different pipeline "
          "and is a descriptive reference.")


def compute():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use a Slurm compute allocation")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows):
    if not rows:
        raise ValueError("Empty paired predictions")
    out = []
    seen_rows = set()
    cell_metadata = {}
    for index, original in enumerate(rows):
        missing = set(ROW_FIELDS) - set(original)
        if missing:
            raise ValueError(f"Missing paired row fields: {sorted(missing)}")
        row = dict(original)
        if row["row"] in seen_rows:
            raise ValueError("Duplicate paired row ID")
        seen_rows.add(row["row"])
        if row["donor"] not in DONORS:
            raise ValueError(f"Unexpected query donor: {row['donor']}")
        if not isinstance(row["gene_present"], (bool, np.bool_)):
            raise ValueError("gene_present must be boolean")
        if not str(row["query_cell_id"]):
            raise ValueError("Query cell ID is empty")
        identity = str(row["query_cell_id"])
        metadata = (row["donor"], bool(row["gene_present"]))
        if identity in cell_metadata and cell_metadata[identity] != metadata:
            raise ValueError("Conflicting metadata for repeated query cell")
        cell_metadata[identity] = metadata
        for field in ("control_prediction", "perturbed_prediction"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError("Nonfinite paired prediction")
        row["query_cell_id"] = identity
        row["gene_present"] = bool(row["gene_present"])
        row["delta_t"] = row["perturbed_prediction"] - row["control_prediction"]
        if "delta_t" in original and not math.isclose(
                float(original["delta_t"]), row["delta_t"], rel_tol=1e-6, abs_tol=5e-4):
            raise ValueError("Stored delta differs from paired predictions")
        if "target_delta_pseudotime" in row and row["target_delta_pseudotime"] is not None:
            row["target_delta_pseudotime"] = float(row["target_delta_pseudotime"])
            if not math.isfinite(row["target_delta_pseudotime"]):
                raise ValueError("Nonfinite target interval")
        out.append(row)
    return out


def read_run(path, historical=False):
    path = Path(path)
    if historical:
        from datasets import load_from_disk
        data = load_from_disk(str(path/"baseline.dataset"))
        with np.load(path/"scores.npz", allow_pickle=False) as scores:
            base = np.asarray(scores["baseline"], dtype=float).ravel()
            pert = np.asarray(scores["perturbed"], dtype=float).ravel()
            delta = np.asarray(scores["delta_t"], dtype=float).ravel()
            present = np.asarray(scores["gene_present"], dtype=bool).ravel()
        if not len(data) == len(base) == len(pert) == len(delta) == len(present):
            raise ValueError("Historical predictions and metadata have different lengths")
        rows = []
        for i, metadata in enumerate(data):
            if str(metadata["group"]) not in DONORS:
                raise ValueError("Historical dataset contains unexpected donors")
            if bool(metadata["gene_present_in_query"]) != bool(present[i]):
                raise ValueError("Historical gene presence metadata mismatch")
            row = dict(row=i, query_cell_id=str(metadata["cell_id"]),
                       donor=str(metadata["group"]), gene_present=bool(present[i]),
                       control_prediction=base[i], perturbed_prediction=pert[i],
                       delta_t=delta[i], context_cell_ids=metadata["context_cell_ids"])
            if metadata.get("query_pseudotime") is not None:
                row["target_delta_pseudotime"] = (
                    float(metadata["query_pseudotime"]) -
                    float(metadata["context_pseudotimes"][-1]))
            rows.append(row)
        provenance = dict(historical=True, directory=str(path.resolve()),
                          scores_sha256=digest(path/"scores.npz"),
                          protocol="Original stored output; different vocabulary/runtime")
        for filename in ("summary.json", "spec.resolved.json"):
            if (path/filename).exists():
                provenance[filename] = json.loads((path/filename).read_text())
    else:
        row_path = path/"rows.json"
        payload = json.loads(row_path.read_text())
        rows = payload["rows"] if isinstance(payload, dict) else payload
        provenance = dict(historical=False, directory=str(path.resolve()),
                          rows_sha256=digest(row_path))
        if (path/"run.json").exists():
            provenance["run"] = json.loads((path/"run.json").read_text())
    return validate_rows(rows), provenance


def bootstrap_ci(rows, *, n_bootstrap=2000, seed=42):
    """Resample whole query clusters, stratified by donor, preserving pairs."""
    if n_bootstrap < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    if not rows:
        return None, None
    rng = np.random.default_rng(seed)
    by_donor = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_donor[row["donor"]][row["query_cell_id"]].append(row["delta_t"])
    replicate = np.zeros(n_bootstrap, dtype=np.float64)
    for donor, clusters in sorted(by_donor.items()):
        ordered = [values for _, values in sorted(clusters.items())]
        sums = np.array([sum(values) for values in ordered], dtype=float)
        counts = np.array([len(values) for values in ordered], dtype=float)
        weight = counts.sum()/len(rows)
        for start in range(0, n_bootstrap, 128):
            stop = min(start+128, n_bootstrap)
            selected = rng.integers(0, len(ordered), size=(stop-start, len(ordered)))
            replicate[start:stop] += weight * (
                sums[selected].sum(axis=1)/counts[selected].sum(axis=1))
    lo, hi = np.quantile(replicate, [.025, .975])
    return float(lo), float(hi)


def summarize(rows, *, n_bootstrap=2000, seed=42):
    n = len(rows)
    if not n:
        return dict(n_rows=0, n_unique_queries=0, n_gene_present=0,
                    mean_delta_t=None, ci95_low=None, ci95_high=None)
    delta = np.array([row["delta_t"] for row in rows], dtype=float)
    control = np.array([row["control_prediction"] for row in rows], dtype=float)
    perturbed = np.array([row["perturbed_prediction"] for row in rows], dtype=float)
    lo, hi = bootstrap_ci(rows, n_bootstrap=n_bootstrap, seed=seed)
    absent = [row["delta_t"] for row in rows if not row["gene_present"]]
    result = dict(n_rows=n, n_unique_queries=len({r["query_cell_id"] for r in rows}),
                  n_gene_present=sum(r["gene_present"] for r in rows),
                  mean_delta_t=float(delta.mean()), median_delta_t=float(np.median(delta)),
                  mean_absolute_delta_t=float(np.abs(delta).mean()),
                  ci95_low=lo, ci95_high=hi,
                  fraction_negative=float((delta < 0).mean()),
                  fraction_positive=float((delta > 0).mean()),
                  fraction_zero=float((delta == 0).mean()),
                  gene_absent_max_absolute_delta=(max(map(abs, absent)) if absent else None),
                  control_prediction_mean=float(control.mean()),
                  control_prediction_std=float(control.std(ddof=0)),
                  control_prediction_min=float(control.min()),
                  control_prediction_max=float(control.max()),
                  perturbed_prediction_mean=float(perturbed.mean()),
                  perturbed_prediction_std=float(perturbed.std(ddof=0)),
                  perturbed_prediction_min=float(perturbed.min()),
                  perturbed_prediction_max=float(perturbed.max()),
                  standard_deviation_convention="population; ddof=0")
    target_rows = [r for r in rows if r.get("target_delta_pseudotime") is not None]
    if target_rows:
        errors = np.array([r["control_prediction"]-r["target_delta_pseudotime"]
                           for r in target_rows])
        targets = np.array([r["target_delta_pseudotime"] for r in target_rows])
        result["target_delta_pseudotime_mean"] = float(targets.mean())
        result["target_delta_pseudotime_std"] = float(targets.std(ddof=0))
        result["target_delta_pseudotime_min"] = float(targets.min())
        result["target_delta_pseudotime_max"] = float(targets.max())
        result["control_ground_truth"] = dict(
            n=len(target_rows), mae=float(np.abs(errors).mean()),
            mean_error=float(errors.mean()), rmse=float(np.sqrt((errors**2).mean())))
    return result


def summarize_run(rows, *, n_bootstrap=2000, seed=42):
    result = {}
    for donor in ("pooled", *DONORS):
        selected = rows if donor == "pooled" else [r for r in rows if r["donor"] == donor]
        result[donor] = {}
        for group in ("all", "gene_present", "gene_absent"):
            subset = selected if group == "all" else [
                r for r in selected if r["gene_present"] == (group == "gene_present")]
            result[donor][group] = summarize(subset, n_bootstrap=n_bootstrap, seed=seed)
    return result


def _cell_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["donor"], row["query_cell_id"])].append(row["delta_t"])
    return {key:float(np.mean(values)) for key, values in grouped.items()}


def compare_runs(reference_rows, current_rows):
    """Descriptive agreement by unique query cell; does not assert matched prompts."""
    ref, cur = _cell_means(reference_rows), _cell_means(current_rows)
    common = sorted(ref.keys() & cur.keys())
    if not common:
        return dict(n_matched_unique_queries=0)
    left = np.array([ref[key] for key in common])
    right = np.array([cur[key] for key in common])
    nonzero = (np.abs(left) > 1e-6) & (np.abs(right) > 1e-6)
    variable = len(common)>1 and left.std()>0 and right.std()>0
    if variable:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(left,right).statistic)
    else:
        spearman = None
    same_context = {}
    for tag, rows in (("reference", reference_rows), ("current", current_rows)):
        by_cell = defaultdict(set)
        for row in rows:
            if "context_cell_ids" in row:
                by_cell[(row["donor"], row["query_cell_id"])].add(
                    tuple(map(str, row["context_cell_ids"])))
        same_context[tag] = by_cell
    equal_context = sum(
        bool(same_context["reference"].get(key)) and
        same_context["reference"].get(key) == same_context["current"].get(key)
        for key in common)
    return dict(n_matched_unique_queries=len(common),
                n_cells_with_identical_context_id_sets=int(equal_context),
                n_nonzero_both=int(nonzero.sum()),
                nonzero_direction_agreement=(
                    float((np.sign(left[nonzero]) == np.sign(right[nonzero])).mean())
                    if nonzero.any() else None),
                delta_pearson=(float(np.corrcoef(left,right)[0,1]) if variable else None),
                delta_spearman=spearman,
                delta_mae=float(np.abs(right-left).mean()),
                mean_current_minus_reference=float((right-left).mean()),
                interpretation="Descriptive same-cell comparison; prompts/runtime may differ")


def _bar_panel(ax, entries, title, colors):
    ys = list(range(len(entries)))[::-1]
    labels = []
    for y, (label, values), color in zip(ys, entries, colors):
        mean = values["mean_delta_t"]
        labels.append(f"{label}\n(n={values['n_rows']}, cells={values['n_unique_queries']})")
        if mean is None:
            ax.text(0, y, "No eligible cells", va="center", fontsize=9)
            continue
        ax.barh(y, mean, height=.55, color=color, edgecolor="black", linewidth=.45)
        # Draw percentile endpoints directly: for small/skewed samples the
        # percentile interval need not enclose the observed point estimate.
        ax.hlines(y, values["ci95_low"], values["ci95_high"], color="black", lw=1.2)
        ax.vlines([values["ci95_low"], values["ci95_high"]], y-.055,y+.055,
                  color="black", lw=1.2)
        ax.scatter([mean], [y], color="black", s=12, zorder=3)
    ax.set_yticks(ys, labels)
    ax.axvline(0, color="black", lw=.8)
    ax.grid(axis="x", alpha=.22)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Mean predicted delta t\n(inhibited - control)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(y=.22)


def render(report, output, symbol="PDK4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42, "svg.fonttype":"none"})
    labels = [label for label,run in report["runs"].items()
              if not run["provenance"]["historical"]]
    palette = ["#c0392b", "#8b1a1a", "#e57f73"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    titles = {"pooled":"OM6 + OM9 pooled",
              "OM6":"OM6 (Stage 2 training donor)",
              "OM9":"OM9 (validation donor)"}
    prefix = symbol.lower()
    for name, group in ((f"{prefix}_inhibition_tbc","all"),
                        (f"{prefix}_inhibition_tbc_gene_present","gene_present"),
                        (f"{prefix}_inhibition_tbc_gene_absent","gene_absent")):
        fig, axes = plt.subplots(1,3,figsize=(17, max(5.2,1.0*len(labels)+3.0)))
        limits = [0.]
        for ax, donor in zip(axes, ("pooled", *DONORS)):
            entries = [(label,report["runs"][label]["statistics"][donor][group])
                       for label in labels]
            _bar_panel(ax, entries, titles[donor], colors)
            for _,values in entries:
                limits.extend(values[key] for key in ("ci95_low","ci95_high","mean_delta_t")
                              if values[key] is not None)
        low,high=min(limits),max(limits)
        padding=.1*(high-low) if high>low else .01
        for ax in axes:
            ax.set_xlim(low-padding,high+padding)
        subtitle = {"all":"All query cells", "gene_present":f"{symbol} present before editing",
                    "gene_absent":f"{symbol} absent before editing (unchanged-prompt check)"}[group]
        fig.suptitle(f"Stage 2 {symbol} inhibition TBC | "+subtitle+"\n"
                     "Whiskers: 95% query-cell bootstrap CI; conditional on these donors",
                     fontsize=13)
        fig.text(.5,.01,"Computational prompt edits; YM2 evenly spaced contexts; "
                 "8k inference, 16k-trained RoPE 4. "
                 "OM9 informed checkpoint selection. Units: pseudotime.",
                 ha="center",fontsize=9)
        # Explicit spacing reserves room for checkpoint names and two-line
        # count labels beside each panel, including historical-length labels.
        fig.subplots_adjust(left=.145,right=.985,bottom=.23,top=.75,wspace=.85)
        for ext in ("png","pdf","svg"):
            fig.savefig(output/f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    if report.get("historical_scaled"):
        historical = report["historical_scaled"]
        raw_label = historical["raw_label"]
        raw = report["runs"][raw_label]["statistics"]["pooled"]["all"]
        scaled = historical["statistics"]["pooled"]["all"]
        fig, axes = plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={"width_ratios":[1,1.6]})
        _bar_panel(axes[0],[(raw_label,raw)],"Archived raw readout (separate scale)",["#888888"])
        axes[0].set_xlabel(f"Archived delta t\n(numeric expectation x {historical['divisor']:g})")
        entries = [(f"{raw_label} / {historical['divisor']:g}",scaled)] + [
            (label,report["runs"][label]["statistics"]["pooled"]["all"]) for label in labels]
        _bar_panel(axes[1],entries,"Numeric-expectation readout scale",["#888888",*colors])
        axes[1].set_xlabel("Mean delta numeric expectation\n(inhibited - control)")
        fig.suptitle(f"{symbol} inhibition | Historical versus Stage 2 predictions\n"
                     "OM6 + OM9 pooled; whiskers: 95% query-cell bootstrap CI",fontsize=13)
        fig.text(.5,.015,f"Historical / {historical['divisor']:g} removes the archived output multiplier only. "
                 "Vocabulary, prompts and training differ; this is not an identical-protocol rerun.",
                 ha="center",fontsize=9)
        fig.subplots_adjust(left=.18,right=.985,bottom=.21,top=.77,wspace=.95)
        for ext in ("png","pdf","svg"):
            fig.savefig(output/f"{prefix}_historical_comparison.{ext}",dpi=180,bbox_inches="tight")
        plt.close(fig)


def write_csv(report, path):
    fields = ("run","donor","subgroup","n_rows","n_unique_queries","n_gene_present",
              "mean_delta_t","ci95_low","ci95_high","median_delta_t",
              "mean_absolute_delta_t","fraction_negative","fraction_positive",
              "fraction_zero","gene_absent_max_absolute_delta",
              "control_prediction_mean","control_prediction_std",
              "control_prediction_min","control_prediction_max",
              "perturbed_prediction_mean","perturbed_prediction_std",
              "perturbed_prediction_min","perturbed_prediction_max",
              "target_delta_pseudotime_mean","target_delta_pseudotime_std",
              "target_delta_pseudotime_min","target_delta_pseudotime_max")
    with Path(path).open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for label, run in report["runs"].items():
            for donor, groups in run["statistics"].items():
                for subgroup, values in groups.items():
                    writer.writerow(dict(run=label,donor=donor,subgroup=subgroup,**values))
        if report.get("historical_scaled"):
            scaled=report["historical_scaled"]
            for donor,groups in scaled["statistics"].items():
                for subgroup,values in groups.items():
                    writer.writerow(dict(run=f"{scaled['raw_label']} / {scaled['divisor']:g}",
                                         donor=donor,subgroup=subgroup,**values))


def ensure_matched_runs(previous, current):
    """Stage 2 checkpoints must have identical ordered query pairs."""
    if len(previous) != len(current):
        raise ValueError("Stage 2 runs have different numbers of query pairs")
    for left,right in zip(previous,current):
        for field in ("row","query_cell_id","donor","gene_present"):
            if left[field] != right[field]:
                raise ValueError(f"Stage 2 query-pair alignment differs: {field}")
        for field in ("context_cell_ids","target_delta_pseudotime"):
            if field in left and field in right and left[field] != right[field]:
                raise ValueError(f"Stage 2 paired metadata differs: {field}")


def build_report(runs, *, n_bootstrap=2000, seed=42, historical_scalar=200.):
    if not runs:
        raise ValueError("At least one run is required")
    if not math.isfinite(historical_scalar) or historical_scalar <= 0:
        raise ValueError("Historical scalar must be finite and positive")
    if len({label for label,_,_ in runs}) != len(runs):
        raise ValueError("Run labels must be unique")
    result = dict(protocol="pdk4_paired_tbc_summary_v1", units="pseudotime",
                  delta_definition="predicted perturbed interval - predicted control interval",
                  ci_method=CI_METHOD, n_bootstrap=n_bootstrap, seed=seed,
                  caveat=CAVEAT, runs={}, descriptive_comparisons={})
    all_rows = {}
    first_current = None
    for label, rows, provenance in runs:
        rows = validate_rows(rows)
        if not provenance["historical"]:
            if first_current is None:
                first_current = rows
            else:
                ensure_matched_runs(first_current,rows)
        all_rows[label] = rows
        result["runs"][label] = dict(provenance=provenance,
            statistics=summarize_run(rows,n_bootstrap=n_bootstrap,seed=seed))
        if provenance["historical"]:
            if "historical_scaled" in result:
                raise ValueError("Only one historical run is supported")
            scaled_rows = [dict(row,control_prediction=row["control_prediction"]/historical_scalar,
                                perturbed_prediction=row["perturbed_prediction"]/historical_scalar,
                                delta_t=row["delta_t"]/historical_scalar) for row in rows]
            result["historical_scaled"] = dict(raw_label=label,divisor=historical_scalar,
                interpretation="Removes output multiplier only; not corrected vocabulary or prompts",
                statistics=summarize_run(scaled_rows,n_bootstrap=n_bootstrap,seed=seed))
            for future_label, future_rows, future_provenance in runs:
                if not future_provenance["historical"]:
                    result["descriptive_comparisons"][f"{future_label} vs {label} / {historical_scalar:g}"] = (
                        compare_runs(scaled_rows,validate_rows(future_rows)))
    labels = list(all_rows)
    for index,reference in enumerate(labels):
        for label in labels[index+1:]:
            result["descriptive_comparisons"][f"{label} vs {reference}"] = (
                compare_runs(all_rows[reference], all_rows[label]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--historical-label",default="Historical original")
    parser.add_argument("--historical-scalar",type=float,default=200.)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--n-bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--gene-symbol",default="PDK4")
    args=parser.parse_args()
    compute()
    if args.output.exists():
        raise FileExistsError("Choose a fresh plot output directory")
    specs = []
    if args.historical:
        specs.append((args.historical_label,args.historical,True))
    for item in args.run:
        if "=" not in item:
            parser.error("--run must be LABEL=DIR")
        label,path=item.split("=",1)
        specs.append((label,Path(path),False))
    if not args.run:
        parser.error("At least one Stage 2 --run is required")
    runs = [(label,*read_run(path,historical=old)) for label,path,old in specs]
    report = build_report(runs,n_bootstrap=args.n_bootstrap,seed=args.seed,
                          historical_scalar=args.historical_scalar)
    args.output.mkdir(parents=True)
    dump(args.output/"summary.json",report)
    write_csv(report,args.output/"bar_statistics.csv")
    render(report,args.output,symbol=args.gene_symbol)
    print(json.dumps({label:run["statistics"]["pooled"]["all"]
                      for label,run in report["runs"].items()},indent=2))


if __name__ == "__main__":
    main()
