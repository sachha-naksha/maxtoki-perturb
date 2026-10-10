"""Δt comparison for the token-dictionary / untrained-row tests (delta/slurm/_run_tbc_dict_test.sh).

Usage: python3 delta/scripts/plot_tbc_dict_test.py <jobid_ABC> <jobid_BD_clean>
Writes delta/figs/tbc_dict_test_<jobid>.{png,svg,csv}.

Left:  mean Δt (perturbed − baseline) ± SEM over the 2000 OM queries, same layout as
       out/combined_aggregated_8k_evenly_3gene.png, for the original May run and runs A–D.
Right: per-cell *baseline* Δt prediction, old dictionary vs. old dictionary with the
       untrained rows re-drawn (A vs B). If the readout came from trained weights the
       points would sit on the diagonal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out"
FIGS = REPO / "delta" / "figs"
BLUE, RED, GRAY = "#1f5fbf", "#c8451f", "#8a8984"   # validated pair (dataviz validator, light surface)


def load(run_dir: Path):
    z = np.load(run_dir / "scores.npz")
    return z["baseline"].astype(float), z["perturbed"].astype(float), z["delta_t"].astype(float)


def main(job_abc: str, job_bd: str):
    runs = [
        ("original (May, old dict)", OUT / "pdk4_217m_inhibit_evenly_seq8k", GRAY),
        ("A  old dict (rerun)", OUT / f"tbc_dict_test_A_olddict_delta_{job_abc}", RED),
        ("B  old dict, untrained rows re-drawn", OUT / f"tbc_dict_test_B_olddict_rr1_delta_{job_bd}", RED),
        ("C  aligned dict", OUT / f"tbc_dict_test_C_aligned_delta_{job_abc}", BLUE),
        ("D  aligned dict, untrained rows re-drawn", OUT / f"tbc_dict_test_D_aligned_rr1_delta_{job_bd}", BLUE),
    ]
    data = {lab: load(p) for lab, p, _ in runs}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ys = np.arange(len(runs))[::-1]
    rows = []
    for y, (lab, _, col) in zip(ys, runs):
        _, _, dt = data[lab]
        m, sem = dt.mean(), dt.std(ddof=1) / np.sqrt(len(dt))
        rows.append((lab, m, sem, len(dt)))
        ax.barh(y, m, height=0.6, color=col, xerr=sem, ecolor="black", capsize=3, linewidth=0)
        ax.text(min(m, 0) - 25, y, f"{m:,.0f}", va="center", ha="right", fontsize=9, color="#0b0b0b")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean Δt  (perturbed − baseline pseudotime), n = 2000 OM cells")
    ax.set_title("PDK4 inhibit, 217M zero-shot TBC — same spec,\nfour weight/dictionary variants", fontsize=10)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xmin = min(r[1] - r[2] for r in rows)
    ax.set_xlim(xmin * 1.25, max(60, -xmin * 0.05))

    a_base = data["A  old dict (rerun)"][0]
    b_base = data["B  old dict, untrained rows re-drawn"][0]
    r = np.corrcoef(a_base, b_base)[0, 1]
    ax2.scatter(a_base, b_base, s=6, alpha=0.35, color=RED, linewidths=0)
    lo, hi = min(a_base.min(), b_base.min()), max(a_base.max(), b_base.max())
    ax2.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("A: baseline Δt prediction, old dict")
    ax2.set_ylabel("B: same, untrained rows re-drawn")
    ax2.set_title(f"Per-cell baseline prediction, A vs B  (r = {r:.2f})", fontsize=10)
    ax2.text(0.02, 0.03, "true Δt_q for these cells: −67 … 0", transform=ax2.transAxes,
             fontsize=8, color="#52514e")
    ax2.grid(color="#e5e4e0", linewidth=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    FIGS.mkdir(exist_ok=True)
    stem = FIGS / f"tbc_dict_test_{job_bd}"
    fig.tight_layout(w_pad=3.0)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    with open(stem.with_suffix(".csv"), "w") as f:
        f.write("run,mean_delta_t,sem,n\n")
        for lab, m, sem, n in rows:
            f.write(f"{lab},{m:.4f},{sem:.4f},{n}\n")
    print(f"wrote {stem}.png/.svg/.csv")
    for lab, m, sem, n in rows:
        print(f"  {lab:42s} mean {m:9.1f}  sem {sem:5.1f}")
    print(f"  corr(baseline A, baseline B) = {r:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
