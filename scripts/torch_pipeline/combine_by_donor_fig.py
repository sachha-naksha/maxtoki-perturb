"""Combined per-donor Δt figure across the 6 8k evenly-spaced YM2-context runs.
Layout: 12 horizontal bars, OM9 (top half, more cells) above OM6 (bottom half).
Each bar labeled with gene + i/oe. SEM as error bars. Output: editable SVG.
"""
from __future__ import annotations

from pathlib import Path

import datasets  # noqa: F401  (used implicitly via load_from_disk)
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

plt.rcParams.update({
    "text.usetex": False,
    "svg.fonttype": "none",  # keep text as <text> elements (editable in Inkscape/AI)
})

FIGURE_PARAMS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "format": "svg",
    "transparent": True,
}

OUT_ROOT = Path("/workspaces/maxToki/out")

# Order matches mean_delta_t_8k_evenly.csv.
RUNS = [
    ("PDK4 i",   "pdk4_217m_inhibit_evenly_seq8k",   "inhibit"),
    ("PDK4 oe",  "pdk4_217m_overexpress_evenly_seq8k", "overexpress"),
    ("IRS2 i",   "irs2_217m_inhibit_evenly_seq8k",   "inhibit"),
    ("IRS2 oe",  "irs2_217m_overexpress_evenly_seq8k", "overexpress"),
    ("NR4A3 i",  "nr4a3_217m_inhibit_evenly_seq8k",  "inhibit"),
    ("NR4A3 oe", "nr4a3_217m_overexpress_evenly_seq8k", "overexpress"),
]

DONOR_ORDER = ["OM9", "OM6"]  # OM9 on top (more cells), OM6 on bottom
COLORS = {"inhibit": "#c0392b", "overexpress": "#2c7fb8"}


def per_donor_stats(run_dir: Path) -> dict[str, tuple[float, float, int]]:
    z = np.load(run_dir / "scores.npz", allow_pickle=True)
    delta_t = np.asarray(z["delta_t"]).ravel()
    ds = load_from_disk(str(run_dir / "baseline.dataset"))
    group = np.array([str(r["group"]) for r in ds])
    out: dict[str, tuple[float, float, int]] = {}
    for donor in DONOR_ORDER:
        sel = group == donor
        n = int(sel.sum())
        if n == 0:
            out[donor] = (float("nan"), float("nan"), 0)
            continue
        vals = delta_t[sel].astype(float)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[donor] = (mean, sem, n)
    return out


def main() -> None:
    rows = []  # (label, donor, mean, sem, n, direction)
    for label, run_subdir, direction in RUNS:
        stats = per_donor_stats(OUT_ROOT / run_subdir)
        for donor in DONOR_ORDER:
            mean, sem, n = stats[donor]
            rows.append((label, donor, mean, sem, n, direction))

    # Y positions: OM9 block (top), gap, OM6 block (bottom). Index 0 is bottom
    # in matplotlib, so build positions from the bottom up.
    n_per_donor = len(RUNS)
    gap = 1.2
    om6_ys = list(range(n_per_donor))
    om9_ys = [y + n_per_donor + gap for y in range(n_per_donor)]

    # Within each block the first RUN ("PDK4 i") goes on top.
    om6_ys = list(reversed(om6_ys))
    om9_ys = list(reversed(om9_ys))

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    bar_h = 0.45

    yticks: list[float] = []
    ylabels: list[str] = []

    for i, (label, _run, _dir) in enumerate(RUNS):
        for donor, ys in (("OM9", om9_ys), ("OM6", om6_ys)):
            y = ys[i]
            row = next(r for r in rows if r[0] == label and r[1] == donor)
            _, _, mean, sem, n, direction = row
            ax.barh(
                y, mean, height=bar_h, color=COLORS[direction],
                edgecolor="black", linewidth=0.4,
                xerr=sem, ecolor="black",
                error_kw={"elinewidth": 0.7, "capsize": 2},
            )
            yticks.append(y)
            ylabels.append(f"{label}  (n={n})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)

    # Donor block headers on the right side.
    om9_mid = (max(om9_ys) + min(om9_ys)) / 2
    om6_mid = (max(om6_ys) + min(om6_ys)) / 2
    ax.text(
        1.02, om9_mid, "OM9", transform=ax.get_yaxis_transform(),
        fontsize=12, fontweight="bold", va="center", ha="left",
    )
    ax.text(
        1.02, om6_mid, "OM6", transform=ax.get_yaxis_transform(),
        fontsize=12, fontweight="bold", va="center", ha="left",
    )

    # Visual divider between donor blocks.
    divider_y = (max(om6_ys) + min(om9_ys)) / 2
    ax.axhline(divider_y, color="grey", linewidth=0.6, linestyle="--", alpha=0.6)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean Δt  (perturbed − baseline pseudotime)")
    ax.set_title(
        "Zero-shot perturbation Δt by donor — 217M, 8k seq, YM2 evenly-spaced ctx",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)

    # Legend for direction colors.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["inhibit"], label="inhibit"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["overexpress"], label="overexpress"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_svg = OUT_ROOT / "combined_by_donor_8k_evenly.svg"
    fig.savefig(out_svg, **FIGURE_PARAMS)
    out_png = OUT_ROOT / "combined_by_donor_8k_evenly.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
