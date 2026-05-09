"""Aggregate 8k vs 16k context-length sweep into a single comparison CSV.

Walks the six PDK4 inhibit run dirs (3 sampling strategies x 2 seq lengths),
pulls headline metrics from summary.json + viz_stats.json + prep_summary.json,
and splits each row's input_ids on the <bos>/<eos> boundaries to recover
per-cell token lengths (3 context cells + 1 query cell).

Writes ./out/comparison_8k_vs_16k.csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import CellTokenizer  # noqa: E402

OUT_DIR = Path("out")
RUNS = [
    ("evenly",  16384, "pdk4_217m_inhibit_evenly_seq16k"),
    ("evenly",   8192, "pdk4_217m_inhibit_evenly_seq8k"),
    ("first",   16384, "pdk4_217m_inhibit_first_seq16k"),
    ("first",    8192, "pdk4_217m_inhibit_first_seq8k"),
    ("oldest3", 16384, "pdk4_217m_inhibit_oldest3_seq16k"),
    ("oldest3",  8192, "pdk4_217m_inhibit_oldest3_seq8k"),
]


def _split_row(input_ids: list[int], bos: int, eos: int, boq: int) -> tuple[list[int], int]:
    """Return (context_cell_lengths, query_length) for one input_ids sequence.

    Layout produced by dataset_prep._build_input_ids:
        [<bos> ctx1 <eos>] [<bos> ctx2 <eos>] [<bos> ctx3 <eos>] <boq> query <eoq> <dummy>
    """
    ctx_lens: list[int] = []
    i = 0
    while i < len(input_ids) and input_ids[i] == bos:
        # find matching <eos>
        j = i + 1
        while j < len(input_ids) and input_ids[j] != eos:
            j += 1
        ctx_lens.append(j - i + 1)  # include both bos and eos
        i = j + 1
    # i should now point at <boq>
    if i >= len(input_ids) or input_ids[i] != boq:
        raise ValueError(f"expected <boq> at position {i}, got token {input_ids[i] if i < len(input_ids) else 'EOF'}")
    # remainder is <boq> query <eoq> <dummy>
    query_len = len(input_ids) - i
    return ctx_lens, query_len


def main() -> None:
    tok_path = os.environ.get(
        "MAXTOKI_TOKEN_DICT_217M",
        "/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json",
    )
    tokenizer = CellTokenizer(tok_path)
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    boq = tokenizer.boq_id

    rows = []
    for sampling, seq_len, run_dir_name in RUNS:
        run_dir = OUT_DIR / run_dir_name
        if not run_dir.exists():
            print(f"[skip] {run_dir} missing")
            continue

        summary = json.loads((run_dir / "summary.json").read_text())
        prep = json.loads((run_dir / "prep_summary.json").read_text())
        viz_stats = {}
        viz_path = run_dir / "viz" / "viz_stats.json"
        if viz_path.exists():
            viz_stats = json.loads(viz_path.read_text())

        ds = datasets.load_from_disk(str(run_dir / "baseline.dataset"))
        # context cells are constant across rows; verify on row 0 + row -1
        ctx_lens_0, _ = _split_row(list(ds[0]["input_ids"]), bos, eos, boq)
        ctx_lens_n, _ = _split_row(list(ds[len(ds) - 1]["input_ids"]), bos, eos, boq)
        assert ctx_lens_0 == ctx_lens_n, f"context lengths drift between rows: {ctx_lens_0} vs {ctx_lens_n}"
        ctx_cell_ids = list(ds[0]["context_cell_ids"])

        # collect per-row query lengths
        query_lens = []
        for r in ds:
            _, ql = _split_row(list(r["input_ids"]), bos, eos, boq)
            query_lens.append(ql)
        q = np.asarray(query_lens)

        rows.append({
            "sampling": sampling,
            "seq_length": seq_len,
            "run_dir": run_dir_name,
            "n_query_total": summary["n_rows"],
            "n_query_with_PDK4": summary["n_rows_with_gene_in_query"],
            "mean_delta_t_overall": round(float(summary["mean_delta_t"]), 3),
            "mean_delta_t_PDK4_present": round(float(summary["mean_delta_t_present"]), 3),
            "abs_mean_delta_t": round(float(summary["abs_mean_delta_t"]), 3),
            "mean_mse_overall": round(float(summary["mean_mse"]), 3),
            "mean_mse_PDK4_present": round(float(summary["mean_mse_present"]), 3),
            "pearson_r_pseudotime": round(float(viz_stats.get("pearson_r", float("nan"))), 4),
            "wilcoxon_p_present_vs_zero": viz_stats.get("wilcoxon_present_vs_zero_p", float("nan")),
            "ctx1_cell_id": ctx_cell_ids[0],
            "ctx1_tokens":  ctx_lens_0[0],
            "ctx2_cell_id": ctx_cell_ids[1],
            "ctx2_tokens":  ctx_lens_0[1],
            "ctx3_cell_id": ctx_cell_ids[2],
            "ctx3_tokens":  ctx_lens_0[2],
            "query_tokens_min":    int(q.min()),
            "query_tokens_median": int(np.median(q)),
            "query_tokens_mean":   round(float(q.mean()), 1),
            "query_tokens_max":    int(q.max()),
            "max_input_len_baseline": prep["max_input_len_baseline"],
        })

    if not rows:
        sys.exit("no runs found")

    out_csv = OUT_DIR / "comparison_8k_vs_16k.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # also print as a quick markdown peek
    print("\n=== headline ===")
    for r in rows:
        print(
            f"{r['sampling']:8s} seq={r['seq_length']:>5d}  "
            f"Δt(all)={r['mean_delta_t_overall']:+7.2f}  "
            f"Δt(PDK4+)={r['mean_delta_t_PDK4_present']:+7.2f}  "
            f"n_PDK4+={r['n_query_with_PDK4']:>4d}  "
            f"ctx_tok=[{r['ctx1_tokens']},{r['ctx2_tokens']},{r['ctx3_tokens']}]  "
            f"query_tok median={r['query_tokens_median']:>4d} max={r['query_tokens_max']:>5d}"
        )


if __name__ == "__main__":
    main()
