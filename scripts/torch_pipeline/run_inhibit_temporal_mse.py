"""End-to-end zero-shot gene-inhibition temporal-MSE driver.

Three stages:
    1. Tokenize the 4111-cell h5ad and emit two HF datasets to disk
       (baseline and perturbed). See dataset_prep.build_paired_dataset.
    2. Run the upstream BioNeMo headless predict step on each dataset,
       producing predictions__rank_*.pt files (regression_preds at <eoq>).
    3. Read both prediction dirs, compute per-cell delta_t and dataset MSE.

Stage 2 is gated on bionemo + nemo + megatron being importable (i.e. you're
running inside the BioNeMo container on Delta). Stages 1 and 3 are pure
numpy / torch and can be run anywhere.

Variants: pass --variant 217m or 1b. The Megatron model architecture is
read out of the distcp checkpoint, so the only thing that changes between
variants is --ckpt-dir and the recommended --micro-batch-size. Default
seq_length is 16384 for temporal trajectory tasks.

Example:
    python -m scripts.torch_pipeline.run_inhibit_temporal_mse \\
        --h5ad ./data/4111_cells.h5ad \\
        --ckpt-dir /weights/maxtoki-1b-bionemo \\
        --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \\
        --variant 1b \\
        --gene-symbol ZLX1 \\
        --direction inhibit \\
        --seq-length 16384 \\
        --out-dir ./out/zlx1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .dataset_prep import build_paired_dataset
from .score import score
from .tokenizer import CellTokenizer


def _resolve_gene(args, tokenizer: CellTokenizer) -> tuple[str, int]:
    if args.gene:
        ensembl = args.gene
    elif args.gene_symbol:
        gene_name_id = (
            Path(__file__).resolve().parents[2]
            / "src" / "maxtoki_mlx" / "resources" / "gene_name_id.json"
        )
        with open(gene_name_id) as f:
            symbol_map = json.load(f)
        if args.gene_symbol not in symbol_map:
            sys.exit(f"unknown gene symbol: {args.gene_symbol}")
        ensembl = symbol_map[args.gene_symbol]
    else:
        sys.exit("must pass --gene <ENSG...> or --gene-symbol <SYMBOL>")
    if ensembl not in tokenizer.token_dict:
        sys.exit(f"gene {ensembl} not in token dictionary")
    return ensembl, int(tokenizer.token_dict[ensembl])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True, help="Path to 4111-cell h5ad")
    p.add_argument("--ckpt-dir", required=True, help="BioNeMo distcp checkpoint dir")
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to FULL token_dictionary_v1.json (with <boq>/<eoq>/numeric tokens)",
    )
    p.add_argument("--variant", choices=["217m", "1b"], default="1b")
    p.add_argument("--gene", help="Ensembl ID, e.g. ENSG00000109906")
    p.add_argument("--gene-symbol", help="HGNC symbol, e.g. ZLX1")
    p.add_argument(
        "--direction",
        choices=["inhibit", "delete", "overexpress"],
        default="inhibit",
    )
    p.add_argument("--seq-length", type=int, default=16384)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--context-parallel-size", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--out-dir", required=True, help="Where to write datasets, predictions, and summary")
    p.add_argument(
        "--prep-only",
        action="store_true",
        help="Only build the paired datasets; skip the BioNeMo predict + score steps.",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Skip prep + predict; only score existing prediction dirs under out-dir.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ds_dir = out_dir / "baseline.dataset"
    pert_ds_dir = out_dir / "perturbed.dataset"
    base_pred_dir = out_dir / "baseline_predictions"
    pert_pred_dir = out_dir / "perturbed_predictions"

    # Make sure CellTokenizer reads the full BioNeMo dict.
    os.environ["MAXTOKI_TOKEN_DICT"] = str(args.tokenizer_path)
    tokenizer = CellTokenizer()
    if not tokenizer.has_temporal_tokens:
        sys.exit(
            "Tokenizer is missing <boq>/<eoq>/numeric tokens. "
            "Pass --tokenizer-path pointing to the FULL BioNeMo token_dictionary_v1.json."
        )

    ensembl, gene_token = _resolve_gene(args, tokenizer)
    print(f"[info] gene={ensembl} token={gene_token} direction={args.direction}")

    if not args.score_only:
        print(f"[info] tokenizing + building paired datasets under {out_dir}")
        base_dir, pert_dir, prep_summary = build_paired_dataset(
            h5ad_path=args.h5ad,
            tokenizer=tokenizer,
            gene_token=gene_token,
            direction=args.direction,
            out_dir_baseline=base_ds_dir,
            out_dir_perturbed=pert_ds_dir,
            seq_length=args.seq_length,
        )
        print(f"[info] prep_summary={json.dumps(prep_summary, indent=2)}")
        with open(out_dir / "prep_summary.json", "w") as f:
            json.dump(prep_summary, f, indent=2)

    if args.prep_only:
        print("[info] --prep-only set; stopping after dataset build.")
        return

    if not args.score_only:
        from .predict_runner import run_headless_predict

        print(f"[info] running BioNeMo predict on baseline dataset")
        run_headless_predict(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            data_path=base_ds_dir,
            output_dir=base_pred_dir,
            variant=args.variant,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            devices=args.devices,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            precision=args.precision,
        )
        print(f"[info] running BioNeMo predict on perturbed dataset")
        run_headless_predict(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            data_path=pert_ds_dir,
            output_dir=pert_pred_dir,
            variant=args.variant,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            devices=args.devices,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            precision=args.precision,
        )

    print(f"[info] scoring")
    # Reload prep summary if --score-only and load gene_present from baseline dataset
    import datasets

    base_ds = datasets.load_from_disk(str(base_ds_dir))
    pert_ds = datasets.load_from_disk(str(pert_ds_dir))
    gene_present = [
        rec_b["n_query_tokens"] != rec_p["n_query_tokens"]
        or any(rec_b["input_ids"][i] != rec_p["input_ids"][i] for i in range(len(rec_b["input_ids"])))
        for rec_b, rec_p in zip(base_ds, pert_ds)
    ]
    paired_metadata = {
        "gene_present": gene_present,
        "direction": args.direction,
        "gene_ensembl": ensembl,
        "gene_token": gene_token,
    }
    base, pert, delta, summary = score(
        baseline_dir=base_pred_dir,
        perturbed_dir=pert_pred_dir,
        paired_metadata=paired_metadata,
        out_path=out_dir / "scores.npz",
    )

    summary_dict = {
        "n_cells": summary.n_cells,
        "n_cells_with_gene": summary.n_cells_with_gene,
        "mean_mse": summary.mean_mse,
        "mean_mse_present": summary.mean_mse_present,
        "mean_delta_t": summary.mean_delta_t,
        "mean_delta_t_present": summary.mean_delta_t_present,
        "abs_mean_delta_t": summary.abs_mean_delta_t,
        "gene_ensembl": summary.gene_ensembl,
        "gene_token": summary.gene_token,
        "direction": summary.direction,
        "variant": args.variant,
        "seq_length": args.seq_length,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("=== summary ===")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
