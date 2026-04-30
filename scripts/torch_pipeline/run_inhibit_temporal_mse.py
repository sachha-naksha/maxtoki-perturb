"""End-to-end zero-shot perturbation -> temporal MSE driver.

Pipeline (three stages):
    1. Build paired (baseline, perturbed) HF datasets from the experiment
       spec. The spec controls h5ad path, pseudotime / group columns, context
       selection, query selection, and the gene KO target. See spec.py.
    2. Run the upstream BioNeMo headless predict step on each dataset
       (writes predictions__rank_*.pt under <out>/baseline_predictions and
       <out>/perturbed_predictions).
    3. Read both prediction dirs, compute per-row delta_t and dataset MSE.

Stages 1 and 3 run anywhere (numpy / torch only). Stage 2 needs the BioNeMo
container (bionemo + nemo + megatron).

Usage::

    python -m scripts.torch_pipeline.run_inhibit_temporal_mse \\
        --spec ./configs/zlx1_inhibit.yaml \\
        --ckpt-dir /weights/maxtoki-1b-bionemo \\
        --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \\
        --variant 1b \\
        --out-dir ./out/zlx1

CLI overrides: --gene / --gene-symbol / --direction / --seq-length /
--h5ad let you tweak a single field of the spec without editing the YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .dataset_prep import build_paired_dataset
from .score import score
from .spec import ExperimentSpec, load_spec, spec_from_dict
from .tokenizer import CellTokenizer


def _resolve_gene_symbol(symbol: str) -> str:
    gene_name_id = (
        Path(__file__).resolve().parents[2]
        / "src" / "maxtoki_mlx" / "resources" / "gene_name_id.json"
    )
    with open(gene_name_id) as f:
        symbol_map = json.load(f)
    if symbol not in symbol_map:
        sys.exit(f"unknown gene symbol: {symbol}")
    return symbol_map[symbol]


def _resolve_gene_for_spec(spec: ExperimentSpec, tokenizer: CellTokenizer) -> tuple[str, int]:
    p = spec.perturbation
    ensembl = p.gene or (_resolve_gene_symbol(p.gene_symbol) if p.gene_symbol else None)
    if ensembl is None:
        sys.exit("perturbation.gene or perturbation.gene_symbol must be set in the spec.")
    if ensembl not in tokenizer.token_dict:
        sys.exit(f"gene {ensembl} not in token dictionary")
    return ensembl, int(tokenizer.token_dict[ensembl])


def _apply_cli_overrides(spec: ExperimentSpec, args) -> ExperimentSpec:
    if args.h5ad:
        spec.data.h5ad = args.h5ad
    if args.gene:
        spec.perturbation.gene = args.gene
        spec.perturbation.gene_symbol = None
    if args.gene_symbol:
        spec.perturbation.gene_symbol = args.gene_symbol
        spec.perturbation.gene = None
    if args.direction:
        spec.perturbation.direction = args.direction
    if args.seq_length:
        spec.seq_length = args.seq_length
    spec.validate()
    return spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="YAML or JSON ExperimentSpec")
    p.add_argument("--ckpt-dir", help="BioNeMo distcp checkpoint dir (required unless --prep-only)")
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to FULL token_dictionary_v1.json (with <boq>/<eoq>/numeric tokens)",
    )
    p.add_argument("--variant", choices=["217m", "1b"], default="1b")
    p.add_argument("--out-dir", required=True)
    # CLI override knobs (optional; spec wins otherwise)
    p.add_argument("--h5ad")
    p.add_argument("--gene")
    p.add_argument("--gene-symbol")
    p.add_argument("--direction", choices=["inhibit", "delete", "overexpress"])
    p.add_argument("--seq-length", type=int)
    # Predict knobs
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument("--context-parallel-size", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--prep-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    spec = load_spec(args.spec)
    spec = _apply_cli_overrides(spec, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ds_dir = out_dir / "baseline.dataset"
    pert_ds_dir = out_dir / "perturbed.dataset"
    base_pred_dir = out_dir / "baseline_predictions"
    pert_pred_dir = out_dir / "perturbed_predictions"

    os.environ["MAXTOKI_TOKEN_DICT"] = str(args.tokenizer_path)
    tokenizer = CellTokenizer()
    if not tokenizer.has_temporal_tokens:
        sys.exit(
            "Tokenizer dict has no <boq>/<eoq>/numeric tokens. Pass the full "
            "BioNeMo token_dictionary via --tokenizer-path."
        )

    ensembl, gene_token = _resolve_gene_for_spec(spec, tokenizer)
    print(f"[info] gene={ensembl} token={gene_token} direction={spec.perturbation.direction} "
          f"context.strategy={spec.context.strategy} query.strategy={spec.query.strategy} "
          f"seq_length={spec.seq_length}")

    with open(out_dir / "spec.resolved.json", "w") as f:
        json.dump({**spec.to_dict(), "gene_ensembl": ensembl, "gene_token": gene_token}, f, indent=2)

    if not args.score_only:
        print(f"[info] building paired datasets under {out_dir}")
        _, _, prep_summary = build_paired_dataset(
            spec=spec,
            tokenizer=tokenizer,
            out_dir_baseline=base_ds_dir,
            out_dir_perturbed=pert_ds_dir,
            gene_token=gene_token,
            gene_ensembl=ensembl,
        )
        with open(out_dir / "prep_summary.json", "w") as f:
            json.dump(prep_summary, f, indent=2)
        print(f"[info] prep_summary={json.dumps(prep_summary, indent=2)}")

    if args.prep_only:
        print("[info] --prep-only set; stopping after dataset build.")
        return

    if not args.ckpt_dir:
        sys.exit("--ckpt-dir is required unless --prep-only is set.")

    if not args.score_only:
        from .predict_runner import run_headless_predict

        for label, ds_dir, pred_dir in [
            ("baseline", base_ds_dir, base_pred_dir),
            ("perturbed", pert_ds_dir, pert_pred_dir),
        ]:
            print(f"[info] running BioNeMo predict on {label} dataset")
            run_headless_predict(
                ckpt_dir=args.ckpt_dir,
                tokenizer_path=args.tokenizer_path,
                data_path=ds_dir,
                output_dir=pred_dir,
                variant=args.variant,
                seq_length=spec.seq_length,
                micro_batch_size=args.micro_batch_size,
                devices=args.devices,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_model_parallel_size=args.pipeline_parallel_size,
                context_parallel_size=args.context_parallel_size,
                precision=args.precision,
            )

    print("[info] scoring")
    import datasets

    base_ds = datasets.load_from_disk(str(base_ds_dir))
    gene_present = [bool(r["gene_present_in_query"]) for r in base_ds]
    paired_metadata = {
        "gene_present": gene_present,
        "direction": spec.perturbation.direction,
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
        "n_rows": summary.n_cells,
        "n_rows_with_gene_in_query": summary.n_cells_with_gene,
        "mean_mse": summary.mean_mse,
        "mean_mse_present": summary.mean_mse_present,
        "mean_delta_t": summary.mean_delta_t,
        "mean_delta_t_present": summary.mean_delta_t_present,
        "abs_mean_delta_t": summary.abs_mean_delta_t,
        "gene_ensembl": ensembl,
        "gene_token": gene_token,
        "direction": spec.perturbation.direction,
        "context_strategy": spec.context.strategy,
        "query_strategy": spec.query.strategy,
        "variant": args.variant,
        "seq_length": spec.seq_length,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("=== summary ===")
    print(json.dumps(summary_dict, indent=2))


if __name__ == "__main__":
    main()
