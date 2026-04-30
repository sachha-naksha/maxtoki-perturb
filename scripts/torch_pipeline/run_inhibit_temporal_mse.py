"""CLI: zero-shot gene-inhibition temporal-MSE scoring on a 4111-cell h5ad.

Example:
    python -m scripts.torch_pipeline.run_inhibit_temporal_mse \\
        --h5ad ./data/4111_cells.h5ad \\
        --distcp /path/to/maxtoki-1b-bionemo \\
        --variant 1b \\
        --gene ENSG00000109906 \\
        --direction inhibit \\
        --batch-size 16 \\
        --out ./out/zlx1_inhibit.npz

Or, if you've already extracted the temporal head to a .pt and have HF backbone:
    python -m scripts.torch_pipeline.run_inhibit_temporal_mse \\
        --h5ad ./data/4111_cells.h5ad \\
        --hf theodoris-lab/MaxToki-1B-HF \\
        --temporal-head ./weights/temporal_head_1b.pt \\
        --variant 1b \\
        --gene ENSG00000109906 \\
        ...

The CLI accepts either an Ensembl ID (--gene ENSGxxxxxxxxxxx) or a HGNC
symbol (--gene-symbol ZLX1) - in the latter case it resolves via the
packaged gene_name_id.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import iter_tokenized_cells
from .model import MaxTokiTemporal
from .scoring import score_inhibition
from .tokenizer import CellTokenizer, _RESOURCE_DIR


def _resolve_gene(args, tokenizer: CellTokenizer) -> tuple[str, int]:
    if args.gene:
        ensembl = args.gene
    elif args.gene_symbol:
        with open(_RESOURCE_DIR / "gene_name_id.json") as f:
            symbol_map = json.load(f)
        if args.gene_symbol not in symbol_map:
            sys.exit(f"unknown gene symbol: {args.gene_symbol}")
        ensembl = symbol_map[args.gene_symbol]
    else:
        sys.exit("must pass --gene <ENSG...> or --gene-symbol <SYMBOL>")
    if not tokenizer.has_gene(ensembl):
        sys.exit(f"gene {ensembl} is not in the model vocabulary")
    return ensembl, tokenizer.gene_token(ensembl)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True, help="Path to 4111-cell h5ad")
    p.add_argument("--variant", choices=["217m", "1b"], default="1b")
    p.add_argument("--distcp", help="BioNeMo distcp dir (full model + temporal head)")
    p.add_argument("--hf", help="HF backbone repo or local path")
    p.add_argument("--temporal-head", help="Path to a saved temporal_head .pt state_dict")
    p.add_argument("--gene", help="Ensembl ID, e.g. ENSG00000109906")
    p.add_argument("--gene-symbol", help="HGNC symbol, e.g. ZLX1")
    p.add_argument(
        "--direction",
        choices=["inhibit", "delete", "overexpress"],
        default="inhibit",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--cell-id-col", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    p.add_argument("--out", help="Path to .npz output (per-cell deltas + summary)")
    args = p.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    tokenizer = CellTokenizer()
    ensembl, gene_token = _resolve_gene(args, tokenizer)
    print(f"[info] perturbing gene={ensembl} token={gene_token} direction={args.direction}")

    if args.distcp:
        model = MaxTokiTemporal.from_distcp(
            args.distcp, variant=args.variant, device=args.device, dtype=dtype
        )
    elif args.hf and args.temporal_head:
        model = MaxTokiTemporal.from_hf_with_temporal(
            args.hf,
            args.temporal_head,
            variant=args.variant,
            device=args.device,
            dtype=dtype,
        )
    else:
        sys.exit("must pass --distcp OR (--hf and --temporal-head)")

    print(f"[info] tokenizing cells from {args.h5ad}")
    cells = list(
        iter_tokenized_cells(
            args.h5ad, tokenizer, max_len=args.max_len, cell_id_col=args.cell_id_col
        )
    )
    print(f"[info] tokenized {len(cells)} cells")

    results, summary = score_inhibition(
        model, cells, gene_token, direction=args.direction, batch_size=args.batch_size
    )

    print("=== summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            cell_ids=np.array([r.cell_id for r in results]),
            gene_present=np.array([r.gene_present for r in results]),
            baseline=np.stack([r.baseline for r in results]),
            perturbed=np.stack([r.perturbed for r in results]),
            delta=np.stack([r.delta for r in results]),
            per_cell_mse=np.array([r.mse for r in results]),
            summary=np.array(json.dumps(summary)),
            gene=np.array(ensembl),
            direction=np.array(args.direction),
        )
        print(f"[info] wrote {args.out}")


if __name__ == "__main__":
    main()
