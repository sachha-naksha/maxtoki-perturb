"""h5ad preprocessing for the MaxToki rank-value tokenizer.

What it does (and why):

1. Move raw integer counts back into ``adata.X``. The rank-value tokenizer
   does its own CPM-x10000 + median-divide normalization, so it needs raw
   counts -- log-normed values silently corrupt the ranking.

2. Resolve Ensembl IDs into ``adata.var["ensembl_id"]``. The tokenizer
   needs ENSG IDs (the model's vocabulary is keyed on them). The bundled
   ``gene_name_id.json`` covers the ~20k genes in MaxToki's vocabulary;
   anything outside that is dropped by the tokenizer anyway, so we drop
   it upfront for a smaller, faster file.

3. Optionally subset cells (e.g. drop 17 y/o donors that aren't in the
   context-pool or the query filter).

4. Standardize ``obs["n_counts"]`` (copy from ``nCount_RNA`` / ``nCount`` /
   recomputed ``X.sum(axis=1)``).

Run:

    python -m scripts.torch_pipeline.preprocess \\
        --in  ./data/rna_adata.h5ad \\
        --out ./data/rna_adata.preprocessed.h5ad \\
        --counts-layer counts \\
        --gene-symbol-col features \\
        --keep-ages 34 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_GENE_NAME_ID_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "maxtoki_mlx" / "resources" / "gene_name_id.json"
)


def _load_symbol_map() -> dict[str, str]:
    """Load the HGNC-symbol -> Ensembl-ID map packaged with maxtoki-mlx."""
    with open(_GENE_NAME_ID_PATH) as f:
        return json.load(f)


def _strip_version(ensg: str) -> str:
    """ENSG00000109906.12 -> ENSG00000109906."""
    if isinstance(ensg, str) and "." in ensg and ensg.startswith("ENSG"):
        return ensg.split(".", 1)[0]
    return ensg


def _resolve_ensembl_ids(adata, gene_symbol_col: str | None) -> None:
    """Add ``adata.var["ensembl_id"]``, mapping from a symbol or ENSG column."""
    var = adata.var
    if "ensembl_id" in var.columns:
        var["ensembl_id"] = var["ensembl_id"].map(_strip_version)
        return

    src_col = None
    if gene_symbol_col and gene_symbol_col in var.columns:
        src_col = gene_symbol_col
    elif "features" in var.columns:
        src_col = "features"
    elif "gene_symbol" in var.columns:
        src_col = "gene_symbol"
    elif "gene_name" in var.columns:
        src_col = "gene_name"

    src_values = var[src_col].astype(str).values if src_col else var.index.astype(str).values

    # Sniff the first non-null value: ENSG-prefixed -> already Ensembl IDs.
    sample = next((v for v in src_values if v and v != "nan"), "")
    if sample.startswith("ENSG"):
        var["ensembl_id"] = [_strip_version(v) for v in src_values]
        return

    # Otherwise treat src as HGNC symbols and look up.
    symbol_map = _load_symbol_map()
    var["ensembl_id"] = [symbol_map.get(s) for s in src_values]


def _set_n_counts(adata) -> None:
    if "n_counts" in adata.obs.columns:
        return
    for cand in ("nCount_RNA", "nCount", "total_counts"):
        if cand in adata.obs.columns:
            adata.obs["n_counts"] = adata.obs[cand].astype(float)
            return
    import numpy as np
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel().astype(float)
    else:
        adata.obs["n_counts"] = X.sum(axis=1).astype(float)


def preprocess(
    in_path: str,
    out_path: str,
    counts_layer: str | None = "counts",
    gene_symbol_col: str | None = "features",
    keep_ages: list[int] | None = None,
    age_col: str = "age",
    drop_unmapped_genes: bool = True,
) -> dict:
    """Apply the four-step pipeline and write a preprocessed h5ad.

    Args:
        in_path: source h5ad
        out_path: destination h5ad
        counts_layer: name of the layer holding raw integer counts. ``None``
            keeps ``adata.X`` as-is (use this if your X is already counts).
        gene_symbol_col: ``var`` column holding HGNC symbols or ENSG IDs.
            Falls back through ``features`` / ``gene_symbol`` / ``gene_name``
            / ``var_names`` if this column is absent.
        keep_ages: optional list of values to retain in ``obs[age_col]``.
            For your setup ``[34, 80]`` drops the 17 y/o donor (P26).
        age_col: ``obs`` column used by ``keep_ages``.
        drop_unmapped_genes: remove vars whose ``ensembl_id`` is NaN. Saves
            disk + tokenizer-iteration time; the tokenizer would drop them
            silently anyway.

    Returns:
        Summary dict (counts, mapping rate).
    """
    import anndata as ad
    ad.settings.allow_write_nullable_strings = True

    print(f"[preprocess] reading {in_path}")
    adata = ad.read_h5ad(in_path)
    n_obs0, n_vars0 = adata.shape

    if counts_layer:
        if counts_layer not in adata.layers:
            sys.exit(f"layer {counts_layer!r} not found - layers: {list(adata.layers)}")
        adata.X = adata.layers[counts_layer].copy()
        print(f"[preprocess] X <- layers['{counts_layer}']")

    _resolve_ensembl_ids(adata, gene_symbol_col)
    n_mapped = int(adata.var["ensembl_id"].notna().sum())
    print(f"[preprocess] mapped {n_mapped}/{n_vars0} vars to Ensembl IDs "
          f"({n_mapped / max(n_vars0, 1):.1%})")

    if drop_unmapped_genes:
        adata = adata[:, adata.var["ensembl_id"].notna()].copy()

    if keep_ages is not None:
        if age_col not in adata.obs.columns:
            sys.exit(f"obs[{age_col!r}] not found - columns: {list(adata.obs.columns)}")
        keep_set = set(keep_ages)
        mask = adata.obs[age_col].isin(keep_set).values
        kept = int(mask.sum())
        print(f"[preprocess] subsetting to obs[{age_col!r}] in {sorted(keep_set)}: "
              f"{kept}/{n_obs0} cells")
        adata = adata[mask].copy()

    _set_n_counts(adata)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"[preprocess] wrote {out_path}: shape={adata.shape}")

    return {
        "in_shape": [n_obs0, n_vars0],
        "out_shape": list(adata.shape),
        "n_genes_mapped_to_ensembl": n_mapped,
        "kept_ages": list(keep_ages) if keep_ages is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--counts-layer", default="counts",
                   help="layers[<this>] -> X. Pass 'none' if X already holds raw counts.")
    p.add_argument("--gene-symbol-col", default="features",
                   help="var column with HGNC symbols or Ensembl IDs.")
    p.add_argument("--keep-ages", nargs="*", type=int, default=None,
                   help="Subset obs[--age-col] to these values. e.g. --keep-ages 34 80")
    p.add_argument("--age-col", default="age")
    p.add_argument("--keep-unmapped-genes", action="store_true",
                   help="Don't drop vars without an Ensembl ID.")
    args = p.parse_args()
    counts_layer = None if args.counts_layer == "none" else args.counts_layer
    summary = preprocess(
        in_path=args.in_path,
        out_path=args.out_path,
        counts_layer=counts_layer,
        gene_symbol_col=args.gene_symbol_col,
        keep_ages=args.keep_ages,
        age_col=args.age_col,
        drop_unmapped_genes=not args.keep_unmapped_genes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()