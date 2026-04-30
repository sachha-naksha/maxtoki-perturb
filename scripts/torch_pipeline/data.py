"""h5ad → tokenized cells loader for the 4111-cell dataset.

Notes:
    * Requires `anndata` and `scipy`. Install: `pip install anndata scipy`.
    * Expects raw counts in ``adata.X`` (or ``adata.raw.X`` if available).
    * Resolves Ensembl IDs from one of: var["feature_id"], var["ensembl_id"],
      or var_names if they look like ENSG ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .tokenizer import CellTokenizer, MODEL_INPUT_SIZE


@dataclass
class TokenizedCell:
    cell_index: int
    cell_id: str
    token_ids: list[int]


def _resolve_ensembl_ids(adata) -> np.ndarray:
    var = adata.var
    if "feature_id" in var.columns:
        return var["feature_id"].values
    if "ensembl_id" in var.columns:
        return var["ensembl_id"].values
    names = np.asarray(adata.var_names.values)
    if names.size and isinstance(names[0], str) and names[0].startswith("ENSG"):
        return names
    raise ValueError(
        "Could not resolve Ensembl IDs - need var['feature_id'], var['ensembl_id'], "
        "or ENSG-prefixed var_names."
    )


def _row_counts(adata, i: int) -> np.ndarray:
    import scipy.sparse as sp

    src = adata.raw.X if adata.raw is not None else adata.X
    row = src[i]
    if sp.issparse(row):
        row = np.asarray(row.todense()).ravel()
    else:
        row = np.asarray(row).ravel()
    return row


def iter_tokenized_cells(
    h5ad_path: str,
    tokenizer: CellTokenizer,
    max_len: int = MODEL_INPUT_SIZE,
    cell_id_col: str | None = None,
) -> Iterator[TokenizedCell]:
    """Stream tokenized cells from a single h5ad file."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    ensembl_ids = _resolve_ensembl_ids(adata)
    cell_ids = (
        adata.obs[cell_id_col].astype(str).values
        if cell_id_col and cell_id_col in adata.obs.columns
        else adata.obs_names.astype(str).values
    )
    n = adata.n_obs
    for i in range(n):
        counts = _row_counts(adata, i)
        n_counts_val = (
            float(adata.obs["n_counts"].iloc[i])
            if "n_counts" in adata.obs.columns
            else None
        )
        toks = tokenizer.tokenize_expression(
            ensembl_ids, counts, n_counts=n_counts_val, max_len=max_len
        )
        yield TokenizedCell(cell_index=i, cell_id=str(cell_ids[i]), token_ids=toks)


def load_all(
    h5ad_path: str,
    tokenizer: CellTokenizer,
    max_len: int = MODEL_INPUT_SIZE,
    cell_id_col: str | None = None,
) -> list[TokenizedCell]:
    return list(iter_tokenized_cells(h5ad_path, tokenizer, max_len, cell_id_col))
