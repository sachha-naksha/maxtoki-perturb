"""Rank-value cell tokenizer.

Ports the Geneformer/MaxToki rank-value encoding:
    1. Keep only genes that are in the token vocabulary
    2. Normalize per-cell: counts / n_counts * 10_000 (CPM-like)
    3. Divide each gene by its training-corpus non-zero median
    4. Sort nonzero genes descending by normalized expression
    5. Wrap with [<bos>, ..., <eos>]

Based on NVIDIA-Digital-Bio/maxToki transcriptome_tokenizer.py.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np


TARGET_SUM = 10_000.0
MODEL_INPUT_SIZE = 4096


class CellTokenizer:
    """Tokenize single cells into MaxToki rank-value sequences.

    Args:
        token_dictionary: Optional override path to the JSON token dict.
            Default loads the packaged one (extracted from MaxToki-1B-bionemo/context/io.json).
        gene_median: Optional override path to the JSON gene median dict.
            Default loads the packaged one (from Geneformer gc104M, verified 100% overlap).
    """

    def __init__(
        self,
        token_dictionary: str | Path | None = None,
        gene_median: str | Path | None = None,
    ) -> None:
        self.token_dict: dict[str, int] = _load_json_resource(
            token_dictionary, "token_dictionary.json"
        )
        raw_median = _load_json_resource(gene_median, "gene_median.json")
        self.gene_median: dict[str, float] = {k: float(v) for k, v in raw_median.items()}

        self.bos_id = int(self.token_dict["<bos>"])
        self.eos_id = int(self.token_dict["<eos>"])
        self.pad_id = int(self.token_dict["<pad>"])

        # Only gene tokens that have both an ID and a median
        self._gene_ids: dict[str, int] = {
            k: int(v)
            for k, v in self.token_dict.items()
            if k.startswith("ENSG") and k in self.gene_median
        }
        # For efficient fencepost: ensembl → median
        # Cache arrays for gene tokens for vectorized operations
        self._ensembl_list: list[str] = sorted(self._gene_ids.keys())
        self._ensembl_to_idx: dict[str, int] = {
            e: i for i, e in enumerate(self._ensembl_list)
        }
        self._token_id_arr = np.array(
            [self._gene_ids[e] for e in self._ensembl_list], dtype=np.int32
        )
        self._median_arr = np.array(
            [self.gene_median[e] for e in self._ensembl_list], dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self.token_dict)

    @property
    def num_genes(self) -> int:
        return len(self._gene_ids)

    def tokenize_expression(
        self,
        ensembl_ids: Iterable[str],
        expression: np.ndarray,
        n_counts: float | None = None,
        max_len: int = MODEL_INPUT_SIZE,
    ) -> list[int]:
        """Tokenize a single cell from explicit (ensembl_id, expression) arrays.

        Args:
            ensembl_ids: gene identifiers (Ensembl IDs) present in the input
            expression: raw UMI counts aligned with ensembl_ids (1D array, length == len(ensembl_ids))
            n_counts: total UMI count for the cell. If None, computed as expression.sum().
            max_len: max tokens including BOS/EOS

        Returns:
            List of token IDs: [<bos>, rank1, rank2, ..., rankK, <eos>]
        """
        ensembl_ids = list(ensembl_ids)
        expression = np.asarray(expression, dtype=np.float64).ravel()
        if len(expression) != len(ensembl_ids):
            raise ValueError(
                f"expression length ({len(expression)}) != ensembl_ids length ({len(ensembl_ids)})"
            )
        if n_counts is None:
            n_counts = float(expression.sum())
        if n_counts <= 0:
            raise ValueError(f"n_counts must be positive, got {n_counts}")

        # Intersect with our gene vocabulary
        # Build per-input-gene lookup into our master arrays
        in_vocab_mask = np.zeros(len(ensembl_ids), dtype=bool)
        master_idx = np.empty(len(ensembl_ids), dtype=np.int64)
        for i, eid in enumerate(ensembl_ids):
            idx = self._ensembl_to_idx.get(eid, -1)
            if idx >= 0:
                in_vocab_mask[i] = True
                master_idx[i] = idx

        if not in_vocab_mask.any():
            return [self.bos_id, self.eos_id]

        kept_expr = expression[in_vocab_mask]
        kept_master_idx = master_idx[in_vocab_mask]

        # Drop zero-expression entries (rank-value only uses nonzero)
        nz_mask = kept_expr > 0
        if not nz_mask.any():
            return [self.bos_id, self.eos_id]
        kept_expr = kept_expr[nz_mask]
        kept_master_idx = kept_master_idx[nz_mask]

        # Normalize: CPM-like * 10_000, then divide by training median
        normalized = (kept_expr / n_counts) * TARGET_SUM
        medians = self._median_arr[kept_master_idx]
        normalized = normalized / medians

        # Sort descending
        sorted_order = np.argsort(-normalized, kind="stable")
        ranked_master_idx = kept_master_idx[sorted_order]
        token_ids = self._token_id_arr[ranked_master_idx].tolist()

        # Truncate to max_len - 2 (leaving room for BOS/EOS)
        token_ids = token_ids[: max_len - 2]
        return [self.bos_id] + token_ids + [self.eos_id]

    def tokenize_adata_row(self, adata_row, max_len: int = MODEL_INPUT_SIZE) -> list[int]:
        """Tokenize a single cell from an AnnData slice.

        Expects:
            adata_row.var["feature_id"] or .var["ensembl_id"] to hold Ensembl IDs
            adata_row.X: raw counts (1 x n_genes)
            adata_row.obs["n_counts"]: optional, computed from X.sum() if absent
        """
        import scipy.sparse as sp

        # Resolve Ensembl IDs column
        var = adata_row.var
        if "feature_id" in var.columns:
            ensembl_ids = var["feature_id"].values
        elif "ensembl_id" in var.columns:
            ensembl_ids = var["ensembl_id"].values
        else:
            # Assume var_names are already Ensembl IDs
            ensembl_ids = adata_row.var_names.values

        X = adata_row.X
        if sp.issparse(X):
            X = np.asarray(X.todense()).ravel()
        else:
            X = np.asarray(X).ravel()

        n_counts_val: float | None = None
        if "n_counts" in adata_row.obs.columns:
            n_counts_val = float(adata_row.obs["n_counts"].iloc[0])

        return self.tokenize_expression(ensembl_ids, X, n_counts=n_counts_val, max_len=max_len)

    def decode_gene(self, token_id: int) -> str | None:
        """Return the Ensembl ID for a given gene token ID, or None if not a gene."""
        # Reverse lookup - build once if needed
        if not hasattr(self, "_id_to_ensembl"):
            self._id_to_ensembl = {v: k for k, v in self._gene_ids.items()}
        return self._id_to_ensembl.get(token_id)


def _load_json_resource(override: str | Path | None, fallback_name: str) -> dict:
    if override is not None:
        with open(override) as f:
            return json.load(f)
    # Load from package resources
    try:
        pkg_files = resources.files("maxtoki_mlx") / "resources" / fallback_name
        with pkg_files.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, AttributeError, ModuleNotFoundError):
        # Fallback for direct source execution
        pkg_dir = Path(__file__).parent / "resources" / fallback_name
        with open(pkg_dir) as f:
            return json.load(f)
