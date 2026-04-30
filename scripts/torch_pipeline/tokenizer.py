"""Rank-value cell tokenizer (PyTorch / x86 build).

Same algorithm as ``maxtoki_mlx.tokenizer.CellTokenizer`` but with no MLX imports
so it installs on CUDA boxes (A100 / H200). Resource JSONs are reused from the
sibling ``maxtoki_mlx`` package on disk - if this script is run from a checkout
of maxtoki-mlx, no extra downloads are needed.

    1. Keep only genes in the token vocab
    2. CPM-like normalize: counts / n_counts * 10_000
    3. Divide by training-corpus non-zero median
    4. Sort nonzero genes descending
    5. Wrap with [<bos>, ..., <eos>]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

TARGET_SUM = 10_000.0
MODEL_INPUT_SIZE = 4096

_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "src" / "maxtoki_mlx" / "resources"


class CellTokenizer:
    def __init__(
        self,
        token_dictionary: str | Path | None = None,
        gene_median: str | Path | None = None,
    ) -> None:
        self.token_dict: dict[str, int] = _load_json(
            token_dictionary, _RESOURCE_DIR / "token_dictionary.json"
        )
        raw_median = _load_json(gene_median, _RESOURCE_DIR / "gene_median.json")
        self.gene_median: dict[str, float] = {k: float(v) for k, v in raw_median.items()}

        self.bos_id = int(self.token_dict["<bos>"])
        self.eos_id = int(self.token_dict["<eos>"])
        self.pad_id = int(self.token_dict["<pad>"])

        self._gene_ids: dict[str, int] = {
            k: int(v)
            for k, v in self.token_dict.items()
            if k.startswith("ENSG") and k in self.gene_median
        }
        self._ensembl_list: list[str] = sorted(self._gene_ids.keys())
        self._ensembl_to_idx: dict[str, int] = {
            e: i for i, e in enumerate(self._ensembl_list)
        }
        self._token_id_arr = np.array(
            [self._gene_ids[e] for e in self._ensembl_list], dtype=np.int64
        )
        self._median_arr = np.array(
            [self.gene_median[e] for e in self._ensembl_list], dtype=np.float64
        )

    @property
    def num_genes(self) -> int:
        return len(self._gene_ids)

    def gene_token(self, ensembl_id: str) -> int:
        return self._gene_ids[ensembl_id]

    def has_gene(self, ensembl_id: str) -> bool:
        return ensembl_id in self._gene_ids

    def tokenize_expression(
        self,
        ensembl_ids: Iterable[str],
        expression: np.ndarray,
        n_counts: float | None = None,
        max_len: int = MODEL_INPUT_SIZE,
    ) -> list[int]:
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

        in_vocab = np.fromiter(
            (self._ensembl_to_idx.get(e, -1) for e in ensembl_ids),
            dtype=np.int64,
            count=len(ensembl_ids),
        )
        keep = in_vocab >= 0
        if not keep.any():
            return [self.bos_id, self.eos_id]
        kept_expr = expression[keep]
        kept_master_idx = in_vocab[keep]

        nz = kept_expr > 0
        if not nz.any():
            return [self.bos_id, self.eos_id]
        kept_expr = kept_expr[nz]
        kept_master_idx = kept_master_idx[nz]

        normalized = (kept_expr / n_counts) * TARGET_SUM
        normalized = normalized / self._median_arr[kept_master_idx]

        order = np.argsort(-normalized, kind="stable")
        ranked_master_idx = kept_master_idx[order]
        token_ids = self._token_id_arr[ranked_master_idx].tolist()
        token_ids = token_ids[: max_len - 2]
        return [self.bos_id] + token_ids + [self.eos_id]


def _load_json(override: str | Path | None, fallback: Path) -> dict:
    path = Path(override) if override is not None else fallback
    with open(path) as f:
        return json.load(f)
