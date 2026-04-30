"""Rank-value cell tokenizer (PyTorch / x86 build).

Same algorithm as ``maxtoki_mlx.tokenizer.CellTokenizer`` but with no MLX
imports so it installs on CUDA boxes (A100 / H200).

Token-dictionary precedence (first match wins):
    1. ``token_dictionary=`` arg
    2. ``$MAXTOKI_TOKEN_DICT`` env var - point this at the full BioNeMo dict
       (e.g. ``maxToki/resources/token_dictionary_v1.json`` or whatever is
       packaged inside your distcp checkpoint). The full dict contains
       ``<boq>``, ``<eoq>`` and the numeric timestep tokens needed for the
       TimeBetweenCells temporal task.
    3. fallback to the maxtoki_mlx packaged dict (gene + special tokens only,
       no temporal tokens - fine for backbone-only work, NOT for temporal MSE).

    1. Keep only genes in the token vocab
    2. CPM-like normalize: counts / n_counts * 10_000
    3. Divide by training-corpus non-zero median
    4. Sort nonzero genes descending
    5. Wrap with [<bos>, ..., <eos>]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

TARGET_SUM = 10_000.0
MODEL_INPUT_SIZE = 4096

_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "src" / "maxtoki_mlx" / "resources"


def _resolve_token_dict_path(arg: str | Path | None) -> Path:
    if arg is not None:
        return Path(arg)
    env = os.environ.get("MAXTOKI_TOKEN_DICT")
    if env:
        return Path(env)
    return _RESOURCE_DIR / "token_dictionary.json"


class CellTokenizer:
    def __init__(
        self,
        token_dictionary: str | Path | None = None,
        gene_median: str | Path | None = None,
    ) -> None:
        self.token_dict: dict[str, int] = _load_json(
            _resolve_token_dict_path(token_dictionary)
        )
        raw_median = _load_json_or_default(
            gene_median, _RESOURCE_DIR / "gene_median.json"
        )
        self.gene_median: dict[str, float] = {k: float(v) for k, v in raw_median.items()}

        self.bos_id = int(self.token_dict["<bos>"])
        self.eos_id = int(self.token_dict["<eos>"])
        self.pad_id = int(self.token_dict["<pad>"])
        self.boq_id = int(self.token_dict["<boq>"]) if "<boq>" in self.token_dict else None
        self.eoq_id = int(self.token_dict["<eoq>"]) if "<eoq>" in self.token_dict else None
        self.numeric_token_ids: dict[int, int] = {
            int(v): int(k)
            for k, v in self.token_dict.items()
            if isinstance(k, str) and k.lstrip("-").isdigit()
        }
        self.has_temporal_tokens = (
            self.boq_id is not None
            and self.eoq_id is not None
            and len(self.numeric_token_ids) > 0
        )

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


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_json_or_default(override: str | Path | None, fallback: Path) -> dict:
    return _load_json(Path(override) if override is not None else fallback)


def pick_dummy_numeric_token(tokenizer: "CellTokenizer") -> int:
    """Pick a numeric token to put after <eoq> so the upstream collate function
    classifies the record as TimeBetweenCells. The actual value is irrelevant
    for prediction - we only read the model's prediction at <eoq>."""
    if not tokenizer.has_temporal_tokens:
        raise RuntimeError(
            "Token dictionary has no numeric/temporal tokens. Point "
            "MAXTOKI_TOKEN_DICT at the full BioNeMo token_dictionary_v1.json."
        )
    if 0 in {int(v) for v in tokenizer.numeric_token_ids.values()}:
        for tid, val in tokenizer.numeric_token_ids.items():
            if int(val) == 0:
                return tid
    return next(iter(tokenizer.numeric_token_ids))
