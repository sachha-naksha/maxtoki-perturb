"""Spec-driven dataset prep for zero-shot perturbation scoring.

Given an :class:`ExperimentSpec`, materialize two HuggingFace datasets
(baseline + perturbed) where each row is one (group, query_cell) pair:

    baseline:  [<bos>, ctx_1, <eos>, ..., <bos>, ctx_K, <eos>,
                <boq>, query, <eoq>, dummy_numeric]
    perturbed: same structure, but query (and optionally context cells) have
               the target gene perturbed via delete / inhibit / overexpress.

Both datasets share row order so the scorer can join by index.

Cell selection rules
--------------------
- ``context.strategy=self``: K=1, context = [query]. Equivalent to the old
  cell-vs-itself baseline.
- ``context.strategy=prefix``: cells in the same group whose pseudotime is
  strictly less than the query's pseudotime, sorted ascending. Optionally
  include the query itself. Capped to ``max_cells`` (keep latest if exceeded).
- ``context.strategy=all_in_group``: every cell in the same group, sorted
  by pseudotime. Capped to ``max_cells``.
- ``context.strategy=explicit``: use ``context.explicit_indices`` as the
  context for *every* query (useful for fixed reference trajectories).

Length budget per cell
----------------------
With K context cells and 1 query, the input sequence costs::

    K * (rve_len_with_bos_eos) + 1 + rve_len_query_only + 2_specials_around_query + 1_dummy
  = K * rve_K + (rve_query - 2) + 4
    where rve_K and rve_query include their own bos/eos.

We solve for the per-cell RVE cap so the worst-case row fits in
``spec.seq_length``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .perturbation import perturb_tokens
from .spec import ContextSpec, ExperimentSpec
from .tokenizer import CellTokenizer, MODEL_INPUT_SIZE, pick_dummy_numeric_token


# ---------------------------------------------------------------------------
# h5ad helpers
# ---------------------------------------------------------------------------


def _resolve_ensembl_ids(adata) -> np.ndarray:
    var = adata.var
    if "ensembl_id" in var.columns:
        return var["ensembl_id"].values
    if "feature_id" in var.columns:
        return var["feature_id"].values
    names = np.asarray(adata.var_names.values)
    if names.size and isinstance(names[0], str) and names[0].startswith("ENSG"):
        return names
    raise ValueError(
        "Need var['ensembl_id'] / var['feature_id'] / ENSG-prefixed var_names."
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


# ---------------------------------------------------------------------------
# Cell selection per spec
# ---------------------------------------------------------------------------


@dataclass
class _CellPick:
    """An (index_into_adata, group_key, pseudotime, cell_id) record."""
    idx: int
    group: object
    pseudotime: float
    cell_id: str


def _enumerate_cells(adata, spec: ExperimentSpec) -> list[_CellPick]:
    n = adata.n_obs
    cell_id_col = spec.data.cell_id_col
    if cell_id_col and cell_id_col in adata.obs.columns:
        cell_ids = adata.obs[cell_id_col].astype(str).values
    else:
        cell_ids = adata.obs_names.astype(str).values

    if spec.data.group_col:
        if spec.data.group_col not in adata.obs.columns:
            raise KeyError(f"group_col {spec.data.group_col!r} not in adata.obs")
        groups = adata.obs[spec.data.group_col].values
    else:
        groups = np.zeros(n, dtype=int)

    if spec.data.pseudotime_col:
        if spec.data.pseudotime_col not in adata.obs.columns:
            raise KeyError(f"pseudotime_col {spec.data.pseudotime_col!r} not in adata.obs")
        ptimes = adata.obs[spec.data.pseudotime_col].astype(float).values
    else:
        ptimes = np.arange(n, dtype=float)

    return [
        _CellPick(idx=i, group=groups[i], pseudotime=float(ptimes[i]), cell_id=str(cell_ids[i]))
        for i in range(n)
    ]


def _filter_query_pool(picks: list[_CellPick], adata, spec: ExperimentSpec) -> list[_CellPick]:
    if not spec.query.filter_obs:
        return list(picks)
    keep = np.ones(len(picks), dtype=bool)
    for col, allowed in spec.query.filter_obs.items():
        if col not in adata.obs.columns:
            raise KeyError(f"query.filter_obs column {col!r} not in adata.obs")
        col_vals = adata.obs[col].values
        allowed_set = set(allowed)
        for i, pk in enumerate(picks):
            if col_vals[pk.idx] not in allowed_set:
                keep[i] = False
    return [p for p, k in zip(picks, keep) if k]


def _select_queries(picks: list[_CellPick], spec: ExperimentSpec) -> list[_CellPick]:
    if spec.query.strategy == "each_cell":
        return picks
    if spec.query.strategy == "latest_per_group":
        by_group: dict = {}
        for p in picks:
            cur = by_group.get(p.group)
            if cur is None or p.pseudotime > cur.pseudotime:
                by_group[p.group] = p
        return list(by_group.values())
    raise ValueError(f"unknown query.strategy: {spec.query.strategy}")


def _resolve_pool(
    all_cells: list[_CellPick],
    adata,
    cspec: ContextSpec,
) -> list[_CellPick]:
    """Resolve the pool ONCE (it's identical across queries when strategy=='pool')."""
    keep = np.ones(len(all_cells), dtype=bool)
    for col, allowed in cspec.pool_filter.items():
        if col not in adata.obs.columns:
            raise KeyError(f"context.pool_filter column {col!r} not in adata.obs")
        col_vals = adata.obs[col].values
        allowed_set = set(allowed)
        for i, p in enumerate(all_cells):
            if col_vals[p.idx] not in allowed_set:
                keep[i] = False
    pool = [p for p, k in zip(all_cells, keep) if k]
    if not pool:
        raise RuntimeError(f"context.pool_filter matched 0 cells: {cspec.pool_filter}")

    sel = cspec.pool_select
    assert sel is not None
    if sel.sort_by == "pseudotime":
        pool = sorted(pool, key=lambda p: p.pseudotime)
    elif sel.sort_by == "obs_index":
        pool = sorted(pool, key=lambda p: p.idx)
    else:
        raise ValueError(f"unknown pool_select.sort_by: {sel.sort_by}")

    n = min(sel.n, len(pool))
    if sel.pick == "first":
        chosen = pool[:n]
    elif sel.pick == "last":
        chosen = pool[-n:]
    elif sel.pick == "evenly_spaced":
        idxs = np.linspace(0, len(pool) - 1, n).round().astype(int).tolist()
        chosen = [pool[i] for i in idxs]
    elif sel.pick == "random":
        rng = np.random.default_rng(sel.seed)
        idxs = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
        chosen = [pool[i] for i in idxs]
    else:
        raise ValueError(f"unknown pool_select.pick: {sel.pick}")

    if cspec.ordering == "pseudotime":
        chosen = sorted(chosen, key=lambda p: p.pseudotime)
    elif cspec.ordering == "obs_index":
        chosen = sorted(chosen, key=lambda p: p.idx)
    return chosen


def _select_context(
    query: _CellPick,
    all_cells: list[_CellPick],
    cspec: ContextSpec,
    pool_cache: Optional[list[_CellPick]] = None,
) -> list[_CellPick]:
    if cspec.strategy == "self":
        return [query]

    if cspec.strategy == "explicit":
        idx_to_pick = {p.idx: p for p in all_cells}
        ctx = [idx_to_pick[i] for i in cspec.explicit_indices or [] if i in idx_to_pick]
        # Preserve user-supplied order; truncate to keep latest max_cells
        return ctx[-cspec.max_cells:]

    if cspec.strategy == "pool":
        if pool_cache is None:
            raise RuntimeError("pool_cache must be provided when strategy=='pool'")
        # Pool is identical across queries; just respect max_cells as a final cap.
        return pool_cache[-cspec.max_cells:]

    same_group = [p for p in all_cells if p.group == query.group]
    if cspec.strategy == "prefix":
        cand = [p for p in same_group if p.pseudotime < query.pseudotime]
    elif cspec.strategy == "all_in_group":
        cand = [p for p in same_group if p.idx != query.idx]
    else:
        raise ValueError(f"unknown context.strategy: {cspec.strategy}")

    if cspec.include_self and not any(p.idx == query.idx for p in cand):
        cand = cand + [query]

    if cspec.ordering == "pseudotime":
        cand = sorted(cand, key=lambda p: p.pseudotime)
    elif cspec.ordering == "obs_index":
        cand = sorted(cand, key=lambda p: p.idx)
    else:
        raise ValueError(f"unknown context.ordering: {cspec.ordering}")

    if not cand:
        # pre-trajectory cell with no valid context: fall back to self so the
        # row is still scoreable (will yield delta_t ~= 0 by construction)
        return [query]
    return cand[-cspec.max_cells:]


# ---------------------------------------------------------------------------
# Tokenization with shared length budget
# ---------------------------------------------------------------------------


def _tokenize_one_cell(adata, ensembl_ids, pick: _CellPick, tokenizer: CellTokenizer, max_len: int) -> list[int]:
    counts = _row_counts(adata, pick.idx)
    n_counts_val = None
    return tokenizer.tokenize_expression(ensembl_ids, counts, n_counts=n_counts_val, max_len=max_len)


def _per_cell_max_len(seq_length: int, n_context: int) -> int:
    """How many tokens (incl. BOS/EOS) each cell can use so the row fits in seq_length.

    Row layout:
        K context cells:  K * rve_K
        query block:      <boq> + (query_genes) + <eoq>  = (rve_query - 2) + 2 = rve_query
        trailing dummy:   1
    Total = K * rve_K + rve_query + 1 <= seq_length
    Distribute equally: rve = (seq_length - 1) // (K + 1)
    """
    k = max(1, n_context)
    rve = (seq_length - 1) // (k + 1)
    return min(rve, MODEL_INPUT_SIZE)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def _strip_bos_eos(toks: list[int], bos: int, eos: int) -> list[int]:
    if not toks:
        return toks
    if toks[0] == bos:
        toks = toks[1:]
    if toks and toks[-1] == eos:
        toks = toks[:-1]
    return toks


def _build_input_ids(
    context_cells: list[list[int]],   # each is [<bos>, genes, <eos>]
    query_cell: list[int],            # [<bos>, genes, <eos>]
    boq_id: int,
    eoq_id: int,
    dummy_numeric: int,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    out: list[int] = []
    for ctx in context_cells:
        out.extend(ctx)
    query_genes = _strip_bos_eos(query_cell, bos_id, eos_id)
    out.append(boq_id)
    out.extend(query_genes)
    out.append(eoq_id)
    out.append(dummy_numeric)
    return out


def _maybe_perturb(
    cell_tokens: list[int],
    apply: bool,
    gene_token: int,
    direction: str,
    bos: int,
    eos: int,
) -> list[int]:
    if not apply:
        return list(cell_tokens)
    return perturb_tokens(cell_tokens, gene_token, direction, bos, eos)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_paired_dataset(
    spec: ExperimentSpec,
    tokenizer: CellTokenizer,
    out_dir_baseline: str | Path,
    out_dir_perturbed: str | Path,
    gene_token: Optional[int] = None,
    gene_ensembl: Optional[str] = None,
) -> tuple[Path, Path, dict]:
    """Materialize baseline + perturbed datasets per the experiment spec."""
    import anndata as ad
    import datasets

    spec.validate()
    if not tokenizer.has_temporal_tokens:
        raise RuntimeError(
            "tokenizer is missing <boq>/<eoq>/numeric tokens - point "
            "MAXTOKI_TOKEN_DICT at the full BioNeMo dictionary."
        )

    if gene_token is None or gene_ensembl is None:
        raise ValueError("Pass gene_token and gene_ensembl (resolved by the caller).")

    adata = ad.read_h5ad(spec.data.h5ad)
    ensembl_ids = _resolve_ensembl_ids(adata)
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    boq, eoq = tokenizer.boq_id, tokenizer.eoq_id
    dummy_numeric = pick_dummy_numeric_token(tokenizer)

    all_picks = _enumerate_cells(adata, spec)
    query_pool = _filter_query_pool(all_picks, adata, spec)
    queries = _select_queries(query_pool, spec)
    if not queries:
        raise RuntimeError("query selection produced 0 cells - check filters / strategy.")

    pool_cache: Optional[list[_CellPick]] = None
    if spec.context.strategy == "pool":
        pool_cache = _resolve_pool(all_picks, adata, spec.context)
        print(
            f"[info] pool resolved to {len(pool_cache)} cells: "
            f"{[(p.cell_id, round(p.pseudotime, 3)) for p in pool_cache]}"
        )

    apply_ctx = (spec.perturbation.apply_to == "query_and_context")
    direction = spec.perturbation.direction

    base_records: list[dict] = []
    pert_records: list[dict] = []

    cache: dict[int, list[int]] = {}  # idx -> tokenized cell

    for q in queries:
        ctx_picks = _select_context(q, all_picks, spec.context, pool_cache=pool_cache)
        per_cell_cap = _per_cell_max_len(spec.seq_length, n_context=len(ctx_picks))

        ctx_tokens: list[list[int]] = []
        for cp in ctx_picks:
            key = (cp.idx, per_cell_cap)
            if key not in cache:
                cache[key] = _tokenize_one_cell(adata, ensembl_ids, cp, tokenizer, per_cell_cap)
            ctx_tokens.append(cache[key])

        q_key = (q.idx, per_cell_cap)
        if q_key not in cache:
            cache[q_key] = _tokenize_one_cell(adata, ensembl_ids, q, tokenizer, per_cell_cap)
        query_tokens = cache[q_key]

        ctx_tokens_pert = [
            _maybe_perturb(t, apply_ctx, gene_token, direction, bos, eos) for t in ctx_tokens
        ]
        query_tokens_pert = perturb_tokens(query_tokens, gene_token, direction, bos, eos)

        base_input_ids = _build_input_ids(
            ctx_tokens, query_tokens, boq, eoq, dummy_numeric, bos, eos
        )
        pert_input_ids = _build_input_ids(
            ctx_tokens_pert, query_tokens_pert, boq, eoq, dummy_numeric, bos, eos
        )

        meta = {
            "cell_id": q.cell_id,
            "group": str(q.group),
            "query_pseudotime": q.pseudotime,
            "context_pseudotimes": [c.pseudotime for c in ctx_picks],
            "context_cell_ids": [c.cell_id for c in ctx_picks],
            "n_context_cells": len(ctx_picks),
            "gene_token": int(gene_token),
            "gene_ensembl": gene_ensembl,
            "direction": direction,
            "apply_to": spec.perturbation.apply_to,
            "gene_present_in_query": gene_token in query_tokens,
        }
        base_records.append({**meta, "input_ids": base_input_ids, "condition": "baseline"})
        pert_records.append({**meta, "input_ids": pert_input_ids, "condition": "perturbed"})

    out_dir_baseline = Path(out_dir_baseline)
    out_dir_perturbed = Path(out_dir_perturbed)
    out_dir_baseline.parent.mkdir(parents=True, exist_ok=True)
    out_dir_perturbed.parent.mkdir(parents=True, exist_ok=True)

    base_ds = datasets.Dataset.from_list(base_records)
    pert_ds = datasets.Dataset.from_list(pert_records)
    base_ds.save_to_disk(str(out_dir_baseline))
    pert_ds.save_to_disk(str(out_dir_perturbed))

    summary = {
        "n_rows": len(base_records),
        "seq_length": spec.seq_length,
        "max_input_len_baseline": max(len(r["input_ids"]) for r in base_records),
        "max_input_len_perturbed": max(len(r["input_ids"]) for r in pert_records),
        "min_context_cells": min(r["n_context_cells"] for r in base_records),
        "max_context_cells": max(r["n_context_cells"] for r in base_records),
        "n_groups": len({r["group"] for r in base_records}),
        "gene_ensembl": gene_ensembl,
        "gene_token": int(gene_token),
        "direction": direction,
        "apply_to": spec.perturbation.apply_to,
        "context_strategy": spec.context.strategy,
        "query_strategy": spec.query.strategy,
        "n_query_cells_with_gene": sum(r["gene_present_in_query"] for r in base_records),
    }
    return out_dir_baseline, out_dir_perturbed, summary
