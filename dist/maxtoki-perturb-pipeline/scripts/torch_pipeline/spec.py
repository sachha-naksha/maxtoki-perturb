"""Spec for what to feed the model in a zero-shot perturbation experiment.

The spec answers four questions per row of the prepared dataset:

    1. Which cell is the QUERY (the cell whose temporal prediction will shift
       under perturbation)?
    2. Which cells form the CONTEXT (the trajectory of past cells the model
       conditions on, in pseudotime order)?
    3. WHICH GENE is being KO'd / inhibited / overexpressed?
    4. Where is the perturbation applied -- query only, or also in context?

A spec is normally written as YAML / JSON and loaded with ``load_spec``.
Programmatic construction works too -- it's just a dataclass.

YAML example::

    data:
      h5ad: ./data/4111_cells.h5ad
      pseudotime_col: pseudotime         # obs column used for ordering
      group_col: donor_id                # cells of the same group share a trajectory
      cell_id_col: null                  # defaults to obs_names
      count_col: n_counts                # null -> compute from X.sum()

    context:
      strategy: prefix                   # self | prefix | all_in_group | explicit
      max_cells: 4                       # cap context length (in cells, not tokens)
      ordering: pseudotime               # pseudotime | obs_index
      include_self: false                # when prefix/all_in_group, include the query?
      explicit_indices: null             # list[int] when strategy == explicit

    query:
      strategy: each_cell                # each_cell | latest_per_group
      filter_obs:                        # optional - keep only cells matching these
        cell_type: ["fibroblast"]

    perturbation:
      gene_symbol: ZLX1                  # one of gene / gene_symbol must be set
      gene: null                         # Ensembl ID; takes precedence if set
      direction: inhibit                 # inhibit | delete | overexpress
      apply_to: query                    # query | query_and_context

Set ``context.strategy: self`` for the simple cell-vs-itself baseline
(equivalent to the previous default). Set ``query.strategy: latest_per_group``
to score one prediction per donor / sample instead of per cell.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional


ContextStrategy = Literal["self", "prefix", "all_in_group", "explicit", "pool"]
QueryStrategy = Literal["each_cell", "latest_per_group"]
Direction = Literal["inhibit", "delete", "overexpress"]
ApplyTo = Literal["query", "query_and_context"]
Ordering = Literal["pseudotime", "obs_index"]
PoolPick = Literal["first", "last", "evenly_spaced", "random"]


@dataclass
class DataSpec:
    """Where the cells live and which obs columns drive selection.

    Attributes:
        h5ad: Path to a single ``.h5ad`` file. Counts must live in ``adata.X``
            or ``adata.raw.X``; gene IDs in ``var["ensembl_id"]``,
            ``var["feature_id"]``, or ENSG-prefixed ``var_names``.
        pseudotime_col: Name of an ``obs`` column with monotone pseudotime /
            age / observation-time values. Required by context strategies
            ``prefix`` and ``all_in_group``, and by ``pool_select.sort_by="pseudotime"``.
        group_col: Name of an ``obs`` column that groups cells into trajectories
            (e.g. ``donor_id``). Required by ``prefix`` and ``all_in_group``.
        cell_id_col: Name of an ``obs`` column to use as the per-cell ID in
            output records. ``None`` falls back to ``adata.obs_names``.
        count_col: Name of an ``obs`` column holding pre-computed total UMI
            counts per cell. ``None`` re-computes from the count matrix.
    """
    h5ad: str
    pseudotime_col: Optional[str] = None
    group_col: Optional[str] = None
    cell_id_col: Optional[str] = None
    count_col: Optional[str] = "n_counts"


@dataclass
class PoolSelect:
    """How to choose context cells from a filtered pool (one pool, reused for every query)."""
    n: int = 3                          # number of context cells to pick
    pick: PoolPick = "evenly_spaced"    # first | last | evenly_spaced | random
    sort_by: Ordering = "pseudotime"    # how to sort the pool before picking
    seed: int = 0


@dataclass
class ContextSpec:
    """How the trajectory of past cells (the context) is assembled per query.

    Attributes:
        strategy: One of ``self`` (cell vs. itself), ``prefix`` (same-group
            cells with smaller pseudotime), ``all_in_group`` (all other cells
            in same group), ``explicit`` (fixed adata indices for every query),
            or ``pool`` (filter then pick from a cross-group pool).
        max_cells: Final cap on number of context cells per row.
        ordering: How context cells are ordered in the input sequence.
        include_self: When ``True`` and strategy is ``prefix`` /
            ``all_in_group``, append the query itself to the context.
        explicit_indices: ``adata.obs`` row indices used as context for every
            query. Required when ``strategy == "explicit"``.
        pool_filter: ``{obs_col: [allowed_values]}`` filter applied to the
            full cell list. Required when ``strategy == "pool"``.
        pool_select: How to pick a fixed-size context from the filtered pool.
            Required when ``strategy == "pool"``.
    """
    strategy: ContextStrategy = "self"
    max_cells: int = 4
    ordering: Ordering = "pseudotime"
    include_self: bool = False
    explicit_indices: Optional[list[int]] = None
    pool_filter: dict[str, list[Any]] = field(default_factory=dict)
    pool_select: Optional[PoolSelect] = None


@dataclass
class QuerySpec:
    """Which cells become queries (one row each in the output dataset).

    Attributes:
        strategy: ``each_cell`` -> one row per cell; ``latest_per_group`` ->
            one row per group, the cell with the largest pseudotime.
        filter_obs: ``{obs_col: [allowed_values]}`` to score only matching
            cells (e.g. ``{"cell_type": ["fibroblast"]}``).
    """
    strategy: QueryStrategy = "each_cell"
    filter_obs: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class PerturbationSpec:
    """Which gene to perturb, how, and where it's applied in the row.

    Attributes:
        gene: Ensembl ID (e.g. ``"ENSG00000109906"``). Takes precedence over
            ``gene_symbol`` if both are set.
        gene_symbol: HGNC symbol (e.g. ``"ZLX1"``). Resolved against the
            packaged ``gene_name_id.json``.
        direction: ``inhibit`` moves the gene's token to the lowest rank;
            ``delete`` removes it; ``overexpress`` moves it to the front.
        apply_to: ``query`` perturbs only the query cell; ``query_and_context``
            also perturbs every cell in the context.
    """
    gene: Optional[str] = None
    gene_symbol: Optional[str] = None
    direction: Direction = "inhibit"
    apply_to: ApplyTo = "query"


@dataclass
class ExperimentSpec:
    """Top-level spec for one zero-shot perturbation experiment.

    A spec is normally written as YAML (see ``configs/`` for examples) and
    loaded via :func:`load_spec`. It can also be constructed programmatically.

    Attributes:
        data: Input h5ad + obs-column mapping.
        context: How the trajectory of past cells is assembled per query.
        query: Which cells are scored.
        perturbation: Gene KO / inhibition / overexpression target.
        seq_length: Model context length. Default ``16384`` for trajectory
            tasks; the per-cell rank-value cap is computed from this so the
            worst-case row fits.
    """
    data: DataSpec
    context: ContextSpec = field(default_factory=ContextSpec)
    query: QuerySpec = field(default_factory=QuerySpec)
    perturbation: PerturbationSpec = field(default_factory=PerturbationSpec)
    seq_length: int = 16384

    def validate(self) -> None:
        """Raise ``ValueError`` if the spec is internally inconsistent."""
        c, p = self.context, self.perturbation
        if c.strategy in ("prefix", "all_in_group") and not self.data.pseudotime_col and c.ordering == "pseudotime":
            raise ValueError(
                f"context.strategy={c.strategy} with ordering=pseudotime requires "
                f"data.pseudotime_col to be set."
            )
        if c.strategy in ("prefix", "all_in_group") and not self.data.group_col:
            raise ValueError(
                f"context.strategy={c.strategy} requires data.group_col so cells "
                f"can be grouped into trajectories."
            )
        if c.strategy == "explicit" and not c.explicit_indices:
            raise ValueError("context.strategy=explicit requires context.explicit_indices.")
        if c.strategy == "pool":
            if not c.pool_filter:
                raise ValueError("context.strategy=pool requires context.pool_filter (e.g. {'donor_age': [34]}).")
            if c.pool_select is None:
                raise ValueError("context.strategy=pool requires context.pool_select.")
            if c.pool_select.sort_by == "pseudotime" and not self.data.pseudotime_col:
                raise ValueError("context.pool_select.sort_by=pseudotime requires data.pseudotime_col.")
        if c.max_cells < 1:
            raise ValueError("context.max_cells must be >= 1")
        if not (p.gene or p.gene_symbol):
            raise ValueError("perturbation must set one of gene or gene_symbol.")

    def to_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "context": asdict(self.context),
            "query": asdict(self.query),
            "perturbation": asdict(self.perturbation),
            "seq_length": self.seq_length,
        }


def load_spec(path: str | Path) -> ExperimentSpec:
    """Load an :class:`ExperimentSpec` from a YAML or JSON file.

    Args:
        path: Filesystem path. Suffix decides the parser (``.yaml`` / ``.yml``
            -> PyYAML, anything else -> ``json``).

    Returns:
        A validated :class:`ExperimentSpec`.

    Raises:
        ValueError: If the spec fails validation.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return spec_from_dict(raw)


def spec_from_dict(raw: dict) -> ExperimentSpec:
    """Build an :class:`ExperimentSpec` from a nested dict (typically parsed YAML)."""
    data = DataSpec(**raw["data"])
    context_raw = dict(raw.get("context", {}))
    pool_select_raw = context_raw.pop("pool_select", None)
    if pool_select_raw:
        context_raw["pool_select"] = PoolSelect(**pool_select_raw)
    context = ContextSpec(**context_raw)
    query_raw = raw.get("query", {})
    query = QuerySpec(
        strategy=query_raw.get("strategy", "each_cell"),
        filter_obs=query_raw.get("filter_obs", {}) or {},
    )
    perturbation = PerturbationSpec(**raw["perturbation"])
    spec = ExperimentSpec(
        data=data,
        context=context,
        query=query,
        perturbation=perturbation,
        seq_length=int(raw.get("seq_length", 16384)),
    )
    spec.validate()
    return spec
