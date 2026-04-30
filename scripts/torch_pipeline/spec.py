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


ContextStrategy = Literal["self", "prefix", "all_in_group", "explicit"]
QueryStrategy = Literal["each_cell", "latest_per_group"]
Direction = Literal["inhibit", "delete", "overexpress"]
ApplyTo = Literal["query", "query_and_context"]
Ordering = Literal["pseudotime", "obs_index"]


@dataclass
class DataSpec:
    h5ad: str
    pseudotime_col: Optional[str] = None
    group_col: Optional[str] = None
    cell_id_col: Optional[str] = None
    count_col: Optional[str] = "n_counts"


@dataclass
class ContextSpec:
    strategy: ContextStrategy = "self"
    max_cells: int = 4
    ordering: Ordering = "pseudotime"
    include_self: bool = False
    explicit_indices: Optional[list[int]] = None


@dataclass
class QuerySpec:
    strategy: QueryStrategy = "each_cell"
    filter_obs: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class PerturbationSpec:
    gene: Optional[str] = None
    gene_symbol: Optional[str] = None
    direction: Direction = "inhibit"
    apply_to: ApplyTo = "query"


@dataclass
class ExperimentSpec:
    data: DataSpec
    context: ContextSpec = field(default_factory=ContextSpec)
    query: QuerySpec = field(default_factory=QuerySpec)
    perturbation: PerturbationSpec = field(default_factory=PerturbationSpec)
    seq_length: int = 16384

    def validate(self) -> None:
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
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return spec_from_dict(raw)


def spec_from_dict(raw: dict) -> ExperimentSpec:
    data = DataSpec(**raw["data"])
    context = ContextSpec(**raw.get("context", {}))
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
