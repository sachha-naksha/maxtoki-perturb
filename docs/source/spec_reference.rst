Spec reference
==============

This page lists every field in the YAML / JSON spec. The dataclass
definitions live in :mod:`scripts.torch_pipeline.spec`.

Top-level (:class:`~scripts.torch_pipeline.spec.ExperimentSpec`)
---------------------------------------------------------------

.. code-block:: yaml

   data: { ... }              # required
   context: { ... }           # default = ContextSpec()
   query: { ... }             # default = QuerySpec()
   perturbation: { ... }      # required (gene or gene_symbol)
   seq_length: 16384          # model context length

``data`` -- :class:`~scripts.torch_pipeline.spec.DataSpec`
----------------------------------------------------------

.. code-block:: yaml

   data:
     h5ad: ./data/4111_cells.h5ad   # required
     pseudotime_col: pseudotime     # optional, required by prefix/all_in_group/pool(sort_by=pseudotime)
     group_col: donor_id            # optional, required by prefix/all_in_group
     cell_id_col: null              # null -> use adata.obs_names
     count_col: n_counts            # null -> compute from X.sum() per row

``context`` -- :class:`~scripts.torch_pipeline.spec.ContextSpec`
----------------------------------------------------------------

The trajectory of past cells the model conditions on.

================== ==================================================== =================================
``strategy``       what it does                                         requires
================== ==================================================== =================================
``self``           context = [query] (cell vs. itself)                  nothing
``prefix``         same group, ``pseudotime < query.pseudotime``        ``group_col``, ``pseudotime_col``
``all_in_group``   same group, exclude query, sort by pseudotime        ``group_col``
``explicit``       fixed adata indices for every query                  ``explicit_indices``
``pool``           filter by obs cols, sort, pick N                     ``pool_filter``, ``pool_select``
================== ==================================================== =================================

Common fields:

.. code-block:: yaml

   context:
     strategy: pool                   # see table above
     max_cells: 4                     # final cap (in cells, not tokens)
     ordering: pseudotime             # pseudotime | obs_index
     include_self: false              # only used by prefix / all_in_group

Strategy-specific fields:

.. code-block:: yaml

   context:
     strategy: explicit
     explicit_indices: [12, 47, 103, 256]   # adata.obs row indices

   context:
     strategy: pool
     pool_filter:
       donor_age: [34]                # OR donor_id: [...] / cell_type: [...]
     pool_select:                     # PoolSelect dataclass
       n: 3                           # how many cells to pick
       pick: evenly_spaced            # first | last | evenly_spaced | random
       sort_by: pseudotime            # pseudotime | obs_index
       seed: 0                        # only used when pick=random

``query`` -- :class:`~scripts.torch_pipeline.spec.QuerySpec`
------------------------------------------------------------

.. code-block:: yaml

   query:
     strategy: each_cell              # each_cell | latest_per_group
     filter_obs:                      # restrict to matching cells
       cell_type: ["fibroblast"]
       donor_age: [80]

``perturbation`` -- :class:`~scripts.torch_pipeline.spec.PerturbationSpec`
--------------------------------------------------------------------------

.. code-block:: yaml

   perturbation:
     gene: ENSG00000109906            # OR gene_symbol: ZLX1 (resolved to Ensembl)
     direction: inhibit               # inhibit | delete | overexpress
     apply_to: query                  # query | query_and_context

Token-grammar semantics:

* ``inhibit`` -> move the gene's token to the lowest non-special position
  (just before ``<eoq>`` for the query, just before ``<eos>`` for context cells).
* ``delete`` -> remove the gene's token entirely.
* ``overexpress`` -> move the gene's token to the front (highest rank).

Validation
----------

:meth:`~scripts.torch_pipeline.spec.ExperimentSpec.validate` raises
``ValueError`` for the typical mismatches:

* ``prefix`` / ``all_in_group`` without ``group_col`` or ``pseudotime_col``
* ``explicit`` without ``explicit_indices``
* ``pool`` without ``pool_filter`` or ``pool_select``
* ``perturbation`` without ``gene`` or ``gene_symbol``

Every CLI invocation also writes ``spec.resolved.json`` under ``--out-dir``
with the fully merged spec (after CLI overrides) plus the resolved Ensembl
ID and token ID, so the experiment is reproducible from logs alone.
