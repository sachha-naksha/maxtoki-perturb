Quickstart
==========

End-to-end run on the bundled "young context, old query" example spec.

1. Stage your data
------------------

Drop your h5ad somewhere readable, e.g. ``./data/4111_cells.h5ad``. Required
columns:

* ``var["ensembl_id"]`` (or ``var["feature_id"]`` / ENSG-prefixed ``var_names``)
* ``obs["donor_age"]`` (or whatever you'll filter on)
* ``obs["pseudotime"]`` if you use trajectory-based context strategies
* ``obs["donor_id"]`` if you group cells into per-donor trajectories
* optional ``obs["n_counts"]``

2. Edit the spec
----------------

.. code-block:: yaml

   # configs/example_young_context_old_query.yaml (excerpt)
   data:
     h5ad: ./data/4111_cells.h5ad
     pseudotime_col: pseudotime
     group_col: donor_id

   context:
     strategy: pool
     pool_filter: {donor_age: [34]}
     pool_select: {n: 3, pick: evenly_spaced, sort_by: pseudotime}

   query:
     strategy: each_cell
     filter_obs: {donor_age: [80]}

   perturbation:
     gene_symbol: ZLX1
     direction: inhibit
     apply_to: query

   seq_length: 16384

3. Run
------

.. code-block:: bash

   python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
       --spec scripts/torch_pipeline/configs/example_young_context_old_query.yaml \
       --ckpt-dir /weights/maxtoki-1b-bionemo \
       --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \
       --variant 1b \
       --out-dir ./out/zlx1

4. Outputs
----------

Under ``./out/zlx1/``:

================================== ==========================================
``spec.resolved.json``             Fully-resolved spec (includes Ensembl ID)
``baseline.dataset/``              HF dataset fed to BioNeMo (baseline rows)
``perturbed.dataset/``             HF dataset (perturbed rows, same order)
``baseline_predictions/``          ``predictions__rank_*.pt`` (BioNeMo)
``perturbed_predictions/``         ``predictions__rank_*.pt`` (BioNeMo)
``scores.npz``                     Per-row arrays + summary blob
``prep_summary.json``              Dataset-build summary
``summary.json``                   Final dataset-level scoring summary
================================== ==========================================

5. Recommended sanity checks
----------------------------

* Always run a no-op control (``--direction delete`` on a non-expressed gene)
  to verify ``mean_delta_t ≈ 0``.
* Run the same gene with ``context.strategy: prefix`` (within-donor) and
  ``pool`` (cross-donor young reference) and compare per-cell ranking. Strong
  perturbation effects should agree.
* Use ``--prep-only`` on a CPU box first to inspect tokenization.
* Use ``--score-only`` to re-score after adjusting the spec without burning
  GPU on prediction.
