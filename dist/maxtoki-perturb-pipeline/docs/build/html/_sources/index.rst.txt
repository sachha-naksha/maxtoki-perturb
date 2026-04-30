maxtoki-mlx torch_pipeline
==========================

CUDA / x86 sibling of the ``maxtoki_mlx`` Apple-Silicon port. Runs **inside the
BioNeMo container** on A100 / H200 (e.g. NCSA Delta), loading the upstream
`NVIDIA-Digital-Bio/maxToki <https://github.com/NVIDIA-Digital-Bio/maxToki>`_
distcp checkpoint directly. The temporal head is the headless-MSE regression
of ``bionemo.maxtoki.model.MaxTokiFineTuneModel`` -- no separate temporal
weights to extract.

What this pipeline does
-----------------------

For each query cell *q* in your h5ad:

1. **Tokenize** the cell with the rank-value encoder (descending by
   median-normalized expression).
2. **Assemble context**: a configurable set of past cells (per-donor prefix /
   cross-donor pool / explicit indices / cell-vs-itself), each rank-value
   tokenized.
3. **Build paired records** -- baseline and perturbed -- with the same row
   order. The perturbed row applies ``inhibit`` / ``delete`` / ``overexpress``
   to the gene of interest in the query (and optionally in the context too).
4. **Run the BioNeMo headless predict step** twice (baseline + perturbed),
   producing per-row regression outputs at the ``<eoq>`` position (interpreted
   as "predicted time elapsed between context and query").
5. **Score**: Δt = ``perturbed - baseline``; per-row MSE = Δt²; dataset MSE =
   ``mean(Δt²)``. Signed Δt is also kept so direction (aging / rejuvenation)
   is recoverable.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   pipeline
   spec_reference
   delta_recipes

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/spec
   api/tokenizer
   api/perturbation
   api/dataset_prep
   api/predict_runner
   api/score
   api/run_inhibit_temporal_mse

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
