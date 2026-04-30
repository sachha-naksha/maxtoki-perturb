Pipeline architecture
=====================

::

  spec + h5ad
       |
       v
  +-----------------------------+
  | dataset_prep                |  scripts/torch_pipeline/dataset_prep.py
  | for each query cell q:      |
  |   ctx = select_context(q)   |  per spec.context.strategy
  |   base_ids = [<bos> ctx <eos>] x K + [<boq> q <eoq> dummy]
  |   pert_ids = same, but perturb(gene, direction) on q
  |              (and ctx if apply_to == query_and_context)
  +-----------------------------+
       |              |
       v              v
  baseline.dataset  perturbed.dataset    (HuggingFace, same row order)
       |              |
       v              v
  +-----------------------------+
  | predict_runner              |  scripts/torch_pipeline/predict_runner.py
  | bionemo.maxtoki.predict     |
  | (headless TimeBetweenCells) |
  | x2  (one per dataset)       |
  +-----------------------------+
       |              |
       v              v
  baseline_predictions/    perturbed_predictions/
  predictions__rank_*.pt   predictions__rank_*.pt
       |              |
       +------+-------+
              v
  +-----------------------------+
  | score                       |  scripts/torch_pipeline/score.py
  | delta_t = pert_t - base_t   |
  | per_cell_mse = delta_t**2   |
  +-----------------------------+
              |
              v
  scores.npz + summary.json

Why the row layout looks like that
----------------------------------

Each row's ``input_ids`` is::

    [<bos> ctx_1 <eos>, ..., <bos> ctx_K <eos>, <boq> query <eoq>, dummy_numeric]

* The K context cells, each wrapped in ``<bos>`` / ``<eos>``, are exactly the
  multi-cell paragraph format MaxToki was trained on.
* The query cell is wrapped in ``<boq>`` / ``<eoq>`` -- the upstream
  TimeBetweenCells task header -- so the model knows it's the cell whose
  time delta should be predicted.
* The trailing numeric token is required only because the upstream
  ``MaxTokiTokenizer.collate_batch_multitask`` calls ``determine_task_type``,
  which looks at the token immediately after ``<eoq>``. Its value never
  enters the prediction; the headless predict step gathers the regression
  output at the ``<eoq>`` position.

How the per-cell rank-value cap is computed
-------------------------------------------

For ``K`` context cells + 1 query, distributed equally::

    per_cell_max_len = (seq_length - 1) // (K + 1)

So at ``seq_length=16384`` with 4 context cells, each cell can use up to
``(16384 - 1) // 5 ≈ 3276`` tokens including its own ``<bos>`` / ``<eos>``.
The rank-value tokenizer truncates per-cell tokens to this cap.

Why the same multitask config covers both 217M and 1B
-----------------------------------------------------

Upstream ``MaxTokiMultitaskFineTuneConfig`` reads the model architecture
(layers, hidden, heads, RoPE) from the distcp checkpoint via
``load_settings_from_checkpoint``. So switching variants is just changing
``--ckpt-dir``. ``seq_length`` and ``scale_factor`` are listed in the
config's ``override_parent_fields``, which is why explicitly passing
``--seq-length 16384`` actually takes effect (without that, the
checkpoint's training-time seq_length wins).

Three context strategies, three different questions
---------------------------------------------------

* ``self`` -- "How much does perturbing g move this cell relative to itself?"
  Useful as a coarse sanity probe; doesn't use the temporal-trajectory
  signal the model was trained on.
* ``prefix`` / ``all_in_group`` -- "Within this donor's pseudotime
  trajectory, how does perturbing g shift apparent progression?"
  Within-donor signal, weak interpretation as years.
* ``pool`` -- "Relative to a young-donor reference trajectory, how much does
  perturbing g move this old-donor cell along the aging axis?" The
  recommended setup for an aging-readout because the predicted Δt has clean
  donor-age semantics.
