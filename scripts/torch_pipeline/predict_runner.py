"""Thin wrapper around upstream ``bionemo.maxtoki.predict.predict``.

Runs the headless temporal regression on a saved HF dataset and writes
per-rank ``predictions__rank_*.pt`` files into ``output_dir``.

Multi-GPU support
-----------------

BioNeMo's ``PredictionWriter`` raises a ``ValueError`` whenever
``trainer.world_size > 1`` because per-rank batch indices are not preserved
through Lightning's predict path. We patch that check at runtime so multi-GPU
predict actually runs. Two distribution patterns work cleanly:

* **tensor-parallel only** (``tp=N, dp=1``): the model is split across GPUs but
  the data path is single-stream. Only rank 0 writes a valid prediction file;
  other ranks contribute partial logits internally that get all-gathered
  before output. ``score.py``/``sink_predictions`` returns the rank-0 file
  unchanged.

* **data-parallel** (``dp=N``): each rank writes its own contiguous slice of
  the dataset. Lightning's distributed sampler (with ``shuffle=False`` in
  predict mode) gives rank R the rows ``[R*N//R_total : (R+1)*N//R_total]``,
  so concatenating rank files in order recovers the original row sequence.
  ``score.py`` already does this.

For the OOM problem on 217M + 16k seq + 23k vocab the right answer is
**tp=4** on H200x4. tp slices the vocab dimension, which is what blows up in
``_headless_timelapse``'s softmax + renormalization step.
"""
from __future__ import annotations

from pathlib import Path

VARIANT_DEFAULTS = {
    "217m": {
        "seq_length": 16384,
        "scale_factor": 8.0,
        "micro_batch_size": 4,      # bump to 16 with --tensor-parallel-size 4 on H200x4
    },
    "1b":   {
        "seq_length": 16384,
        "scale_factor": 8.0,
        "micro_batch_size": 1,      # NOTE: 1B has no temporal head in stage-2 release
    },
}


def _patch_bionemo_for_multigpu_predict() -> None:
    """Bypass BioNeMo's blanket ``world_size > 1`` ban on predict.

    Idempotent; safe to call multiple times.
    """
    try:
        import bionemo.llm.utils.callbacks as _bncb
    except ImportError:
        return
    if getattr(_bncb.PredictionWriter, "_multigpu_check_patched", False):
        return

    _original_setup = _bncb.PredictionWriter.setup

    def _patched_setup(self, trainer, pl_module, stage):
        try:
            return _original_setup(self, trainer, pl_module, stage)
        except ValueError as e:
            if "Multi-GPU" in str(e) or "multi-gpu" in str(e).lower():
                return
            raise

    _bncb.PredictionWriter.setup = _patched_setup
    _bncb.PredictionWriter._multigpu_check_patched = True


def run_headless_predict(
    ckpt_dir: str | Path,
    tokenizer_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    variant: str = "1b",
    seq_length: int | None = None,
    micro_batch_size: int | None = None,
    devices: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    precision: str = "bf16-mixed",
    work_dir: str | Path | None = None,
    write_interval: str = "epoch",
    using_pretrain_dataset: bool = False,
    limit_predict_batches_to_n: int | None = None,
) -> Path:
    from bionemo.maxtoki.predict import predict as _bionemo_predict  # type: ignore

    defaults = VARIANT_DEFAULTS[variant]
    if seq_length is None:
        seq_length = defaults["seq_length"]
    if micro_batch_size is None:
        micro_batch_size = defaults["micro_batch_size"]

    if devices > 1 or tensor_parallel_size > 1 or pipeline_model_parallel_size > 1 \
       or context_parallel_size > 1:
        _patch_bionemo_for_multigpu_predict()
        print(f"[predict_runner] multi-GPU patch applied "
              f"(devices={devices}, tp={tensor_parallel_size}, "
              f"pp={pipeline_model_parallel_size}, cp={context_parallel_size})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _bionemo_predict(
        ckpt_dir=str(ckpt_dir),
        tokenizer_path=str(tokenizer_path),
        data_path=str(data_path),
        output_dir=output_dir,
        work_dir=Path(work_dir) if work_dir else None,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        devices=devices,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        precision=precision,
        generate_next_cell=False,
        write_interval=write_interval,
        using_pretrain_dataset=using_pretrain_dataset,
        limit_predict_batches_to_n=limit_predict_batches_to_n,
    )
    return output_dir