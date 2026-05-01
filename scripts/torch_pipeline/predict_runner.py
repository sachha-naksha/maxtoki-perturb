"""Thin wrapper around upstream ``bionemo.maxtoki.predict.predict``.

Runs the headless temporal regression on a saved HF dataset and writes
per-rank ``predictions__rank_*.pt`` files into ``output_dir``.

Both the 217M and 1B variants share the same ``MaxTokiMultitaskFineTuneConfig``
- the architecture is read out of the distcp via ``load_settings_from_checkpoint``
so all that changes between variants is ``ckpt_dir``. ``seq_length`` and
``scale_factor`` are listed in the config's ``override_parent_fields``,
which is why setting them here actually takes effect.
"""
from __future__ import annotations

from pathlib import Path

VARIANT_DEFAULTS = {
    "217m": {
        "seq_length": 16384,
        "scale_factor": 8.0,        # passes through; 217m uses standard RoPE
        "micro_batch_size": 4,
    },
}


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
    """Run upstream BioNeMo prediction on a single HF dataset.

    Returns the output dir. The caller is expected to invoke this twice -
    once on the baseline dataset, once on the perturbed dataset - then join
    by row order in ``score.py``.

    Imports upstream lazily so this module can be inspected without the
    BioNeMo container.
    """
    from bionemo.maxtoki.predict import predict as _bionemo_predict  # type: ignore

    defaults = VARIANT_DEFAULTS[variant]
    if seq_length is None:
        seq_length = defaults["seq_length"]
    if micro_batch_size is None:
        micro_batch_size = defaults["micro_batch_size"]

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
        # the headless temporal task - no token generation
        generate_next_cell=False,
        write_interval=write_interval,
        using_pretrain_dataset=using_pretrain_dataset,
        limit_predict_batches_to_n=limit_predict_batches_to_n,
    )
    return output_dir
