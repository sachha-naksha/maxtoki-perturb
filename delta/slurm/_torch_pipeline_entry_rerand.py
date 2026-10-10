"""Delta entry wrapper: same as deltaai/slurm/_torch_pipeline_entry.py (datasets
fingerprint patch) plus an optional re-randomization of the checkpoint rows that
were never trained.

The HF->BioNeMo import copies the 20275 HF embedding / LM-head rows positionally
and leaves rows >= 20275 (<boq>, <eoq>, 3000 numeric tokens in the aligned dict) at
their random init (std 0.02). Set RERAND_SEED=<int> to re-draw those rows with a
different seed right before the first predict step. If the TimeBetweenCells readout
(argmax over numeric-token logits at <eoq>) depends on those rows, the predictions
change; if it reads something trained, they do not.

Env:
  RERAND_SEED   int, enables the re-randomization (unset = plain run)
  RERAND_FROM   first row to re-draw (default 20275)
  RERAND_STD    normal std (default 0.02 = model.yaml init_method_std)
"""
from __future__ import annotations

import hashlib
import os
import runpy
import sys
from pathlib import Path

import datasets.arrow_dataset
import datasets.fingerprint


def _schema_fingerprint(dataset) -> str:
    try:
        payload = dataset.data.table.schema.serialize().to_pybytes()
    except Exception:
        payload = repr(getattr(dataset, "data", "")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


datasets.fingerprint.generate_fingerprint = _schema_fingerprint
datasets.arrow_dataset.generate_fingerprint = _schema_fingerprint

if os.environ.get("RERAND_SEED"):
    import torch
    import bionemo.maxtoki.predict as _bp

    _seed = int(os.environ["RERAND_SEED"])
    _from = int(os.environ.get("RERAND_FROM", "20275"))
    _std = float(os.environ.get("RERAND_STD", "0.02"))
    # Keyed by model object: the driver calls predict() twice (baseline, perturbed) and
    # each call builds a fresh model, so both instances must be re-drawn identically.
    _done: set[int] = set()

    def _rerandomize(model):
        if id(model) in _done:
            return
        n = 0
        for name, p in model.named_parameters():
            if name.endswith("word_embeddings.weight") or name.endswith("output_layer.weight"):
                g = torch.Generator(device="cpu").manual_seed(_seed + (1 if "output_layer" in name else 0))
                rows = p.shape[0] - _from
                new = torch.randn(rows, p.shape[1], generator=g) * _std
                with torch.no_grad():
                    p[_from:].copy_(new.to(p.dtype).to(p.device))
                print(f"[rerand] seed={_seed} rows {_from}..{p.shape[0]-1} of {name} re-drawn (std={_std})")
                n += 1
        if n == 0:
            raise RuntimeError("[rerand] no word_embeddings/output_layer parameters found")
        _done.add(id(model))

    def _wrap(fn):
        def inner(model, batch, *a, **kw):
            _rerandomize(model)
            return fn(model, batch, *a, **kw)
        return inner

    _bp.maxtoki_headless_predict_step = _wrap(_bp.maxtoki_headless_predict_step)
    _bp.maxtoki_generate_predict_step = _wrap(_bp.maxtoki_generate_predict_step)
    print(f"[rerand] enabled: seed={_seed} from_row={_from} std={_std}")

_DRIVER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "torch_pipeline"
    / "run_inhibit_temporal_mse.py"
)
sys.argv[0] = str(_DRIVER)
runpy.run_path(str(_DRIVER), run_name="__main__")
