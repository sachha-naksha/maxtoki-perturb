"""DeltaAI entry wrapper for scripts/torch_pipeline/run_inhibit_temporal_mse.py.

Applies a schema-based `datasets.generate_fingerprint` override before
delegating to the real driver. The override exists because the DeltaAI env
pairs `pyarrow 25.0.1` with `datasets 5.0.1` + `dill 0.4.1`, and
`datasets.utils._dill._save_arrowTable` -> `dill.save_function` eventually
tries to pickle `pyarrow.MonthDayNano`, which pyarrow 14+ no longer exposes
as `builtins.MonthDayNano`. The fingerprint is only used as a cache key for
`.map()`/`.filter()` chains and as the dataset's `__fingerprint__`
attribute; a deterministic schema hash is a safe substitute.

Usage (same CLI as the underlying driver): ::

    python _torch_pipeline_entry.py --spec ... --ckpt-dir ... --out-dir ...
"""
from __future__ import annotations

import hashlib
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


# Patch BOTH module locations: arrow_dataset.py does `from datasets.fingerprint
# import generate_fingerprint` at import time, so patching only the source module
# leaves the already-imported alias stale.
datasets.fingerprint.generate_fingerprint = _schema_fingerprint
datasets.arrow_dataset.generate_fingerprint = _schema_fingerprint

_DRIVER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "torch_pipeline"
    / "run_inhibit_temporal_mse.py"
)
sys.argv[0] = str(_DRIVER)
runpy.run_path(str(_DRIVER), run_name="__main__")
