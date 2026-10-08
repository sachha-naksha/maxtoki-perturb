"""CPU-only tests for the NextCell pipeline additions.

Scope:
    - spec round-trip (yaml-shaped dict -> ExperimentSpec -> dict is lossless)
    - task_type validation (per-cell cap sanity floor)
    - _build_input_ids grammar for both tasks
    - _per_cell_max_len arithmetic
    - score_nextcell decoder: specials/numeric/duplicates/invalid handling
    - score_nextcell metrics (Jaccard@k, Spearman on shared, rank shift) on
      hand-crafted paired lists with known answers
    - writer-fixture normalization: ragged batches across rank files

No bionemo / torch / anndata imports; runs on CPU inside the container.
Launched via deltaai/slurm/_run_nextcell_tests.sh (NOT on login node).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Mirror the pipeline's own sys.path bootstrap (scripts/torch_pipeline is not a
# proper package from the repo-root view - the module files add themselves to
# sys.path when run as `python file.py`).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "torch_pipeline"))


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def _minimal_raw_spec(task_type: str = "time_between_cells") -> dict:
    return {
        "data": {
            "h5ad": "/tmp/does-not-exist.h5ad",
            "pseudotime_col": "Pseudotime",
            "group_col": "sample",
            "cell_id_col": None,
            "count_col": "nCount_RNA",
        },
        "context": {
            "strategy": "pool",
            "max_cells": 3,
            "ordering": "pseudotime",
            "pool_filter": {"age": [34]},
            "pool_select": {"n": 3, "pick": "evenly_spaced",
                            "sort_by": "pseudotime", "seed": 0},
        },
        "query": {"strategy": "each_cell", "filter_obs": {"age": [80]}},
        "perturbation": {
            "gene": None, "gene_symbol": "PDK4",
            "direction": "inhibit", "apply_to": "query",
        },
        "seq_length": 16384,
        "task_type": task_type,
        "generation": {"max_tokens": 2048, "top_k": 1, "top_p": 0.0,
                       "temperature": 1.0},
    }


def test_spec_roundtrip_time_between_cells():
    import spec as _spec
    raw = _minimal_raw_spec("time_between_cells")
    s = _spec.spec_from_dict(raw)
    d = s.to_dict()
    assert d["task_type"] == "time_between_cells"
    assert d["generation"]["max_tokens"] == 2048  # default survives
    # Second trip is identical.
    s2 = _spec.spec_from_dict(d)
    assert s2.to_dict() == d


def test_spec_roundtrip_next_cell():
    import spec as _spec
    raw = _minimal_raw_spec("next_cell")
    s = _spec.spec_from_dict(raw)
    assert s.task_type == "next_cell"
    assert s.generation.top_k == 1
    d = s.to_dict()
    s2 = _spec.spec_from_dict(d)
    assert s2.to_dict() == d


def test_spec_default_task_is_time_between_cells_when_omitted():
    import spec as _spec
    raw = _minimal_raw_spec("time_between_cells")
    del raw["task_type"]
    del raw["generation"]
    s = _spec.spec_from_dict(raw)
    assert s.task_type == "time_between_cells"
    # Dataclass default follows NVIDIA documented command; value is unused on the
    # TBC path but the spec still carries it so the full config is serializable.
    assert s.generation.max_tokens == 4096


def test_spec_validate_per_cell_cap_floor():
    """next_cell with absurdly large max_tokens must fail validation, not silently
    produce a tiny per-cell cap."""
    import spec as _spec
    raw = _minimal_raw_spec("next_cell")
    raw["generation"]["max_tokens"] = 15000  # leaves (16384 - 15000 - 1) // 4 = 345 < 512
    with pytest.raises(ValueError, match="per-cell token cap"):
        _spec.spec_from_dict(raw)


def test_spec_validate_bad_sampling():
    import spec as _spec
    raw = _minimal_raw_spec("next_cell")
    raw["generation"]["top_p"] = 1.5
    with pytest.raises(ValueError, match="top_p"):
        _spec.spec_from_dict(raw)


# ---------------------------------------------------------------------------
# dataset_prep: grammar + cap arithmetic (no anndata needed)
# ---------------------------------------------------------------------------


def test_per_cell_max_len_time_between_cells():
    import dataset_prep as dp
    # (16384 - 1) // 4 = 4095, capped by MODEL_INPUT_SIZE=4096 -> 4095
    assert dp._per_cell_max_len(16384, n_context=3) == 4095
    assert dp._per_cell_max_len(16384, 3, task_type="time_between_cells") == 4095


def test_per_cell_max_len_next_cell_reserves_generation_budget():
    import dataset_prep as dp
    # NVIDIA grammar: K cells + (K-1) inter-Dt + 3 (query block) + 1 (sentinel)
    #                 + 1 (off-by-one safety) + max_tokens <= seq_length
    # (16384 - 4096 - 3 - 3 - 1) // 3 = 4093
    assert dp._per_cell_max_len(16384, 3, task_type="next_cell",
                                max_tokens_to_generate=4096) == 4093
    # Smaller gen budget leaves more per cell, capped by MODEL_INPUT_SIZE=4096.
    assert dp._per_cell_max_len(16384, 3, task_type="next_cell",
                                max_tokens_to_generate=2048) == 4096


def test_build_input_ids_time_between_cells_ends_with_numeric():
    import dataset_prep as dp
    bos, eos, boq, eoq, dummy = 100, 101, 102, 103, 42
    ctx = [[bos, 1, 2, 3, eos]]
    q = [bos, 7, 8, 9, eos]
    out = dp._build_input_ids(ctx, q, boq_id=boq, eoq_id=eoq,
                              dummy_numeric=dummy, bos_id=bos, eos_id=eos,
                              task_type="time_between_cells")
    assert out[-1] == dummy
    assert out[-2] == eoq
    assert out[-3] == 9  # last query gene (bos/eos stripped from query)


def test_build_input_ids_rejects_next_cell_task_type():
    """The TBC builder refuses next_cell; the caller must use the dedicated
    builder with Delta_t interleaving."""
    import dataset_prep as dp
    with pytest.raises(ValueError, match="next_cell"):
        dp._build_input_ids([[100, 1, 101]], [100, 7, 101], 102, 103, 42, 100, 101,
                            task_type="next_cell")


def test_build_input_ids_next_cell_grammar_nvidia():
    """NextCell grammar matches NVIDIA ``compose_example``:

        <bos> c1 <eos> Dt_12 <bos> c2 <eos> Dt_23 <bos> c3 <eos>
        <boq> Dt_q <eoq>  <bos>(sentinel)
    """
    import dataset_prep as dp
    bos, eos, boq, eoq = 100, 101, 102, 103
    c1 = [bos, 1, 2, eos]
    c2 = [bos, 3, 4, eos]
    c3 = [bos, 5, 6, eos]
    dt_12, dt_23, dt_q = 210, 211, 220
    out = dp._build_input_ids_next_cell(
        context_cells=[c1, c2, c3],
        inter_cell_dt_tokens=[dt_12, dt_23],
        query_dt_token=dt_q,
        boq_id=boq, eoq_id=eoq, bos_id=bos,
    )
    assert out == [
        bos, 1, 2, eos,  dt_12,
        bos, 3, 4, eos,  dt_23,
        bos, 5, 6, eos,
        boq, dt_q, eoq,
        bos,                       # sentinel
    ]
    # Collator grammar check.
    eoq_index = out.index(eoq)
    assert out[eoq_index + 1] == bos, "determine_task_type needs <bos> after <eoq>"


def test_build_input_ids_next_cell_rejects_dt_count_mismatch():
    import dataset_prep as dp
    bos, eos, boq, eoq = 100, 101, 102, 103
    with pytest.raises(ValueError, match="inter_cell_dt_tokens"):
        dp._build_input_ids_next_cell(
            context_cells=[[bos, 1, eos], [bos, 2, eos], [bos, 3, eos]],
            inter_cell_dt_tokens=[200],  # should be 2
            query_dt_token=210,
            boq_id=boq, eoq_id=eoq, bos_id=bos,
        )


def test_dt_token_lookup_and_clamp(monkeypatch):
    """_dt_token clamps out-of-range Dt values and returns the token id."""
    import dataset_prep as dp
    from types import SimpleNamespace
    # Mini-tokenizer: numeric tokens cover -2..2 inclusive.
    token_dict = {"-2": 1000, "-1": 1001, "0": 1002, "1": 1003, "2": 1004}
    numeric_ids = {1000: -2, 1001: -1, 1002: 0, 1003: 1, 1004: 2}
    fake = SimpleNamespace(token_dict=token_dict, numeric_token_ids=numeric_ids)
    assert dp._dt_token(0, fake) == (0, 1002)
    assert dp._dt_token(2, fake) == (2, 1004)
    # Out-of-range clamps.
    assert dp._dt_token(50, fake) == (2, 1004)
    assert dp._dt_token(-50, fake) == (-2, 1000)
    # Non-integer rounds to nearest int before lookup.
    assert dp._dt_token(1.4, fake) == (1, 1003)
    assert dp._dt_token(1.6, fake) == (2, 1004)


def test_build_input_ids_default_matches_time_between_cells():
    import dataset_prep as dp
    bos, eos, boq, eoq, dummy = 100, 101, 102, 103, 42
    ctx = [[bos, 1, 2, 3, eos]]
    q = [bos, 7, 8, 9, eos]
    default = dp._build_input_ids(ctx, q, boq, eoq, dummy, bos, eos)
    tbc = dp._build_input_ids(ctx, q, boq, eoq, dummy, bos, eos,
                              task_type="time_between_cells")
    assert default == tbc


def test_build_input_ids_rejects_unknown_task():
    """TBC builder refuses anything other than time_between_cells so bad
    task_type values can't silently build a wrong-grammar row."""
    import dataset_prep as dp
    with pytest.raises(ValueError, match="time_between_cells"):
        dp._build_input_ids([[100, 1, 101]], [100, 7, 101], 102, 103, 42, 100, 101,
                            task_type="generate_something_else")


# ---------------------------------------------------------------------------
# score_nextcell: decoder
# ---------------------------------------------------------------------------


def test_decode_generation_normal_sequence():
    import score_nextcell as sn
    id_to_ensg = {10: "ENSG00000000001", 11: "ENSG00000000002", 12: "ENSG00000000003"}
    specials = {"<bos>": 2, "<eos>": 3, "<boq>": 4, "<eoq>": 5, "<pad>": 0}
    numeric_ids: set[int] = set()
    tokens = [2, 10, 11, 12, 3]  # <bos>, g1, g2, g3, <eos>
    d = sn.decode_generation(tokens, id_to_ensg, specials, numeric_ids)
    assert d.ensg_order == ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
    assert d.saw_bos is True
    assert d.saw_eos is True
    assert d.n_invalid == 0
    assert d.n_duplicates == 0


def test_decode_generation_drops_duplicates_and_invalid():
    import score_nextcell as sn
    id_to_ensg = {10: "ENSG00000000001", 11: "ENSG00000000002"}
    specials = {"<bos>": 2, "<eos>": 3, "<boq>": 4, "<eoq>": 5, "<pad>": 0}
    numeric_ids = {99}
    tokens = [2, 10, 11, 10, 99, 999, 11, 3]  # dup 10, dup 11, numeric 99, invalid 999
    d = sn.decode_generation(tokens, id_to_ensg, specials, numeric_ids)
    assert d.ensg_order == ["ENSG00000000001", "ENSG00000000002"]
    assert d.n_duplicates == 2
    assert d.n_invalid == 2
    assert d.saw_eos is True


def test_decode_generation_no_eos_marks_capped():
    import score_nextcell as sn
    id_to_ensg = {10: "ENSG1", 11: "ENSG2"}
    specials = {"<bos>": 2, "<eos>": 3, "<boq>": 4, "<eoq>": 5, "<pad>": 0}
    tokens = [2, 10, 11]  # no <eos>
    d = sn.decode_generation(tokens, id_to_ensg, specials, set())
    assert d.saw_eos is False
    assert d.ensg_order == ["ENSG1", "ENSG2"]


# ---------------------------------------------------------------------------
# score_nextcell: metrics
# ---------------------------------------------------------------------------


def test_jaccard_at_k_known_values():
    import score_nextcell as sn
    assert sn.jaccard_at_k(["A", "B", "C"], ["A", "B", "C"], 3) == 1.0
    assert sn.jaccard_at_k(["A", "B", "C"], ["D", "E", "F"], 3) == 0.0
    # Overlap {A,B} / union {A,B,C,D} = 2/4
    assert sn.jaccard_at_k(["A", "B", "C"], ["A", "B", "D"], 3) == pytest.approx(2 / 4)


def test_spearman_on_shared_identical_lists():
    import score_nextcell as sn
    rho, n = sn.spearman_on_shared(["A", "B", "C", "D"], ["A", "B", "C", "D"])
    assert rho == pytest.approx(1.0)
    assert n == 4


def test_spearman_on_shared_reversed():
    import score_nextcell as sn
    rho, n = sn.spearman_on_shared(["A", "B", "C", "D"], ["D", "C", "B", "A"])
    assert rho == pytest.approx(-1.0)
    assert n == 4


def test_spearman_on_shared_handles_small_overlap():
    import score_nextcell as sn
    rho, n = sn.spearman_on_shared(["A"], ["A"])
    assert np.isnan(rho)
    assert n == 1


def test_rank_shift_mixed():
    import score_nextcell as sn
    shifts = sn.rank_shift(["A", "B", "C"], ["B", "A", "D"])
    # A: 0 -> 1 (+1); B: 1 -> 0 (-1); C only in baseline -> NaN; D only in perturbed -> NaN
    assert shifts["A"] == 1.0
    assert shifts["B"] == -1.0
    assert np.isnan(shifts["C"])
    assert np.isnan(shifts["D"])


# ---------------------------------------------------------------------------
# score_nextcell: writer-fixture normalization (ragged batches)
# ---------------------------------------------------------------------------


def _write_fake_rank_file(path: Path, batches: list[dict]) -> None:
    import torch
    payload = {"predictions": batches}
    torch.save(payload, path)


def test_extract_per_row_tokens_ragged_batches(tmp_path):
    import torch

    import score_nextcell as sn

    # Two batches of different sizes; different gen widths.
    b1 = {
        "generated_tokens": torch.tensor([[10, 11, 3, 0, 0],   # len 3
                                          [10, 11, 12, 13, 3]]),  # len 5
        "lengths":          torch.tensor([3, 5]),
        "finished_naturally": torch.tensor([True, True]),
    }
    b2 = {
        "generated_tokens": torch.tensor([[10, 11, 12, 0, 0, 0]]),  # len 3, no EOS
        "lengths":          torch.tensor([3]),
        "finished_naturally": torch.tensor([False]),
    }
    rank0 = tmp_path / "predictions__rank_0.pt"
    _write_fake_rank_file(rank0, [b1, b2])

    rows = sn._extract_per_row_tokens(tmp_path)
    assert len(rows) == 3
    assert rows[0]["tokens"] == [10, 11, 3]
    assert rows[1]["tokens"] == [10, 11, 12, 13, 3]
    assert rows[2]["tokens"] == [10, 11, 12]
    assert rows[2]["finished"] is False


def test_extract_per_row_tokens_nested_list_of_lists(tmp_path):
    """NextCell writer actually stores list[list[dict]] — one outer entry per
    microbatch, each wrapped in a singleton list. Smoke 3343372 crashed on
    exactly this shape; regression-guard here."""
    import torch

    import score_nextcell as sn

    def _mb(tokens, length, finished):
        return {
            "generated_tokens": torch.tensor([tokens]),
            "lengths": torch.tensor([length]),
            "finished_naturally": torch.tensor([finished]),
        }

    nested = [
        [_mb([10, 11, 3, 0], 3, True)],
        [_mb([10, 11, 12, 3], 4, True)],
        [_mb([10, 11, 12], 3, False)],
    ]
    rank0 = tmp_path / "predictions__rank_0.pt"
    torch.save(nested, rank0)

    rows = sn._extract_per_row_tokens(tmp_path)
    assert len(rows) == 3
    assert rows[0]["tokens"] == [10, 11, 3]
    assert rows[1]["tokens"] == [10, 11, 12, 3]
    assert rows[2]["tokens"] == [10, 11, 12]
    assert rows[2]["finished"] is False


def test_extract_per_row_tokens_multiple_rank_files_ordered(tmp_path):
    import torch

    import score_nextcell as sn

    b_rank0 = {
        "generated_tokens": torch.tensor([[10, 3]]),
        "lengths":          torch.tensor([2]),
        "finished_naturally": torch.tensor([True]),
    }
    b_rank1 = {
        "generated_tokens": torch.tensor([[11, 3]]),
        "lengths":          torch.tensor([2]),
        "finished_naturally": torch.tensor([True]),
    }
    _write_fake_rank_file(tmp_path / "predictions__rank_0.pt", [b_rank0])
    _write_fake_rank_file(tmp_path / "predictions__rank_1.pt", [b_rank1])

    rows = sn._extract_per_row_tokens(tmp_path)
    assert [r["tokens"] for r in rows] == [[10, 3], [11, 3]]
