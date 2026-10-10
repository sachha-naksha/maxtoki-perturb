"""Cross-donor Stage 2 coverage, held-out-donor isolation and signed targets."""
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "torch_pipeline"))
import aging_temporal as at
import prepare_aging_stage2_trajectory as trajectory


def test_context_spans_donors_without_query_and_retains_signed_chronology():
    pools = {}
    for donor, times in (("YM2", [10., 12., 14., 16.]), ("OM6", [1., 2., 3., 4.])):
        pools[(donor, "SKM")] = [dict(cell_id=f"{donor}_{i}", donor=donor, time=time)
                                for i, time in enumerate(times)]
    context = trajectory.sample_context(pools, "SKM", "OM6_1", random.Random(3))
    assert [cell["donor"] for cell in context] == ["YM2", "YM2", "OM6"]
    assert len({cell["cell_id"] for cell in context}) == 3
    assert "OM6_1" not in {cell["cell_id"] for cell in context}
    # Donor chronology is retained even if raw pseudotimes overlap/invert.
    assert context[2]["time"] - context[1]["time"] < 0
    reverse_counts = trajectory.sample_context(pools, "SKM", "YM2_1", random.Random(3), row=1)
    assert [cell["donor"] for cell in reverse_counts] == ["YM2", "OM6", "OM6"]


def test_training_query_balance_covers_young_and_old_annotations():
    groups = {(donor, annotation): [dict(cell_id=f"{donor}_{annotation}_{i}")
                                    for i in range(3)]
              for donor in ("YM2", "OM6") for annotation in ("A", "B")}
    selected = trajectory.training_queries(groups, 24, random.Random(9))
    assert len(selected) == 24
    assert {cell["cell_id"] for cell in selected} == {
        cell["cell_id"] for group in groups.values() for cell in group}
    assert all(sum(cell["cell_id"].startswith(f"{donor}_{annotation}") for cell in selected) == 6
               for donor, annotation in groups)


@pytest.fixture
def prepared_source(tmp_path):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset

    tokens = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3,
              "ENSG1": 4, "ENSG2": 5, "ENSG3": 6, "ENSG4": 7}
    tokens.update({str(i): i + 28 for i in range(-20, 21)})
    tokens.update({"<boq>": 49, "<eoq>": 50})
    records, rows = [], []
    for donor in ("YM2", "OM6", "OM9"):
        for annotation in ("A", "B"):
            for i in range(5):
                time = float(i + 8 if donor == "YM2" else i)
                if donor == "YM2" and annotation == "A" and i == 4:
                    time = np.nan
                cell_id = f"{donor}_{annotation}_{i}"
                records.append(dict(cell_id=cell_id,
                                    input_ids=[2, 7 if donor == "OM9" else 4 + i % 3, 3],
                                    Pseudotime=time, sample=donor))
                rows.append(dict(sample=donor, Annotation=annotation, Pseudotime=time))
    h5ad_path = tmp_path / "source.h5ad"
    with pd.option_context("future.infer_string", False):
        obs = pd.DataFrame(rows, index=[record["cell_id"] for record in records])
        ad.AnnData(np.ones((len(rows), 4)), obs=obs).write_h5ad(h5ad_path)
    cell_path = tmp_path / "cells.dataset"
    Dataset(pa.Table.from_pylist(records), fingerprint="crossdonorfixture").save_to_disk(str(cell_path))
    source = tmp_path / "source"
    source.mkdir()
    at.dump(source / "token_dictionary.json", tokens)
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_hash = hashlib.sha256(data.obs[["Pseudotime", "sample", "Annotation"]].to_csv().encode()).hexdigest()
    finally:
        data.file.close()
    at.dump(source / "manifest.json", dict(
        units="pseudotime", splits={"train": ["YM2"], "val": ["OM6"], "test": ["OM9"]},
        pseudotime_col="Pseudotime", trajectory_col="Annotation",
        time_scale=1., label_scalar=200., seq_length=40, max_tokens=4, cell_cap=4,
        seed=42, tokenizer_sha256=at.digest(source / "token_dictionary.json"),
        source_h5ad=str(h5ad_path), source_cells=str(cell_path), source_obs_sha256=obs_hash,
        rope_factor=4., old_context_len=4096, attention_backend="sdpa",
        counts={"train": {}, "val": {}, "test": {}}))
    return SimpleNamespace(source=source, output=tmp_path / "cross_donor", train_young="YM2",
                           train_old="OM6", val_donor="OM9", examples=24, eval_examples=6, seed=44)


def test_preparation_supervises_both_training_donors_and_holds_out_old(prepared_source):
    from datasets import load_from_disk

    args = prepared_source
    trajectory.prepare(args)
    model = at.manifest(args.output)
    assert model["splits"] == {"train": ["YM2", "OM6"], "val": ["OM9"]}
    assert model["dataset_paths"] == {"train": "train.dataset", "val": "val.dataset", "test": "val.dataset"}
    assert not (args.output / "test.dataset").exists()
    assert model["label_scalar"] == 200.
    assert (args.output / "token_dictionary.json").read_bytes() == (
        args.source / "token_dictionary.json").read_bytes()
    bank = json.loads((args.output / "training_reference_bank.json").read_text())
    assert len(bank) == 19
    assert {cell["donor"] for cell in bank} == {"YM2", "OM6"}
    train = json.loads((args.output / "train_references.json").read_text())
    val = json.loads((args.output / "val_references.json").read_text())
    assert {r["donor"] for r in train} == {"YM2", "OM6"}
    assert all(sum(r["donor"] == donor for r in train) == 12 for donor in ("YM2", "OM6"))
    assert len({r["query_cell_id"] for r in val}) == 6
    assert all(r["donor"] == "OM9" for r in val)
    for ref in train + val:
        assert set(ref["context_donors"]) == {"YM2", "OM6"}
        assert ref["context_donors"] == sorted(ref["context_donors"], key=lambda d: {"YM2": 0, "OM6": 1}[d])
        assert ref["query_cell_id"] not in ref["context_cell_ids"]
        assert all(f"_{ref['trajectory']}_" in cell_id for cell_id in ref["context_cell_ids"])
        assert ref["delta_pseudotime"] == ref["query_pseudotime"] - ref["context_pseudotimes"][-1]
        assert ref["context_intervals"] == [b-a for a,b in zip(
            ref["context_pseudotimes"], ref["context_pseudotimes"][1:])]
    assert any(dt < 0 for ref in train for dt in ref["context_intervals"])
    assert all("OM9" not in cell_id for ref in train
               for cell_id in [ref["query_cell_id"], *ref["context_cell_ids"]])
    assert len(load_from_disk(str(args.output / "train.dataset"))) == 48
    assert len(load_from_disk(str(args.output / "val.dataset"))) == 12
    nc = list(load_from_disk(str(args.output / "val_nc.dataset")))
    tbc = list(load_from_disk(str(args.output / "val_tbc.dataset")))
    assert all(7 not in row["input_ids"] for row in nc)
    assert all(7 in row["input_ids"] for row in tbc)
    assert all(row["input_ids"][-1] == 2 for row in nc)
    assert json.loads((args.output / "audit.json").read_text())["excluded_by_donor"] == {"YM2": 1}
    with pytest.raises(FileExistsError):
        trajectory.prepare(args)


def test_heldout_values_cannot_change_training_or_context_selection(prepared_source):
    import anndata as ad
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset, load_from_disk

    args = prepared_source
    trajectory.prepare(args)
    before_train = list(load_from_disk(str(args.output / "train.dataset")))
    before_refs = json.loads((args.output / "val_references.json").read_text())
    original = at.manifest(args.source)
    records = list(load_from_disk(original["source_cells"]))
    # Authoritative held-out values change, while cell IDs and annotation do not.
    for record in records:
        if record["sample"] == "OM9":
            record["Pseudotime"] += .5
            record["input_ids"] = [2, 6, 3]
    revised_cells = args.source.parent / "changed_cells.dataset"
    Dataset(pa.Table.from_pylist(records), fingerprint="changedheldout").save_to_disk(str(revised_cells))
    revised_h5ad = args.source.parent / "changed.h5ad"
    with pd.option_context("future.infer_string", False):
        data = ad.read_h5ad(original["source_h5ad"])
        data.obs.loc[data.obs["sample"].astype(str) == "OM9", "Pseudotime"] += .5
        data.write_h5ad(revised_h5ad)
    original["source_cells"] = str(revised_cells)
    original["source_h5ad"] = str(revised_h5ad)
    reread = ad.read_h5ad(revised_h5ad)
    original["source_obs_sha256"] = hashlib.sha256(
        reread.obs[["Pseudotime", "sample", "Annotation"]].to_csv().encode()).hexdigest()
    at.dump(args.source / "manifest.json", original)
    args.output = args.source.parent / "changed_prepared"
    trajectory.prepare(args)
    after_refs = json.loads((args.output / "val_references.json").read_text())
    assert before_train == list(load_from_disk(str(args.output / "train.dataset")))
    assert [r["query_cell_id"] for r in before_refs] == [r["query_cell_id"] for r in after_refs]
    assert [r["context_cell_ids"] for r in before_refs] == [r["context_cell_ids"] for r in after_refs]
    assert [r["delta_pseudotime"] + .5 for r in before_refs] == [r["delta_pseudotime"] for r in after_refs]
    assert [r["target_tokens"] for r in before_refs] != [r["target_tokens"] for r in after_refs]


def test_donor_overlap_and_stale_metadata_fail_before_writing(prepared_source):
    args = prepared_source
    args.val_donor = "OM6"
    with pytest.raises(ValueError, match="must differ"):
        trajectory.prepare(args)
    assert not args.output.exists()
    args.val_donor = "OM9"
    source = at.manifest(args.source)
    source["source_obs_sha256"] = "invalid"
    at.dump(args.source / "manifest.json", source)
    with pytest.raises(ValueError, match="obs changed"):
        trajectory.prepare(args)
    assert not args.output.exists()
