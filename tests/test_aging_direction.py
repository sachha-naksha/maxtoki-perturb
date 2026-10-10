"""Directed evaluation uses young context without consulting held-out query values."""
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
import prepare_aging_direction as direction


def test_context_covers_young_range_without_duplicates():
    pool = [dict(cell_id=f"young-{i:02}", time=float(i), tokens=[2, 4, 3])
            for i in range(12)]
    selected = direction.sample_context(pool, random.Random(12))
    assert len({cell["cell_id"] for cell in selected}) == 3
    assert [int(cell["time"]) // 4 for cell in selected] == [0, 1, 2]
    assert selected == direction.sample_context(list(reversed(pool)), random.Random(12))
    # Ties may prevent three distinct times, but must not duplicate cells.
    for cell in pool:
        cell["time"] = 0.0
    assert len({cell["cell_id"] for cell in direction.sample_context(pool, random.Random(12))}) == 3


def test_early_context_bound_and_minimal_expansion():
    pool = [dict(cell_id=f"young-{i:03}", time=float(i)) for i in range(100)]
    selected, audit = direction.early_context_pool(pool, .25)
    assert max(cell["time"] for cell in selected) == 24
    assert audit["requested_threshold"] == 24.75
    assert audit["context_pool_cells"] == 25
    assert not audit["expanded_threshold"]
    ties = [dict(cell_id=str(i), time=time) for i, time in enumerate([0, 0, 0, 1, 2, 9])]
    selected, audit = direction.early_context_pool(ties, .25)
    assert audit["requested_threshold"] == 0
    assert audit["actual_threshold"] == 2
    assert {cell["time"] for cell in selected} == {0, 1, 2}
    assert audit["expanded_threshold"]


def test_annotation_balanced_query_prefix_and_unique_queries():
    groups = {annotation: [dict(cell_id=f"{annotation}-{i}") for i in range(8)]
              for annotation in ("A", "B", "C")}
    result = direction.stratified_queries(groups, 20, random.Random(9))
    assert len({cell["cell_id"] for cell in result}) == 20
    assert [cell["cell_id"][0] for cell in result[:6]] == ["A", "B", "C", "A", "B", "C"]
    with pytest.raises(ValueError, match="exceeds"):
        direction.stratified_queries(groups, 25, random.Random(9))


@pytest.fixture
def source(tmp_path):
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
                time = float(i if donor == "YM2" else i - 1)
                if donor == "YM2" and annotation == "A" and i == 4:
                    time = np.nan
                cell_id = f"{donor}_{annotation}_{i}"
                records.append(dict(cell_id=cell_id, input_ids=[2, 4 + i % 4, 3],
                                    Pseudotime=time, sample=donor))
                rows.append(dict(sample=donor, Annotation=annotation, Pseudotime=time))
    h5ad_path = tmp_path / "source.h5ad"
    with pd.option_context("future.infer_string", False):
        obs = pd.DataFrame(rows, index=[record["cell_id"] for record in records])
        ad.AnnData(np.ones((len(rows), 4)), obs=obs).write_h5ad(h5ad_path)
    cell_path = tmp_path / "cells.dataset"
    Dataset(pa.Table.from_pylist(records), fingerprint="directedfixture").save_to_disk(str(cell_path))
    prepared = tmp_path / "original"
    prepared.mkdir()
    at.dump(prepared / "token_dictionary.json", tokens)
    columns = ["Pseudotime", "sample", "Annotation"]
    obs = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_hash = hashlib.sha256(obs.obs[columns].to_csv().encode()).hexdigest()
    finally:
        obs.file.close()
    model = dict(
        units="pseudotime", splits={"train": ["YM2"], "val": ["OM6"], "test": ["OM9"]},
        pseudotime_col="Pseudotime", trajectory_col="Annotation",
        time_scale=1.0, label_scalar=200.0, seq_length=40, max_tokens=4,
        cell_cap=4, seed=42, tokenizer_sha256=at.digest(prepared / "token_dictionary.json"),
        source_h5ad=str(h5ad_path), source_cells=str(cell_path), source_obs_sha256=obs_hash,
        rope_factor=4.0, old_context_len=4096, attention_backend="sdpa",
        counts={"train": {"trajectories": 2}, "val": {"trajectories": 2}, "test": {"trajectories": 2}},
    )
    at.dump(prepared / "manifest.json", model)
    return SimpleNamespace(source=prepared, output=tmp_path / "directed",
                           queries_per_donor=6, seed=43)


def test_prepare_preserves_signed_targets_and_checkpoint_compatibility(source):
    from datasets import load_from_disk

    direction.prepare(source)
    manifest = at.manifest(source.output)
    original = at.manifest(source.source)
    assert manifest["model_training_splits"] == original["splits"]
    assert manifest["evaluation_splits"]["test"] == {
        "context_donors": ["YM2"], "query_donors": ["OM9"]}
    assert (source.output / "token_dictionary.json").read_bytes() == (
        source.source / "token_dictionary.json").read_bytes()
    checkpoint = source.output / "fixture_checkpoint"
    checkpoint.mkdir()
    at.dump(checkpoint / "aging_temporal_manifest.json",
            original | dict(tasks=["tbc", "nc"], head="headless"))
    at.checkpoint_check(checkpoint, manifest)
    for split, donor in (("val", "OM6"), ("test", "OM9")):
        refs = json.loads((source.output / f"{split}_references.json").read_text())
        assert len(refs) == len({r["query_cell_id"] for r in refs}) == 6
        assert [r["trajectory"] for r in refs] == ["A", "B", "A", "B", "A", "B"]
        assert all(r["donor"] == donor and r["context_donors"] == ["YM2"] * 3 for r in refs)
        assert any(r["delta_pseudotime"] < 0 for r in refs)
        for ref in refs:
            assert ref["delta_pseudotime"] == ref["query_pseudotime"] - ref["context_pseudotimes"][-1]
            assert ref["context_pseudotimes"] == sorted(ref["context_pseudotimes"])
            assert len(set(ref["context_cell_ids"])) == 3
        assert len(load_from_disk(str(source.output / f"{split}_nc.dataset"))) == 6
    bank = json.loads((source.output / "young_reference_bank.json").read_text())
    assert len(bank) == 9 and all(cell["donor"] == "YM2" for cell in bank)
    audit = json.loads((source.output / "audit.json").read_text())
    assert audit["excluded_by_donor"] == {"YM2": 1}
    assert audit["splits"]["test"]["signed_pseudotime_direction_counts"]["negative"] > 0
    with pytest.raises(FileExistsError):
        direction.prepare(source)


def test_query_values_do_not_select_context_and_nc_prompt_excludes_target_genes(source):
    import anndata as ad
    import pyarrow as pa
    from datasets import Dataset, load_from_disk

    direction.prepare(source)
    before = json.loads((source.output / "test_references.json").read_text())
    nc_before = list(load_from_disk(str(source.output / "test_nc.dataset")))
    model = at.manifest(source.source)
    cells = list(load_from_disk(model["source_cells"]))
    # Change every old query's expression while preserving all selection inputs.
    for cell in cells:
        if cell["sample"] != "YM2":
            cell["input_ids"] = [2, 7 if cell["input_ids"][1] != 7 else 4, 3]
    revised = source.source.parent / "revised_cells.dataset"
    Dataset(pa.Table.from_pylist(cells), fingerprint="revised-expression").save_to_disk(str(revised))
    model["source_cells"] = str(revised)
    at.dump(source.source / "manifest.json", model)
    source.output = source.source.parent / "changed_expression"
    direction.prepare(source)
    after = json.loads((source.output / "test_references.json").read_text())
    assert [r["target_tokens"] for r in before] != [r["target_tokens"] for r in after]
    assert [r["context_cell_ids"] for r in before] == [r["context_cell_ids"] for r in after]
    assert nc_before == list(load_from_disk(str(source.output / "test_nc.dataset")))

    # Change every old pseudotime and update authoritative metadata provenance.
    import pandas as pd
    with pd.option_context("future.infer_string", False):
        data = ad.read_h5ad(model["source_h5ad"])
        old = data.obs["sample"].astype(str) != "YM2"
        data.obs.loc[old, "Pseudotime"] += .5
        changed_h5ad = source.source.parent / "changed_times.h5ad"
        data.write_h5ad(changed_h5ad)
    for cell in cells:
        if cell["sample"] != "YM2":
            cell["Pseudotime"] += .5
    changed_cells = source.source.parent / "changed_times.dataset"
    Dataset(pa.Table.from_pylist(cells), fingerprint="revised-times").save_to_disk(str(changed_cells))
    model["source_cells"] = str(changed_cells)
    model["source_h5ad"] = str(changed_h5ad)
    reread = ad.read_h5ad(changed_h5ad)
    model["source_obs_sha256"] = hashlib.sha256(
        reread.obs[["Pseudotime", "sample", "Annotation"]].to_csv().encode()).hexdigest()
    at.dump(source.source / "manifest.json", model)
    source.output = source.source.parent / "changed_times"
    direction.prepare(source)
    changed = json.loads((source.output / "test_references.json").read_text())
    assert [r["query_cell_id"] for r in before] == [r["query_cell_id"] for r in changed]
    assert [r["context_cell_ids"] for r in before] == [r["context_cell_ids"] for r in changed]
    assert [r["delta_pseudotime"] + .5 for r in before] == [r["delta_pseudotime"] for r in changed]


def test_changed_source_obs_rejected_before_output(source):
    model = at.manifest(source.source)
    model["source_obs_sha256"] = "incorrect"
    at.dump(source.source / "manifest.json", model)
    with pytest.raises(ValueError, match="obs changed"):
        direction.prepare(source)
    assert not source.output.exists()
