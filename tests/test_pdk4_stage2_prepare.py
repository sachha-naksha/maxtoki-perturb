"""Ensure a Stage 2 rerun preserves the historical intervention comparison."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "torch_pipeline"))
import aging_temporal as at
import prepare_pdk4_stage2 as pdk4


@pytest.fixture
def tokens():
    result = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3,
              pdk4.PDK4: 63, "ENSG_OTHER1": 64, "ENSG_OTHER2": 65, "ENSG_OTHER3": 66,
              "<boq>": 67, "<eoq>": 68}
    result.update({str(i): 1000 + i for i in range(-100, 101)})
    return result


def legacy_prompt(context, query, tokens):
    return [t for cell in context for t in cell] + [tokens["<boq>"]] + query[1:-1] + [
        tokens["<eoq>"], tokens["-1"]]


def test_temporal_grammar_preserves_rank_and_keeps_query_target_out(tokens):
    context = [[2, 64, 3], [2, 65, 3], [2, 66, 3]]
    query, perturbed = [2, 63, 64, 65, 3], [2, 64, 65, 63, 3]
    numeric = at.dictionary_check(tokens, {})
    baseline, inhibited, labels = pdk4.timed_pair(
        context, [1., 57., 100.], query, perturbed, 33., tokens, numeric, 1., 8192)
    assert baseline == [2, 64, 3, tokens["56"], 2, 65, 3, tokens["43"],
                        2, 66, 3, 67, 63, 64, 65, 68, tokens["0"]]
    assert inhibited[12:15] == [64, 65, 63]
    assert labels["delta_pseudotime"] == -67.
    changed_query_time, _, _ = pdk4.timed_pair(
        context, [1., 57., 100.], query, perturbed, 99., tokens, numeric, 1., 8192)
    assert changed_query_time == baseline
    assert pdk4.parse_legacy_tbc(legacy_prompt(context, query, tokens), tokens) == (context, query)


def test_inhibition_moves_instead_of_deleting_and_absent_is_unchanged(tokens):
    assert pdk4.inhibit_query([2, 63, 64, 65, 3], 63, 2, 3) == [2, 64, 65, 63, 3]
    assert pdk4.inhibit_query([2, 64, 65, 3], 63, 2, 3) == [2, 64, 65, 3]
    assert pdk4.inhibit_query([2, 64, 63, 3], 63, 2, 3) == [2, 64, 63, 3]


def test_derived_pair_inhibits_other_gene_inside_stored_baseline_prompt(tokens):
    context = [[2, 64, 3], [2, 65, 3], [2, 66, 3]]
    stored = dict(input_ids=legacy_prompt(context, [2, 63, 65, 64, 3], tokens),
                  gene_ensembl=pdk4.PDK4, gene_token=63, gene_present_in_query=True,
                  direction="inhibit", apply_to="query", n_context_cells=3)
    base, pert = pdk4.derive_legacy_pair(stored, tokens, "ENSG_OTHER2")
    assert base["input_ids"] == stored["input_ids"]
    assert pert["input_ids"][-5:-2] == [63, 64, 65]
    assert base["gene_token"] == pert["gene_token"] == 65 and base["gene_present_in_query"]
    absent, _ = pdk4.derive_legacy_pair(stored, tokens, "ENSG_OTHER3")
    assert not absent["gene_present_in_query"]
    assert pdk4.validate_pair(*pdk4.derive_legacy_pair(stored, tokens, "ENSG_OTHER2"),
        *pdk4.derive_legacy_pair(stored, tokens, "ENSG_OTHER2"),
        {v: v for v in tokens.values()}, tokens, 0, gene="ENSG_OTHER2")[-1]


def test_token_name_mapping_rejects_missing_or_ambiguous_ids(tokens):
    legacy = {key: value + 30000 for key, value in tokens.items()}
    remap = pdk4.token_remapping(legacy, tokens)
    assert pdk4.remap_sequence([30002, 30063, 30003], remap) == [2, 63, 3]
    with pytest.raises(ValueError, match="Unknown legacy"):
        pdk4.remap_sequence([-50], remap)
    with pytest.raises(ValueError, match="Ambiguous"):
        pdk4.token_remapping({"a": 1, "b": 1}, {})
    with pytest.raises(ValueError, match="absent"):
        pdk4.token_remapping({"ENSG_missing": 12}, tokens)


def test_budget_rejects_without_truncating_original_genes(tokens):
    context = [[2, 64, 3], [2, 65, 3], [2, 66, 3]]
    with pytest.raises(ValueError, match="exceed prompt budget"):
        pdk4.timed_pair(context, [1., 57., 100.], [2, 63, 64, 3], [2, 64, 63, 3],
                       70., tokens, at.dictionary_check(tokens, {}), 1., 12)


def test_parser_rejects_corrupt_or_already_timed_contexts(tokens):
    context = [[2, 64, 3], [2, 65, 3], [2, 66, 3]]
    sequence = legacy_prompt(context, [2, 63, 64, 3], tokens)
    sequence.insert(3, tokens["56"])
    with pytest.raises(ValueError, match="contiguous"):
        pdk4.parse_legacy_tbc(sequence, tokens)
    sequence = legacy_prompt(context, [2, 63, 63, 3], tokens)
    with pytest.raises(ValueError, match="Duplicate query"):
        pdk4.parse_legacy_tbc(sequence, tokens)


@pytest.fixture
def historical_fixture(tmp_path, monkeypatch, tokens):
    import anndata as ad
    import numpy as np
    import pandas as pd

    monkeypatch.setattr(pdk4, "compute", lambda: None)
    monkeypatch.setattr(pdk4, "EXPECTED_ROWS", 4)
    monkeypatch.setitem(pdk4.GENES, "PDK4", dict(pdk4.GENES["PDK4"], n_present=2))
    contexts = [[2, 64, 3], [2, 65, 3], [2, 66, 3]]
    query_ids = ["OM6a", "OM9a", "OM6b", "OM9b"]
    donors = ["OM6", "OM9", "OM6", "OM9"]
    query_times = [33., 100., 45., 60.]
    context_rows = [dict(Pseudotime=time, sample="YM2", Annotation="A")
                    for time in pdk4.EXPECTED_CONTEXT_TIMES]
    query_rows = [dict(Pseudotime=time, sample=donor, Annotation="B")
                  for time, donor in zip(query_times, donors)]
    with pd.option_context("future.infer_string", False):
        obs = pd.DataFrame(context_rows + query_rows, index=pdk4.EXPECTED_CONTEXT_IDS + query_ids)
        h5ad = tmp_path / "obs.h5ad"
        ad.AnnData(np.ones((len(obs), 1)), obs=obs).write_h5ad(h5ad)
    source_obs = ad.read_h5ad(h5ad, backed="r")
    try:
        obs_hash = hashlib.sha256(
            source_obs.obs[["Pseudotime", "sample", "Annotation"]].to_csv().encode()).hexdigest()
    finally:
        source_obs.file.close()
    training = tmp_path / "training"
    training.mkdir()
    at.dump(training / "token_dictionary.json", tokens)
    model = dict(splits={"train": ["YM2", "OM6"], "val": ["OM9"]},
        seq_length=16384, time_scale=1., label_scalar=200.,
        pseudotime_col="Pseudotime", trajectory_col="Annotation",
        source_h5ad=str(h5ad), source_obs_sha256=obs_hash,
        tokenizer_sha256=at.digest(training / "token_dictionary.json"),
        runtime_protocol="aging_stage2_runtime_v1", numeric_expectation_normalized=True,
        nc_full_answer_loss=True)
    at.dump(training / "manifest.json", model)
    legacy = {key: value + 30000 for key, value in tokens.items()}
    legacy_path = tmp_path / "legacy.json"
    at.dump(legacy_path, legacy)
    old_id = {value: legacy[name] for name, value in tokens.items()}
    original, corrected = tmp_path / "original", tmp_path / "corrected"
    original.mkdir()
    corrected.mkdir()
    at.dump(original / "spec.resolved.json", {})
    row_manifest, old_baseline, old_perturbed, baseline, perturbed = [], [], [], [], []
    for row, (cell_id, donor, time) in enumerate(zip(query_ids, donors, query_times)):
        query = [2, 63, 64, 65, 3] if row % 2 == 0 else [2, 64, 65, 3]
        inhibited = pdk4.inhibit_query(query, 63, 2, 3)
        metadata = dict(row_index=row, cell_id=cell_id, group=donor, query_pseudotime=time,
            context_pseudotimes=pdk4.EXPECTED_CONTEXT_TIMES,
            context_cell_ids=pdk4.EXPECTED_CONTEXT_IDS, n_context_cells=3,
            gene_token=63, gene_ensembl=pdk4.PDK4, direction="inhibit",
            apply_to="query", gene_present_in_query=row % 2 == 0,
            task_type="time_between_cells")
        base = dict(metadata, input_ids=legacy_prompt(contexts, query, tokens), condition="baseline")
        pert = dict(metadata, input_ids=legacy_prompt(contexts, inhibited, tokens), condition="perturbed")
        baseline.append(base)
        perturbed.append(pert)
        for record, target in ((base, old_baseline), (pert, old_perturbed)):
            record = deepcopy(record)
            record["input_ids"] = [old_id[token] for token in record["input_ids"]]
            record["gene_token"] = old_id[63]
            del record["row_index"], record["task_type"]
            target.append(record)
        row_manifest.append({key: base[key] for key in ("row_index", "cell_id", "group",
            "query_pseudotime", "context_cell_ids", "gene_present_in_query")})
    for directory, base, pert in ((original, old_baseline, old_perturbed), (corrected, baseline, perturbed)):
        pdk4.save_dataset(directory / "baseline.dataset", base)
        pdk4.save_dataset(directory / "perturbed.dataset", pert)
    at.dump(corrected / "row_manifest.json", dict(n_rows=4, rows=row_manifest))
    return SimpleNamespace(original=original, corrected=corrected, training_data=training,
        output=tmp_path / "prepared", legacy_dictionary=legacy_path, gene="PDK4",
        records=(old_baseline, old_perturbed, baseline, perturbed), tokens=tokens, model=model)


def test_pair_validation_detects_metadata_drift_and_wrong_intervention(historical_fixture):
    fixture = historical_fixture
    values = [deepcopy(records[0]) for records in fixture.records]
    remap = pdk4.token_remapping(json.loads(fixture.legacy_dictionary.read_text()), fixture.tokens)
    assert pdk4.validate_pair(*values, remap, fixture.tokens, 0)[-1]
    values[3]["gene_present_in_query"] = False
    with pytest.raises(ValueError, match="metadata mismatch"):
        pdk4.validate_pair(*values, remap, fixture.tokens, 0)
    values = [deepcopy(records[0]) for records in fixture.records]
    # Even matched original/current deletion is rejected: this is inhibition by ranking.
    for record, gene in ((values[1], 30063), (values[3], 63)):
        query_start = record["input_ids"].index(
            30067 if record is values[1] else 67)
        record["input_ids"].pop(record["input_ids"].index(gene, query_start))
    with pytest.raises(ValueError, match="only move"):
        pdk4.validate_pair(*values, remap, fixture.tokens, 0)


def test_prepare_atomic_paired_interleaving_provenance_and_repeat(historical_fixture):
    from datasets import load_from_disk
    fixture = historical_fixture
    pdk4.prepare(fixture)
    output = fixture.output
    manifest = json.loads((output / "manifest.json").read_text())
    refs = json.loads((output / "references.json").read_text())
    baseline, perturbed, paired, repeat = [
        load_from_disk(str(output / f"{name}.dataset"))
        for name in ("baseline", "perturbed", "paired", "repeat")]
    assert len(refs) == 4 and len(paired) == len(repeat) == 8
    assert manifest["splits"] == fixture.model["splits"]
    assert manifest["seq_length"] == 16384
    assert manifest["prompt_token_budget"] == 8192
    assert manifest["perturbation_protocol"]["context_intervals"] == [56., 43.]
    assert manifest["perturbation_protocol"]["query_time_input"] is False
    assert (output / "token_dictionary.json").read_bytes() == (
        fixture.training_data / "token_dictionary.json").read_bytes()
    assert manifest["prepared_pair"]["references_sha256"] == at.digest(output / "references.json")
    for row, ref in enumerate(refs):
        assert paired[2 * row] == baseline[row]
        assert paired[2 * row + 1] == perturbed[row]
        assert ref["raw_target_delta"] == ref["query_pseudotime"] - 100.
        assert ref["context_intervals"] == [56., 43.]
        assert baseline[row]["input_ids"][-1] == fixture.tokens["0"]
        if not ref["gene_present"]:
            assert baseline[row] == perturbed[row]
    for name in ("baseline", "perturbed", "paired", "repeat"):
        assert pdk4.dataset_digest(output / f"{name}.dataset")[0] == (
            manifest["prepared_pair"]["dataset_sha256"][name])
    with pytest.raises(FileExistsError):
        pdk4.prepare(fixture)


def test_prepare_without_corrected_rerun_remaps_legacy_rows(historical_fixture):
    from datasets import load_from_disk
    fixture = historical_fixture
    pdk4.prepare(fixture)
    with_corrected = load_from_disk(str(fixture.output / "paired.dataset"))["input_ids"]
    fixture.corrected = None
    fixture.output = fixture.output.parent / "prepared_no_corrected"
    pdk4.prepare(fixture)
    assert load_from_disk(str(fixture.output / "paired.dataset"))["input_ids"] == with_corrected
    manifest = json.loads((fixture.output / "manifest.json").read_text())
    assert manifest["prepared_pair"]["corrected"] is None
    assert manifest["prepared_pair"]["corrected_row_manifest_sha256"] is None


def test_prepare_rejects_authoritative_obs_or_source_row_manifest_change(historical_fixture):
    fixture = historical_fixture
    row_manifest = json.loads((fixture.corrected / "row_manifest.json").read_text())
    row_manifest["rows"][0]["gene_present_in_query"] = False
    at.dump(fixture.corrected / "row_manifest.json", row_manifest)
    with pytest.raises(ValueError, match="row manifest"):
        pdk4.prepare(fixture)
    assert not fixture.output.exists()


def test_failure_during_save_removes_temporary_outputs(historical_fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated interrupted dataset write")
    monkeypatch.setattr(pdk4, "save_dataset", fail)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        pdk4.prepare(historical_fixture)
    assert not historical_fixture.output.exists()
    assert not list(historical_fixture.output.parent.glob("prepared.preparing.*"))


def test_repeat_selects_present_and_absent_in_stable_original_order():
    refs = [dict(row=i, gene_present=i % 3 != 0) for i in range(30)]
    rows = pdk4.repeat_query_indices(refs)
    assert rows == sorted(rows) and len(rows) == 12
    assert sum(refs[row]["gene_present"] for row in rows) == 8
    assert sum(not refs[row]["gene_present"] for row in rows) == 4
