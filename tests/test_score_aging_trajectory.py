"""Training-donor fitting, held-out leakage, and val-only matched scoring checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"torch_pipeline"))
import score_aging_direction as scoring


@pytest.fixture
def trajectory_data(tmp_path):
    data = tmp_path/"trajectory"
    data.mkdir()
    tokens = {"<pad>":0,"<mask>":1,"<bos>":2,"<eos>":3,
              "ENSG1":4,"ENSG2":5,"ENSG3":6,"ENSG4":7,"<boq>":8,"<eoq>":9}
    tokens.update({str(t):1000+t+40 for t in range(-40,41)})
    bank = [
        dict(cell_id="y0", donor="YM2", trajectory="SKM", time=0., tokens=[2,4,5,6,3]),
        dict(cell_id="y1", donor="YM2", trajectory="SKM", time=10., tokens=[2,5,4,6,3]),
        dict(cell_id="o0", donor="OM6", trajectory="SKM", time=20., tokens=[2,6,5,4,3]),
        dict(cell_id="o1", donor="OM6", trajectory="SKM", time=30., tokens=[2,7,6,5,3]),
    ]
    scoring.dump(data/"token_dictionary.json", tokens)
    m = dict(tokenizer_sha256=scoring.sha256(data/"token_dictionary.json"),
             cell_cap=5, time_scale=1., label_scalar=100., rope_factor=4.,
             attention_backend="sdpa", max_tokens=5,
             splits={"train":["YM2","OM6"],"val":["OM9"]},
             training_reference_bank="training_reference_bank.json",
             runtime_protocol="aging_stage2_runtime_v1",
             evaluation_splits={"val":{"context_donors":["YM2","OM6"],"query_donors":["OM9"]}})
    scoring.dump(data/"manifest.json", m)
    scoring.dump(data/"training_reference_bank.json", bank)
    refs = [dict(row=i, query_cell_id=f"held_out_{i}", donor="OM9", trajectory="SKM",
                 query_pseudotime=time, context_cell_ids=[r["cell_id"] for r in bank[:3]],
                 context_donors=[r["donor"] for r in bank[:3]],
                 context_pseudotimes=[r["time"] for r in bank[:3]],
                 context_tokens=[r["tokens"] for r in bank[:3]], target_tokens=bank[-1]["tokens"],
                 delta_pseudotime=time-20., represented_delta_pseudotime=time-20.)
            for i, time in enumerate((30., 15.))]
    scoring.dump(data/"val_references.json", refs)
    train_ref = dict(refs[0], query_cell_id="o1", donor="OM6", query_pseudotime=30.,
                     target_tokens=bank[-1]["tokens"])
    scoring.dump(data/"train_references.json", [train_ref])
    return data, m, tokens, bank, refs


def test_old_training_donor_is_used_and_heldout_donor_is_rejected(trajectory_data):
    _, _, _, bank, _ = trajectory_data
    fits, provenance = scoring.fit_young_bank(
        bank, {4,5,6,7}, max_genes=3, allowed_donors=["YM2","OM6"])
    assert fits["SKM"]["available"]
    assert provenance["SKM"]["donors"] == ["OM6","YM2"]
    assert provenance["SKM"]["n_cells"] == 4
    heldout = bank + [dict(bank[-1], cell_id="held_out_2", donor="OM9")]
    with pytest.raises(ValueError, match="training donors"):
        scoring.fit_young_bank(heldout, {4,5,6,7}, max_genes=3, allowed_donors=["YM2","OM6"])
    with pytest.raises(ValueError, match="every declared training donor"):
        scoring.fit_young_bank(bank[:2], {4,5,6,7}, max_genes=3, allowed_donors=["YM2","OM6"])


def test_multidonor_baselines_do_not_read_hidden_task_targets(trajectory_data):
    _, _, tokens, bank, refs = trajectory_data
    fits, _ = scoring.fit_young_bank(bank, {4,5,6,7}, max_genes=3,
                                     allowed_donors=["YM2","OM6"])
    def predict(ref):
        return scoring.predict_baselines(ref, fits, {4,5,6,7}, tokens, 1.,
                                          max_genes=3, fitted_prefix="training")
    original = predict(refs[0])
    changed_genes = predict(dict(refs[0], target_tokens=bank[0]["tokens"]))
    changed_time = predict(dict(refs[0], query_pseudotime=999., delta_pseudotime=979.,
                                represented_delta_pseudotime=-10.))
    assert original["nc"] == changed_genes["nc"]
    assert original["tbc"] == changed_time["tbc"]
    assert original["tbc"]["training_ridge_clock"] != changed_genes["tbc"]["training_ridge_clock"]


def test_val_only_schema_and_matched_comparison(trajectory_data, tmp_path):
    import torch

    data, m, tokens, bank, refs = trajectory_data
    _, _, _, read_refs = scoring.read_data(data)
    assert set(read_refs) == {"val"}
    baseline_dir = tmp_path/"baselines"
    scoring.baselines(SimpleNamespace(data=data, output=baseline_dir))
    provenance = json.loads((baseline_dir/"provenance.json").read_text())
    assert provenance["training_donors"] == ["YM2","OM6"]
    assert provenance["bank_filename"] == "training_reference_bank.json"
    assert provenance["fitted_annotations"]["SKM"]["n_cells"] == 4
    predictions = tmp_path/"predictions"
    for task in ("tbc","nc"):
        output = predictions/f"{task}_val"
        output.mkdir(parents=True)
        run = dict(task=task, split="val", limit=None if task=="tbc" else 1,
                   tokenizer_sha256=m["tokenizer_sha256"], rope_factor=4.,
                   attention_backend="sdpa", time_scale=1., label_scalar=100.,
                   references_sha256=scoring.sha256(data/"val_references.json"),
                   manifest_sha256=scoring.sha256(data/"manifest.json"), checkpoint="fixture",
                   runtime_protocol=m["runtime_protocol"])
        scoring.dump(output/"run.json", run)
        payload = ({"regression_preds":torch.tensor([r["delta_pseudotime"] for r in refs])}
                   if task=="tbc" else
                   [[dict(generated_tokens=torch.tensor([refs[0]["target_tokens"]]),
                          lengths=torch.tensor([5]), finished_naturally=torch.tensor([True]))]])
        torch.save(payload, output/"predictions__rank_0.pt")
    output = tmp_path/"comparison"
    scoring.compare(SimpleNamespace(data=data, baselines=baseline_dir,
                                     predictions=predictions, output=output))
    report = json.loads((output/"comparison_metrics.json").read_text())
    assert set(report["by_split"]) == {"val"}
    methods = report["by_split"]["val"]["methods"]
    assert methods["tbc"]["maxtoki_stage2"]["overall"]["mae"] == 0.
    assert methods["tbc"]["training_ridge_clock"]["overall"]["n_total"] == 2
    assert methods["nc"]["training_linear_trend"]["overall"]["n_total"] == 1
    assert methods["nc"]["maxtoki_stage2"]["overall"]["jaccard_at_100"] == 1.
    assert "young_ridge_clock" not in methods["tbc"]
    pair = report["by_split"]["val"]["paired_comparisons"]["nc"]["training_linear_trend"]
    assert pair["n_matched_available"] == 1
    assert pair["model"]["overall"]["n_total"] == pair["baseline"]["overall"]["n_total"]
    run_path = predictions/"tbc_val"/"run.json"
    valid_run = json.loads(run_path.read_text())
    scoring.dump(run_path, valid_run | {"runtime_protocol":"unfixed_runtime"})
    with pytest.raises(ValueError, match="runtime_protocol"):
        scoring.compare(SimpleNamespace(data=data, baselines=baseline_dir,
                                         predictions=predictions, output=tmp_path/"bad_runtime"))
    scoring.dump(run_path, valid_run)
    provenance["training_donors"] = ["YM2"]
    scoring.dump(baseline_dir/"provenance.json", provenance)
    with pytest.raises(ValueError, match="training donor mismatch"):
        scoring.compare(SimpleNamespace(data=data, baselines=baseline_dir,
                                         predictions=predictions, output=tmp_path/"mismatch"))


@pytest.mark.parametrize("mutation", ["query_in_context", "heldout_context", "bank_metadata"])
def test_context_leakage_and_metadata_are_rejected(trajectory_data, tmp_path, mutation):
    data, m, tokens, bank, refs = trajectory_data
    if mutation == "query_in_context":
        refs[0]["query_cell_id"] = refs[0]["context_cell_ids"][0]
    elif mutation == "heldout_context":
        refs[0]["context_donors"][0] = "OM9"
    else:
        refs[0]["context_tokens"][0] = [2,7,4,5,3]
    scoring.dump(data/"val_references.json", refs)
    with pytest.raises(ValueError, match="leakage|donor violates|differs from training"):
        scoring.baselines(SimpleNamespace(data=data, output=tmp_path/"invalid"))


def test_query_cannot_leak_into_fitting_bank(trajectory_data, tmp_path):
    data, _, _, bank, refs = trajectory_data
    bank.append(dict(bank[-1], cell_id=refs[0]["query_cell_id"], donor="OM9"))
    scoring.dump(data/"training_reference_bank.json", bank)
    with pytest.raises(ValueError, match="leaked into baseline fit"):
        scoring.baselines(SimpleNamespace(data=data, output=tmp_path/"invalid"))


def test_protocol_requires_heldout_disjoint_donors(trajectory_data):
    _, m, _, _, _ = trajectory_data
    m["evaluation_splits"]["val"]["query_donors"] = ["OM6"]
    with pytest.raises(ValueError, match="disjoint"):
        scoring.evaluation_protocol(m)


def test_fit_uses_exact_model_training_cells_not_extra_bank_cells(trajectory_data, tmp_path):
    data, m, _, bank, refs = trajectory_data
    extra = dict(bank[-1], cell_id="not_sampled_for_training", time=40.)
    bank.append(extra)
    scoring.dump(data/"training_reference_bank.json", bank)
    baseline_dir = tmp_path/"matched_bank"
    scoring.baselines(SimpleNamespace(data=data, output=baseline_dir))
    provenance = json.loads((baseline_dir/"provenance.json").read_text())
    audit = provenance["training_fit"]
    assert audit["n_fit_cells"] == 4 and audit["eligible_bank_cells"] == 5
    assert audit["source_split"] == "train"
    assert audit["training_references_sha256"] == scoring.sha256(data/"train_references.json")
    assert len(audit["fit_cell_ids_sha256"]) == 64
    assert provenance["fitted_annotations"]["SKM"]["cell_ids"] == ["o0","o1","y0","y1"]
    assert "not_sampled_for_training" not in provenance["fitted_annotations"]["SKM"]["cell_ids"]


def test_training_reference_cannot_import_heldout_cells(trajectory_data):
    data, m, _, bank, refs = trajectory_data
    training = json.loads((data/"train_references.json").read_text())
    training[0]["donor"] = "OM9"
    scoring.dump(data/"train_references.json", training)
    with pytest.raises(ValueError, match="Training reference cell differs"):
        scoring.fitting_bank(data, bank, scoring.evaluation_protocol(m))
