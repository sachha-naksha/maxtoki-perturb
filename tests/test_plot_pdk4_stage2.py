"""Scientific pairing and uncertainty checks for the PDK4 rerun figure."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"/"torch_pipeline"))
import plot_pdk4_stage2 as plotting


def row(i, *, donor="OM6", delta=-2., present=True, cell=None, context=None):
    return dict(row=i,query_cell_id=cell or f"{donor}_{i}",donor=donor,
                gene_present=present,control_prediction=10.,
                perturbed_prediction=10.+delta,target_delta_pseudotime=12.,
                context_cell_ids=context or ["y0","y1","y2"])


def rows_fixture():
    return plotting.validate_rows([
        row(0,delta=-1.),row(1,delta=-3.),
        row(2,donor="OM9",delta=0.,present=False),
        row(3,donor="OM9",delta=-4.)])


def test_complete_pairs_and_finite_values_required():
    original=rows_fixture()
    with pytest.raises(ValueError,match="Duplicate paired row"):
        plotting.validate_rows([original[0],original[0]])
    with pytest.raises(ValueError,match="Nonfinite paired"):
        plotting.validate_rows([dict(original[0],perturbed_prediction=float("nan"))])
    with pytest.raises(ValueError,match="Stored delta differs"):
        plotting.validate_rows([dict(original[0],delta_t=123.)])
    with pytest.raises(ValueError,match="boolean"):
        plotting.validate_rows([dict(original[0],gene_present="False")])
    with pytest.raises(ValueError,match="Conflicting metadata"):
        plotting.validate_rows([original[0],dict(original[1],query_cell_id=original[0]["query_cell_id"],
                                                 donor="OM9")])


def test_checkpoints_must_use_identical_ordered_pairs():
    rows=rows_fixture()
    changed=[dict(r,perturbed_prediction=r["perturbed_prediction"]+1.,delta_t=r["delta_t"]+1.)
             for r in rows]
    plotting.ensure_matched_runs(rows,changed)
    with pytest.raises(ValueError,match="alignment"):
        plotting.ensure_matched_runs(rows,list(reversed(changed)))
    with pytest.raises(ValueError,match="metadata"):
        plotting.ensure_matched_runs(rows,[dict(changed[0],context_cell_ids=["x"]),*changed[1:]])
    with pytest.raises(ValueError,match="different numbers"):
        plotting.ensure_matched_runs(rows,changed[:-1])


def test_unchanged_prompts_have_exact_zero_response_and_interval():
    rows=plotting.validate_rows([row(i,delta=0.,present=False,
                                      donor="OM6" if i<3 else "OM9") for i in range(8)])
    report=plotting.summarize_run(rows,n_bootstrap=200)
    for group in ("pooled","OM6","OM9"):
        absent=report[group]["gene_absent"]
        assert absent["mean_delta_t"] == absent["ci95_low"] == absent["ci95_high"] == 0.
        assert absent["gene_absent_max_absolute_delta"] == 0.
        assert absent["fraction_zero"] == 1.
        assert report[group]["gene_present"]["n_rows"] == 0


def test_repeated_queries_are_clusters_not_independent_cells():
    rows=rows_fixture()
    # Replicating every observation for the same cells cannot narrow the interval.
    repeated=[dict(r,row=j) for j,r in enumerate(rows*5)]
    once=plotting.summarize(rows,n_bootstrap=400,seed=23)
    five=plotting.summarize(repeated,n_bootstrap=400,seed=23)
    assert five["n_rows"] == 5*once["n_rows"]
    assert once["n_unique_queries"] == five["n_unique_queries"] == 4
    assert np.allclose([once["ci95_low"],once["ci95_high"]],
                       [five["ci95_low"],five["ci95_high"]])
    assert once["mean_delta_t"] == five["mean_delta_t"]


def test_pooled_bootstrap_fixes_observed_donor_weights():
    rows=plotting.validate_rows([row(i,delta=-10.) for i in range(3)] +
                               [row(3,donor="OM9",delta=30.)])
    values=plotting.summarize(rows,n_bootstrap=200)
    # The observed 3:1 mixture has mean zero. Resampling donor identities would
    # invent a nonzero CI despite constant responses within both donors.
    assert values["mean_delta_t"] == 0.
    assert values["ci95_low"] == values["ci95_high"] == 0.


def test_historical_raw_and_scale_adjusted_agreement_are_separate():
    rows=rows_fixture()
    historical=[dict(r,control_prediction=r["control_prediction"]*200,
                     perturbed_prediction=r["perturbed_prediction"]*200,
                     delta_t=r["delta_t"]*200) for r in rows]
    report=plotting.build_report([
        ("Original",historical,{"historical":True}),
        ("Best step 100",rows,{"historical":False}),
        ("Last step 500",rows,{"historical":False})],n_bootstrap=200)
    raw=report["runs"]["Original"]["statistics"]["pooled"]["all"]["mean_delta_t"]
    scaled=report["historical_scaled"]["statistics"]["pooled"]["all"]["mean_delta_t"]
    current=report["runs"]["Best step 100"]["statistics"]["pooled"]["all"]["mean_delta_t"]
    assert raw == 200*current
    assert scaled == current
    pairs=report["descriptive_comparisons"]
    assert pairs["Best step 100 vs Original / 200"]["delta_mae"] == 0.
    assert pairs["Best step 100 vs Original"]["delta_mae"] > 0.
    assert pairs["Best step 100 vs Original"]["n_nonzero_both"] == 3
    assert pairs["Best step 100 vs Original"]["nonzero_direction_agreement"] == 1.
    assert report["runs"]["Best step 100"]["statistics"]["pooled"]["all"]["control_ground_truth"]["mae"] == 2.


def test_new_run_reader_uses_paired_prediction_schema(tmp_path):
    rows=rows_fixture()
    (tmp_path/"rows.json").write_text(json.dumps(rows))
    (tmp_path/"run.json").write_text(json.dumps({"checkpoint":"saved-step100"}))
    got,provenance=plotting.read_run(tmp_path)
    assert got == rows
    assert provenance["run"]["checkpoint"] == "saved-step100"
    assert len(provenance["rows_sha256"]) == 64


def test_deliverable_csv_and_figures_include_primary_and_historical(tmp_path):
    rows=rows_fixture()
    old=[dict(r,control_prediction=r["control_prediction"]*200,
              perturbed_prediction=r["perturbed_prediction"]*200,
              delta_t=r["delta_t"]*200) for r in rows]
    report=plotting.build_report([
        ("Historical original",old,{"historical":True}),
        ("Stage 2 best step 100",rows,{"historical":False}),
        ("Stage 2 last step 500",rows,{"historical":False})],n_bootstrap=100)
    plotting.write_csv(report,tmp_path/"bar_statistics.csv")
    plotting.render(report,tmp_path)
    csv_text=(tmp_path/"bar_statistics.csv").read_text()
    assert "Historical original / 200" in csv_text
    assert "Stage 2 best step 100" in csv_text
    for name in ("pdk4_inhibition_tbc","pdk4_inhibition_tbc_gene_present",
                 "pdk4_inhibition_tbc_gene_absent","pdk4_historical_comparison"):
        for extension in ("png","pdf","svg"):
            assert (tmp_path/f"{name}.{extension}").stat().st_size > 100


def test_prediction_collapse_is_visible_against_variable_ground_truth():
    rows=plotting.validate_rows([
        dict(row(0,delta=-.0001),control_prediction=-44.,
             perturbed_prediction=-44.0001,target_delta_pseudotime=-10.),
        dict(row(1,donor="OM9",delta=.0001),control_prediction=-44.,
             perturbed_prediction=-43.9999,target_delta_pseudotime=10.)])
    report=plotting.summarize(rows,n_bootstrap=100)
    assert report["control_prediction_mean"] == -44.
    assert report["control_prediction_std"] == 0.
    assert report["control_prediction_min"] == report["control_prediction_max"] == -44.
    assert report["target_delta_pseudotime_mean"] == 0.
    assert report["target_delta_pseudotime_std"] == 10.
    assert report["target_delta_pseudotime_min"] == -10.
    assert report["target_delta_pseudotime_max"] == 10.
    assert report["perturbed_prediction_mean"] == -44.
    assert np.isclose(report["perturbed_prediction_std"],.0001)
    assert report["standard_deviation_convention"] == "population; ddof=0"
