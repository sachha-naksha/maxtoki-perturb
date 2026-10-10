"""Leakage, scoring and matched-row integration checks for directed aging evaluation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"torch_pipeline"))
import score_aging_direction as scoring


@pytest.fixture
def vocabulary():
    tokens={"<pad>":0,"<mask>":1,"<bos>":2,"<eos>":3,
            "ENSG1":4,"ENSG2":5,"ENSG3":6,"ENSG4":7,"<boq>":8,"<eoq>":9}
    tokens.update({str(t):1000+t+40 for t in range(-40,41)})
    return tokens


@pytest.fixture
def bank():
    return [
        dict(cell_id="y0",donor="YM2",trajectory="SKM",time=0.,tokens=[2,4,5,6,3]),
        dict(cell_id="y1",donor="YM2",trajectory="SKM",time=10.,tokens=[2,5,4,6,3]),
        dict(cell_id="y2",donor="YM2",trajectory="SKM",time=20.,tokens=[2,6,5,4,3]),
        dict(cell_id="y3",donor="YM2",trajectory="SKM",time=30.,tokens=[2,7,6,5,3]),
    ]


def test_old_donors_cannot_enter_fit(bank):
    bank[-1]=dict(bank[-1],donor="OM6")
    with pytest.raises(ValueError,match="YM2"):
        scoring.fit_young_bank(bank,{4,5,6,7},max_genes=3)


def test_bank_inputs_are_task_specific_and_origin_invariant(bank):
    fits,_=scoring.fit_young_bank(bank,{4,5,6,7},max_genes=3)
    context=[row["tokens"] for row in bank[:3]]
    args=(fits["SKM"],context,[-20.,-10.,0.])
    tbc,nc=scoring.bank_predict(*args,bank[3]["tokens"],10.)
    changed_tbc,same_nc=scoring.bank_predict(*args,bank[0]["tokens"],10.)
    same_tbc,changed_nc=scoring.bank_predict(*args,bank[3]["tokens"],-10.)
    assert tbc["prediction"] != pytest.approx(changed_tbc["prediction"])
    assert nc==same_nc  # NC never consumes query expression.
    assert tbc==same_tbc  # TBC never consumes the requested/ground-truth interval.
    assert nc["diagnostics"]["estimated_absolute_query_time"] != changed_nc["diagnostics"]["estimated_absolute_query_time"]
    shifted=[dict(row,time=row["time"]+100.) for row in bank]
    shifted_fits,_=scoring.fit_young_bank(shifted,{4,5,6,7},max_genes=3)
    shifted_tbc,shifted_nc=scoring.bank_predict(shifted_fits["SKM"],context,[-20.,-10.,0.],bank[3]["tokens"],10.)
    assert shifted_tbc["prediction"]==pytest.approx(tbc["prediction"])
    assert shifted_nc["gene_tokens"]==nc["gene_tokens"]


def test_degenerate_fit_is_unavailable_not_zero(bank):
    same=[dict(row,time=1.) for row in bank]
    fits,_=scoring.fit_young_bank(same,{4,5,6,7},max_genes=3)
    tbc,nc=scoring.bank_predict(fits["SKM"],[r["tokens"] for r in bank[:3]],
                              [-2.,-1.,0.],bank[-1]["tokens"],2.)
    assert not tbc["available"] and tbc["prediction"] is None
    assert not nc["available"] and nc["gene_tokens"]==[]


def test_prompt_time_quantization_does_not_use_absolute_offset(vocabulary):
    ref=dict(context_pseudotimes=[100.2,105.6,109.2],context_tokens=[[2,4,3]]*3,
             represented_delta_pseudotime=7.)
    times,delta=scoring.prompt_times(ref,vocabulary,1.)
    assert times==[-9.,-4.,0.]
    assert delta==7.


def test_metric_is_true_shared_rank_spearman():
    metrics=scoring.nc_metrics(["a","b","c","d","e","f"],["c","x","y","z","a","b"])
    assert metrics["spearman_shared"]==pytest.approx(-.5)
    assert metrics["jaccard_at_100"]==pytest.approx(3/9)


def test_unavailable_rows_are_counted_and_not_scored_as_zero():
    records=[
        dict(available=True,query_cell_id="a",predicted_delta=3.,target_delta=1.,reason=None),
        dict(available=False,query_cell_id="b",reason="no_context_time_variation"),
    ]
    result=scoring.aggregate(records,"tbc")
    assert result["n_total"]==2 and result["n_available"]==1
    assert result["coverage"]==.5 and result["mae"]==2.
    assert result["unavailable_reasons"]=={"no_context_time_variation":1}
    empty=scoring.aggregate(records[1:],"tbc")
    assert empty["mae"] is None and empty["mse"] is None


def test_baseline_and_model_comparison_uses_same_limited_rows(tmp_path,vocabulary,bank):
    import torch

    data=tmp_path/"data"
    data.mkdir()
    scoring.dump(data/"token_dictionary.json",vocabulary)
    m=dict(tokenizer_sha256=scoring.sha256(data/"token_dictionary.json"),
           cell_cap=5,time_scale=1.,label_scalar=100.,rope_factor=4.,
           attention_backend="sdpa",max_tokens=5)
    scoring.dump(data/"manifest.json",m)
    scoring.dump(data/"young_reference_bank.json",bank)
    references={}
    for split,donor in (("val","OM6"),("test","OM9")):
        rows=[]
        for i,query_time in enumerate((30.,15.)):
            rows.append(dict(row=i,query_cell_id=f"{donor}_{i}",donor=donor,
                trajectory="SKM",query_pseudotime=query_time,
                context_cell_ids=["y0","y1","y2"],context_donors=["YM2"]*3,
                context_pseudotimes=[0.,10.,20.],
                context_tokens=[r["tokens"] for r in bank[:3]],
                target_tokens=bank[3-i]["tokens"],
                delta_pseudotime=query_time-20.,
                represented_delta_pseudotime=query_time-20.))
        references[split]=rows
        scoring.dump(data/f"{split}_references.json",rows)
    baseline_dir=tmp_path/"baselines"
    args=SimpleNamespace(data=data,output=baseline_dir)
    scoring.baselines(args)
    with pytest.raises(FileExistsError):
        scoring.baselines(args)
    predictions=tmp_path/"predictions"
    for split,refs in references.items():
        for task in ("tbc","nc"):
            p=predictions/f"{task}_{split}"
            p.mkdir(parents=True)
            run=dict(task=task,split=split,limit=None if task=="tbc" else 1,
                     tokenizer_sha256=m["tokenizer_sha256"],rope_factor=4.,
                     attention_backend="sdpa",time_scale=1.,label_scalar=100.,
                     references_sha256=scoring.sha256(data/f"{split}_references.json"),
                     data="/workspaces/maxToki/out/aliased-data",
                     manifest_sha256=scoring.sha256(data/"manifest.json"),checkpoint="synthetic-test")
            scoring.dump(p/"run.json",run)
            if task=="tbc":
                payload={"predictions":[{"regression_preds":torch.tensor(
                    [r["delta_pseudotime"] for r in refs])}]}
            else:
                payload=[[dict(generated_tokens=torch.tensor([refs[0]["target_tokens"]]),
                               lengths=torch.tensor([5]),finished_naturally=torch.tensor([True]))]]
            torch.save(payload,p/"predictions__rank_0.pt")
    output=tmp_path/"comparison"
    scoring.compare(SimpleNamespace(data=data,baselines=baseline_dir,
                                    predictions=predictions,output=output))
    report=json.loads((output/"comparison_metrics.json").read_text())
    tbc=report["by_split"]["test"]["methods"]["tbc"]
    nc=report["by_split"]["test"]["methods"]["nc"]
    assert tbc["maxtoki_stage2"]["overall"]["mae"]==0.
    assert tbc["maxtoki_stage2"]["overall"]["n_total"]==2
    assert nc["maxtoki_stage2"]["overall"]["n_total"]==1
    assert nc["maxtoki_stage2"]["overall"]["jaccard_at_100"]==1.
    assert nc["context_linear"]["overall"]["n_total"]==1
    assert nc["maxtoki_stage2"]["overall"]["saw_eos_count"]==1
    assert set(tbc["maxtoki_stage2"]["by_delta_sign"])=={"positive","negative"}
    raw=(output/"comparison_metrics.json").read_text()
    assert "zero_interval" not in raw and "copy_context" not in raw
    paired=report["by_split"]["test"]["paired_comparisons"]["nc"]["young_linear_trend"]
    assert paired["n_matched_available"]==1
    assert paired["model"]["overall"]["n_total"]==paired["baseline"]["overall"]["n_total"]
    # Provenance guards prevent pairing predictions with different reference rows.
    run_path=predictions/"tbc_val"/"run.json"
    invalid=json.loads(run_path.read_text()) | dict(references_sha256="wrong")
    scoring.dump(run_path,invalid)
    with pytest.raises(ValueError,match="reference hash"):
        scoring.compare(SimpleNamespace(data=data,baselines=baseline_dir,predictions=predictions,
                                        output=tmp_path/"must_fail"))
    scoring.dump(run_path, invalid | dict(
        references_sha256=scoring.sha256(data/"val_references.json"),manifest_sha256="wrong"))
    with pytest.raises(ValueError,match="Prediction manifest hash"):
        scoring.compare(SimpleNamespace(data=data,baselines=baseline_dir,predictions=predictions,
                                        output=tmp_path/"manifest_must_fail"))
    altered=references["test"]
    altered[0]=dict(altered[0],target_tokens=[2,4,6,5,3])
    scoring.dump(data/"test_references.json",altered)
    with pytest.raises(ValueError,match="Baseline reference hash"):
        scoring.compare(SimpleNamespace(data=data,baselines=baseline_dir,predictions=predictions,
                                        output=tmp_path/"stale_baseline_must_fail"))
