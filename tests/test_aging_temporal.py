"""Temporal targets, vocabulary compatibility, donor leakage and prompt leakage."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/torch_pipeline/aging_temporal.py"
spec=importlib.util.spec_from_file_location("aging_temporal",path)
at=importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)

@pytest.fixture
def vocabulary():
    t={"<pad>":0,"<mask>":1,"<bos>":2,"<eos>":3,
       "ENSG1":4,"ENSG2":5,"ENSG3":6,"ENSG4":7}
    t.update({str(i):i+28 for i in range(-20,21)})
    t.update({"<boq>":49,"<eoq>":50})
    return t

def test_signed_obs_intervals_and_no_target_leakage(vocabulary):
    numeric=at.dictionary_check(vocabulary,{k:v for k,v in vocabulary.items()
                                           if k.startswith("<") or k.startswith("ENSG")})
    e=at.compose([[2,4,3],[2,5,3],[2,6,3]],[1.25,4.25,8.25],
                 [2,7,3],2.75,vocabulary,numeric)
    assert e["context_intervals"]==[3,4]
    assert e["delta_pseudotime"]==-5.5
    assert e["represented_delta_pseudotime"]==-5
    assert e["tbc_predict"][-1]==vocabulary["0"]
    assert e["tbc_train"][-1]==vocabulary["-5"]
    assert e["nc_predict"][-1]==2
    assert 7 not in e["nc_predict"]
    assert e["nc_train"][-3:]==[2,7,3]
    assert e["tbc_predict"][:11]==[2,4,3,vocabulary["3"],2,5,3,vocabulary["4"],2,6,3]

def test_no_silent_time_clamp_and_scale():
    numeric={-10.:1,0.:2,10.:3}
    assert at.interval(1,numeric,10)==(3,1)
    with pytest.raises(ValueError,match="range"):
        at.interval(11,numeric,1)
    with pytest.raises(ValueError):
        at.interval(float("nan"),numeric,1)

def test_gene_id_shift_and_donor_overlap_rejected(vocabulary):
    with pytest.raises(ValueError,match="official"):
        at.dictionary_check(vocabulary,{"ENSG1":99})
    with pytest.raises(ValueError,match="disjoint"):
        at.donor_check({"train":["YM2"],"val":["YM2"],"test":["OM9"]})

def test_metrics_are_prediction_error():
    m=at.tbc_metrics([2,6],[1,4])
    assert m["mse"]==2.5
    assert m["mae"]==1.5
    with pytest.raises(ValueError):
        at.tbc_metrics([2],[1,4])

def test_stage1_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError,match="Stage 1"):
        at.checkpoint_check(tmp_path,{})

def test_prepare_reads_obs_and_separates_donors(tmp_path,vocabulary):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset,load_from_disk
    records=[]
    metadata=[]
    for donor in ("YM2","OM6","OM9"):
        for i in range(5):
            records.append(dict(cell_id=f"{donor}_{i}",input_ids=[2,4+i%4,3],
                                Pseudotime=i*0.5,sample=donor))
            metadata.append(dict(sample=donor,Annotation="SKM",Pseudotime=i*0.5))
    source=tmp_path/"source.h5ad"
    # Use legacy string storage for a fixture readable by older AnnData.
    with pd.option_context("future.infer_string", False):
        obs=pd.DataFrame(metadata,index=[r["cell_id"] for r in records])
        ad.AnnData(np.ones((15,4)),obs=obs).write_h5ad(source)
    cells=tmp_path/"cells.dataset"
    Dataset(pa.Table.from_pylist(records),fingerprint="testfixture").save_to_disk(str(cells))
    tokens=tmp_path/"tokens.json"
    official=tmp_path/"official.json"
    tokens.write_text(json.dumps(vocabulary))
    official.write_text(json.dumps({k:v for k,v in vocabulary.items()
                                   if k.startswith("ENSG") or k in ("<pad>","<mask>","<bos>","<eos>")}))
    args=SimpleNamespace(output=tmp_path/"prepared",h5ad=source,cells=cells,tokenizer=tokens,
        official_dictionary=official,pseudotime_col="Pseudotime",donor_col="sample",
        trajectory_col="Annotation",train_donors=["YM2"],val_donors=["OM6"],test_donors=["OM9"],
        n_context=3,examples=5,eval_examples=3,cell_cap=4,max_tokens=4,seq_length=40,
        time_scale=1.,seed=42)
    at.prepare(args)
    refs=json.loads((args.output/"test_references.json").read_text())
    assert all(r["donor"]=="OM9" for r in refs)
    for r in refs:
        assert r["delta_pseudotime"]==r["query_pseudotime"]-r["context_pseudotimes"][-1]
        assert r["query_cell_id"] not in r["context_cell_ids"]
    assert len(load_from_disk(str(args.output/"train.dataset")))==10
    assert len(load_from_disk(str(args.output/"test_nc.dataset")))==3
    with pytest.raises(FileExistsError):
        at.prepare(args)
    # Missing ground truth must be audited and excluded, never imputed.
    with pd.option_context("future.infer_string", False):
        obs.loc["YM2_0","Pseudotime"]=np.nan
        ad.AnnData(np.ones((15,4)),obs=obs).write_h5ad(source)
    args.output=tmp_path/"prepared_missing"
    at.prepare(args)
    exclusions=json.loads((args.output/"excluded_cells.json").read_text())
    assert exclusions==[dict(cell_id="YM2_0",donor="YM2",reason="missing_obs",columns=["Pseudotime"])]
    m=json.loads((args.output/"manifest.json").read_text())
    assert m["valid_cells"]==14
    assert m["excluded_by_donor"]=={"YM2":1}
