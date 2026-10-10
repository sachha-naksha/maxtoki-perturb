"""Joint SKM TBC/NextCell workflow; all CLI workloads require Slurm compute."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys

def compute():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Use a Slurm compute allocation")

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dictionary_check(tokens, original):
    if len(set(tokens.values())) != len(tokens):
        raise ValueError("Duplicate token IDs")
    if any(tokens.get(k) != v for k,v in original.items()):
        raise ValueError("Temporal dictionary changes official token IDs")
    for key in ("<bos>", "<eos>", "<boq>", "<eoq>", "0"):
        if key not in tokens:
            raise ValueError(f"Missing required token {key}")
    numeric = {float(k): int(v) for k,v in tokens.items()
               if not k.startswith("<") and not k.startswith("ENSG")}
    if not numeric or not all(math.isfinite(x) for x in numeric):
        raise ValueError("Invalid numeric vocabulary")
    return numeric

def interval(delta, numeric, scale):
    if not math.isfinite(delta) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite intervals and positive scale required")
    value = delta * scale
    if value < min(numeric) or value > max(numeric):
        raise ValueError(f"Interval {delta} outside time-token range at scale {scale}")
    encoded = min(numeric, key=lambda x: (abs(value-x), abs(x), x))
    return numeric[encoded], encoded / scale

def donor_check(splits):
    used = set()
    for name, donors in splits.items():
        if not donors or len(set(donors)) != len(donors) or used.intersection(donors):
            raise ValueError(f"Nonempty disjoint donor splits required: {name}")
        used.update(donors)

def compose(context, times, query, query_time, tokens, numeric, scale=1):
    if len(context) != len(times) or len(context) < 2:
        raise ValueError("Need at least two timed context cells")
    prefix = []
    for i, cell in enumerate(context):
        if i:
            prefix.append(interval(times[i]-times[i-1], numeric, scale)[0])
        prefix.extend(cell)
    if query[0] != tokens["<bos>"] or query[-1] != tokens["<eos>"]:
        raise ValueError("Query must contain BOS/EOS")
    delta = query_time-times[-1]
    answer, represented = interval(delta, numeric, scale)
    tbc = prefix + [tokens["<boq>"]] + query[1:-1] + [tokens["<eoq>"]]
    nc = prefix + [tokens["<boq>"], answer, tokens["<eoq>"]]
    return dict(tbc_train=tbc+[answer], nc_train=nc+query,
                tbc_predict=tbc+[tokens["0"]], nc_predict=nc+[tokens["<bos>"]],
                delta_pseudotime=delta, represented_delta_pseudotime=represented,
                context_intervals=[times[i]-times[i-1] for i in range(1,len(times))])

def prepare(a):
    compute()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset, load_from_disk
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    if a.n_context < 2 or min(a.examples,a.eval_examples) < 1:
        raise ValueError("Need >=2 context cells and positive example counts")
    label_scalar = getattr(a, "label_scalar", 1.0)
    if not math.isfinite(label_scalar) or label_scalar <= 0:
        raise ValueError("label_scalar must be finite and positive")
    if a.cell_cap < 3 or a.max_tokens < a.cell_cap:
        raise ValueError("Generation budget must accommodate the complete capped cell")
    splits = dict(train=a.train_donors, val=a.val_donors, test=a.test_donors)
    donor_check(splits)
    tokens = {k:int(v) for k,v in json.loads(a.tokenizer.read_text()).items()}
    official = json.loads(a.official_dictionary.read_text())
    numeric = dictionary_check(tokens, official)
    h5ad = ad.read_h5ad(a.h5ad, backed="r")
    try:
        obs = h5ad.obs.copy()
    finally:
        h5ad.file.close()
    ds = load_from_disk(str(a.cells))
    if not {"cell_id","input_ids"} <= set(ds.column_names):
        raise ValueError("Expected rebuilt single-cell tokenized dataset")
    cells = list(ds)
    if len({c["cell_id"] for c in cells}) != len(cells):
        raise ValueError("Duplicate cell IDs")
    if set(c["cell_id"] for c in cells) != set(obs.index.astype(str)):
        raise ValueError("H5AD and tokenized dataset cells differ")
    cols = [a.pseudotime_col,a.donor_col,a.trajectory_col]
    if any(c not in obs for c in cols):
        raise ValueError(f"Required obs columns: {cols}")
    genes = {int(v) for k,v in official.items() if k.startswith("ENSG")}
    pools = defaultdict(list)
    excluded = []
    for cell in cells:
        row = obs.loc[cell["cell_id"]]
        missing = [c for c in cols if pd.isna(row[c])]
        if missing:
            excluded.append(dict(cell_id=cell["cell_id"], donor=str(row[a.donor_col]),
                                 reason="missing_obs", columns=missing))
            continue
        time = float(row[a.pseudotime_col])
        if not math.isfinite(time):
            excluded.append(dict(cell_id=cell["cell_id"], donor=str(row[a.donor_col]),
                                 reason="nonfinite_pseudotime", columns=[a.pseudotime_col]))
            continue
        if a.pseudotime_col in cell and not np.isclose(float(cell[a.pseudotime_col]),time):
            raise ValueError("Stale pseudotime in tokenized dataset")
        donor = str(row[a.donor_col])
        if a.donor_col in cell and str(cell[a.donor_col]) != donor:
            raise ValueError("Stale donor metadata")
        seq = list(map(int,cell["input_ids"]))
        if (seq[0],seq[-1]) != (tokens["<bos>"],tokens["<eos>"]) or not set(seq[1:-1]) <= genes:
            raise ValueError("Single-cell vocabulary mismatch")
        if len(seq) > a.cell_cap:
            seq = seq[:a.cell_cap-1]+[tokens["<eos>"]]
        cell.update(time=time, donor=donor, trajectory=str(row[a.trajectory_col]), tokens=seq)
        pools[(donor,cell["trajectory"])].append(cell)
    valid_cells = [cell for pool in pools.values() for cell in pool]
    requested = set(sum(splits.values(),[]))
    if not requested <= {c["donor"] for c in valid_cells}:
        raise ValueError("Unknown donor in split")
    groups = {s:[p for (d,_),p in sorted(pools.items()) if d in donors and len(p)>=a.n_context+1]
              for s,donors in splits.items()}
    if any(not g for g in groups.values()):
        raise ValueError("No eligible trajectory in one or more splits")
    a.output.mkdir(parents=True)
    dump(a.output/"token_dictionary.json",tokens)
    dump(a.output/"excluded_cells.json",excluded)
    manifest = dict(units="pseudotime", splits=splits, pseudotime_col=a.pseudotime_col,
                    trajectory_col=a.trajectory_col, time_scale=a.time_scale, label_scalar=label_scalar,
                    seq_length=a.seq_length, max_tokens=a.max_tokens, cell_cap=a.cell_cap,
                    seed=a.seed, tokenizer_sha256=digest(a.output/"token_dictionary.json"),
                    source_h5ad=str(a.h5ad.resolve()), source_cells=str(a.cells.resolve()),
                    source_obs_sha256=hashlib.sha256(obs[cols].to_csv().encode()).hexdigest(),
                    rope_factor=4.0, old_context_len=4096, attention_backend="sdpa",
                    excluded_cells=len(excluded), valid_cells=len(valid_cells),
                    excluded_by_donor=dict(__import__("collections").Counter(c["donor"] for c in excluded)),
                    counts={})
    def save(name,rows):
        table=pa.Table.from_pylist(rows)
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink,table.schema) as writer:
                writer.write_table(table)
            fingerprint=hashlib.sha256(sink.getvalue()).hexdigest()
        Dataset(table,fingerprint=fingerprint).save_to_disk(str(a.output/name))
    for split_index,(split,donors) in enumerate(splits.items()):
        rng=random.Random(a.seed+split_index)
        count=a.examples if split=="train" else a.eval_examples
        mixed,tbc,nc,refs=[],[],[],[]
        for i in range(count):
            selected=rng.sample(rng.choice(groups[split]),a.n_context+1)
            query=selected.pop()
            context=sorted(selected,key=lambda c:(c["time"],c["cell_id"]))
            e=compose([c["tokens"] for c in context],[c["time"] for c in context],
                      query["tokens"],query["time"],tokens,numeric,a.time_scale)
            if max(len(e["tbc_train"]),len(e["nc_train"]),
                   len(e["nc_predict"])+a.max_tokens-1)>a.seq_length:
                raise ValueError("Sequence budget exceeded: reduce cell cap or context count")
            mixed.extend([dict(input_ids=e["tbc_train"]),dict(input_ids=e["nc_train"])])
            tbc.append(dict(input_ids=e["tbc_predict"]))
            nc.append(dict(input_ids=e["nc_predict"]))
            refs.append({k:v for k,v in e.items() if not k.endswith(("_train","_predict"))} |
                        dict(row=i,query_cell_id=query["cell_id"],donor=query["donor"],
                             trajectory=query["trajectory"],query_pseudotime=query["time"],
                             context_cell_ids=[c["cell_id"] for c in context],
                             context_pseudotimes=[c["time"] for c in context],
                             target_tokens=query["tokens"],copy_context_tokens=context[-1]["tokens"]))
        rng.shuffle(mixed)
        save(f"{split}.dataset",mixed)
        if split!="train":
            save(f"{split}_tbc.dataset",tbc)
            save(f"{split}_nc.dataset",nc)
            dump(a.output/f"{split}_references.json",refs)
        manifest["counts"][split]=dict(trajectories=count,training_rows=len(mixed),
                                      eligible_groups=len(groups[split]),
                                      eligible_cells=sum(map(len,groups[split])) )
    dump(a.output/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2))

def manifest(data):
    m=json.loads((data/"manifest.json").read_text())
    if digest(data/"token_dictionary.json")!=m["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer changed")
    return m

def scaling(scalar):
    from bionemo.maxtoki.model import MaxTokiFineTuneModel
    original=MaxTokiFineTuneModel.__init__
    def init(self,*args,**kwargs):
        kwargs["label_scalar"]=scalar
        original(self,*args,**kwargs)
    MaxTokiFineTuneModel.__init__=init

def train(a):
    compute()
    m=manifest(a.data)
    if a.output.exists():
        raise FileExistsError("Choose a fresh training output directory")
    if not (a.checkpoint/"weights").exists():
        raise ValueError("Expected BioNeMo checkpoint with weights/")
    scaling(m["label_scalar"])
    if m.get("runtime_protocol"):
        from aging_stage2_runtime import install
        print("Runtime fixes:", install(m["runtime_protocol"]), flush=True)
    from predict_runner import _patch_determine_task_type_bos_check
    _patch_determine_task_type_bos_check()
    import bionemo.maxtoki.train as upstream
    paths=m.get("dataset_paths", {split:f"{split}.dataset" for split in ("train","val","test")})
    for split in ("train","val","test"):
        if not (a.data/paths[split]).is_dir():
            raise ValueError(f"Missing {split} dataset: {paths[split]}")
    argv=["--train-data-path",str(a.data/paths["train"]),
          "--val-data-path",str(a.data/paths["val"]),"--test-data-path",str(a.data/paths["test"]),
          "--tokenizer-path",str(a.data/"token_dictionary.json"),
          "--initial-ckpt-path",str(a.checkpoint),"--use-finetuning-config",
          "--output-weights","separate",
          "--result-dir",str(a.output),"--experiment-name","aging_skm_joint",
          "--num-gpus","1","--num-steps",str(a.steps),"--seq-length",str(m["seq_length"]),
          "--micro-batch-size","1","--accumulate-grad-batches","4","--lr",str(a.lr),
          "--rope-scaling-factor","4","--old-context-len","4096","--label-scalar",str(m["label_scalar"]),
          "--timelapse-loss","mse","--use-sdpa","--val-check-interval",str(min(a.steps,100)),
          "--limit-val-batches","1.0","--log-every-n-steps","5","--num-dataset-workers","0"]
    old=sys.argv
    sys.argv=["bionemo.maxtoki.train",*argv]
    try:
        upstream.entrypoint()
    finally:
        sys.argv=old
    checkpoints=sorted(p for p in a.output.rglob("*")
                       if p.is_dir() and (p/"weights").is_dir() and (p/"context").is_dir())
    if not checkpoints:
        raise RuntimeError("Training did not save a checkpoint")
    saved_steps={}
    if m.get("runtime_protocol"):
        from aging_stage2_runtime import checkpoint_callbacks
        for callback in checkpoint_callbacks:
            for name,step in callback.saved_optimizer_steps.items():
                path=Path(name)
                if path.suffix==".ckpt":
                    path=path.with_suffix("")
                saved_steps[str(path.resolve())]=step
    for p in checkpoints:
        dump(p/"aging_temporal_manifest.json",m |
             dict(tasks=["tbc","nc"],head="headless",requested_steps=a.steps,
                  steps=saved_steps.get(str(p.resolve()),a.steps),
                  initial_checkpoint=str(a.checkpoint.resolve())))
    summary=dict(requested_steps=a.steps, data=str(a.data.resolve()),
                 splits=m["splits"], runtime_protocol=m.get("runtime_protocol"),
                 checkpoints=[str(p.resolve()) for p in checkpoints])
    if m.get("runtime_protocol"):
        from aging_stage2_runtime import checkpoint_callbacks
        for callback in checkpoint_callbacks:
            if callback.best_model_path:
                best=Path(callback.best_model_path)
                if not best.is_dir() and best.suffix==".ckpt":
                    best=best.with_suffix("")
                if not (best/"weights").is_dir():
                    raise RuntimeError(f"Best validation checkpoint was not saved: {best}")
                summary.update(best_checkpoint=str(best.resolve()),
                    best_val_loss=float(callback.best_model_score),
                    checkpoint_selection="minimum current full OM9 mixed validation loss",
                    last_checkpoint=str(callback.last_model_path))
    dump(a.output/"training_summary.json",summary)
    print("Trained checkpoints:",*checkpoints,sep="\n")
    print("Training summary:",json.dumps(summary),flush=True)

def checkpoint_check(checkpoint,m):
    p=checkpoint/"aging_temporal_manifest.json"
    if not p.exists():
        raise ValueError("No joint-training provenance: Stage 1 cannot be used for this evaluation")
    c=json.loads(p.read_text())
    for key in ("tokenizer_sha256","time_scale","label_scalar","seq_length","splits","source_obs_sha256","rope_factor","old_context_len","attention_backend"):
        if c[key]!=m[key]:
            raise ValueError(f"Checkpoint/dataset mismatch: {key}")
    for key in ("runtime_protocol","numeric_expectation_normalized","nc_full_answer_loss"):
        if c.get(key)!=m.get(key):
            raise ValueError(f"Checkpoint/dataset mismatch: {key}")
    if set(c["tasks"])!={"tbc","nc"} or c["head"]!="headless":
        raise ValueError("Expected joint TBC/NextCell checkpoint")

def inference_config(m):
    # Upstream deliberately overrides checkpoint RoPE with CLI/default fields.
    # Supply the training settings explicitly instead of its factor=8 default.
    from dataclasses import dataclass
    import bionemo.maxtoki.predict as bp
    from bionemo.maxtoki.sdpa_attention import sdpa_layer_spec
    base=bp.MaxTokiMultitaskFineTuneConfig
    @dataclass
    class AgingPredictConfig(base):
        scale_factor: float = 4.0
        old_context_len: int = 4096
        transformer_layer_spec: object = sdpa_layer_spec
        def __post_init__(self):
            parent=getattr(super(),"__post_init__",None)
            if parent:
                parent()
            self.scale_factor=m["rope_factor"]
            self.old_context_len=m["old_context_len"]
            self.transformer_layer_spec=sdpa_layer_spec
            self.override_parent_fields=list(self.override_parent_fields)
            for field in ("scale_factor","old_context_len","transformer_layer_spec"):
                if field not in self.override_parent_fields:
                    self.override_parent_fields.append(field)
    bp.MaxTokiMultitaskFineTuneConfig=AgingPredictConfig

def predict(a):
    compute()
    m=manifest(a.data)
    checkpoint_check(a.checkpoint,m)
    inference_config(m)
    if a.output.exists():
        raise FileExistsError("Choose a fresh prediction output directory")
    scaling(m["label_scalar"])
    if m.get("runtime_protocol"):
        from aging_stage2_runtime import install
        print("Runtime fixes:", install(m["runtime_protocol"]), flush=True)
    from predict_runner import run_headless_predict
    run_headless_predict(ckpt_dir=a.checkpoint,tokenizer_path=a.data/"token_dictionary.json",
        data_path=a.data/f"{a.split}_{a.task}.dataset",output_dir=a.output,variant="217m",
        seq_length=m["seq_length"],micro_batch_size=1,devices=1,work_dir=a.output/"work",
        write_interval="epoch",limit_predict_batches_to_n=a.limit,
        generate_next_cell=a.task=="nc",max_tokens_to_generate=m["max_tokens"],
        top_k=1,buffer_size_gb=a.buffer_gb,buffer_overflow_factor=50.0)
    dump(a.output/"run.json",dict(task=a.task,split=a.split,limit=a.limit,
         checkpoint=str(a.checkpoint.resolve()),tokenizer_sha256=m["tokenizer_sha256"],
         data=str(a.data.resolve()), manifest_sha256=digest(a.data/"manifest.json"),
         references_sha256=digest(a.data/f"{a.split}_references.json"),
         label_scalar=m["label_scalar"], time_scale=m["time_scale"],
         runtime_protocol=m.get("runtime_protocol"),
         rope_factor=m["rope_factor"],attention_backend=m["attention_backend"]))
    if a.score:
        score(argparse.Namespace(data=a.data,predictions=a.output))

def tbc_metrics(predicted,target):
    import numpy as np
    p,t=np.asarray(predicted,dtype=float),np.asarray(target,dtype=float)
    if p.shape!=t.shape or not p.size or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Finite matching nonempty predictions and targets required")
    return dict(n=len(t),mae=float(np.abs(p-t).mean()),mse=float(np.square(p-t).mean()),
                pearson=float(np.corrcoef(p,t)[0,1]) if len(t)>1 and p.std()>0 and t.std()>0 else None)

def score(a):
    compute()
    import numpy as np
    m=manifest(a.data)
    run=json.loads((a.predictions/"run.json").read_text())
    if "invalid_reason" in run:
        raise ValueError(run["invalid_reason"])
    if run["rope_factor"]!=m["rope_factor"] or run["attention_backend"]!=m["attention_backend"]:
        raise ValueError("Inference settings differ from training")
    if run["tokenizer_sha256"]!=m["tokenizer_sha256"]:
        raise ValueError("Prediction tokenizer mismatch")
    refs=json.loads((a.data/f"{run['split']}_references.json").read_text())
    if run["limit"] is not None:
        refs=refs[:run["limit"]]
    report=dict(task=run["task"],split=run["split"],units="pseudotime")
    if run["task"]=="tbc":
        import torch
        def flatten(v):
            if isinstance(v,(tuple,list)):
                return sum((flatten(x) for x in v),[])
            if isinstance(v,dict):
                if "regression_preds" in v:
                    return torch.as_tensor(v["regression_preds"]).float().reshape(-1).tolist()
                if "predictions" in v:
                    return flatten(v["predictions"])
            raise ValueError("Unrecognized TBC payload")
        paths=list(a.predictions.glob("predictions__rank_*.pt"))
        if len(paths)!=1:
            raise ValueError("Expected single-GPU predictions")
        values=flatten(torch.load(paths[0],map_location="cpu",weights_only=False))
        values=[v/m["time_scale"] for v in values]
        target=[r["delta_pseudotime"] for r in refs]
        report.update(tbc_metrics(values,target))
        report["quantized_target_metrics"]=tbc_metrics(
            values,[r["represented_delta_pseudotime"] for r in refs])
        report["per_row"]=[dict(query_cell_id=r["query_cell_id"],donor=r["donor"],
                               target=t,predicted=p,error=p-t) for r,t,p in zip(refs,target,values)]
    else:
        from score_nextcell import (_extract_per_row_tokens,decode_generation,
                                    jaccard_at_k,spearman_on_shared)
        rows=_extract_per_row_tokens(a.predictions)
        if len(rows)!=len(refs):
            raise ValueError("NextCell prediction/target counts differ")
        tokens=json.loads((a.data/"token_dictionary.json").read_text())
        genes={v:k for k,v in tokens.items() if k.startswith("ENSG")}
        specials={k:v for k,v in tokens.items() if k.startswith("<")}
        numeric=set(tokens.values())-set(genes)-set(specials.values())
        records=[]
        for row,ref in zip(rows,refs):
            decoded=decode_generation(row["tokens"],genes,specials,numeric)
            target=[genes[t] for t in ref["target_tokens"] if t in genes]
            rho,n=spearman_on_shared(decoded.ensg_order,target)
            records.append(dict(query_cell_id=ref["query_cell_id"],donor=ref["donor"],
                delta_pseudotime=ref["delta_pseudotime"],saw_eos=decoded.saw_eos,
                jaccard_at_100=jaccard_at_k(decoded.ensg_order,target,100),
                spearman_shared=float(rho) if np.isfinite(rho) else None,n_shared=n,
                invalid_tokens=decoded.n_invalid,duplicate_tokens=decoded.n_duplicates))
        report.update(n=len(records),per_row=records,
            mean_jaccard_at_100=float(np.mean([r["jaccard_at_100"] for r in records])))
    dump(a.predictions/"metrics.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="per_row"},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare")
    for name,default in (
        ("h5ad","data/zero_shot/rna_zero_shot.preprocessed.h5ad"),
        ("cells","data/zero_shot/aging_skm.dataset"),
        ("tokenizer","delta/configs/token_dictionary_tbc_official_extended.json"),
        ("official-dictionary","data/zero_shot/resources/token_dictionary_v1.json")):
        q.add_argument("--"+name,type=Path,default=Path(default))
    q.add_argument("--output",type=Path,required=True)
    for name,default in (("pseudotime-col","Pseudotime"),("donor-col","sample"),("trajectory-col","Annotation")):
        q.add_argument("--"+name,default=default)
    for name,default in (("train-donors",["YM2"]),("val-donors",["OM6"]),("test-donors",["OM9"])):
        q.add_argument("--"+name,nargs="+",default=default)
    for name,default in (("n-context",3),("examples",2000),("eval-examples",20),
                         ("cell-cap",2048),("seq-length",16384),("max-tokens",2048),("seed",42)):
        q.add_argument("--"+name,type=int,default=default)
    q.add_argument("--time-scale",type=float,default=1.0)
    q.add_argument("--label-scalar",type=float,default=200.0,
                   help="Regression loss normalization; inference restores pseudotime units")
    q.set_defaults(func=prepare)
    q=sub.add_parser("train")
    for name in ("data","checkpoint","output"):
        q.add_argument("--"+name,type=Path,required=True)
    q.add_argument("--steps",type=int,default=500)
    q.add_argument("--lr",type=float,default=5e-5)
    q.set_defaults(func=train)
    q=sub.add_parser("predict")
    for name in ("data","checkpoint","output"):
        q.add_argument("--"+name,type=Path,required=True)
    q.add_argument("--task",choices=["tbc","nc"],required=True)
    q.add_argument("--split",choices=["val","test"],default="test")
    q.add_argument("--limit",type=int)
    q.add_argument("--buffer-gb",type=float,default=16.0)
    q.add_argument("--score",action="store_true")
    q.set_defaults(func=predict)
    q=sub.add_parser("score")
    q.add_argument("--data",type=Path,required=True)
    q.add_argument("--predictions",type=Path,required=True)
    q.set_defaults(func=score)
    a=p.parse_args()
    if getattr(a,"steps",1)<1 or (getattr(a,"limit",None) is not None and a.limit<1):
        p.error("steps and limit must be positive")
    a.func(a)
if __name__=="__main__":
    main()
