"""Paired gene-inhibition TBC inference using versioned Stage 2 checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from aging_temporal import (checkpoint_check, compute, digest, dump,
                            inference_config, manifest, scaling)

def checkpoint_path(summary_path, which):
    summary = json.loads(Path(summary_path).read_text())
    raw = summary["best_checkpoint" if which == "selected" else "last_checkpoint"]
    path = Path(raw)
    if path.suffix == ".ckpt" and not path.is_dir():
        path = path.with_suffix("")
    if not path.is_dir() and str(path).startswith("/workspaces/maxToki/"):
        path = Path(__file__).resolve().parents[2] / path.relative_to("/workspaces/maxToki")
    if not (path / "weights").is_dir():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    return path

def dataset_hash(path):
    result = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            result.update(str(item.relative_to(path)).encode())
            result.update(bytes.fromhex(digest(item)))
    return result.hexdigest()

def predict(a):
    compute()
    import numpy as np
    from datasets import load_from_disk
    from score import sink_predictions
    if a.output.exists():
        raise FileExistsError(f"Output already exists: {a.output}")
    m = manifest(a.data)
    checkpoint = checkpoint_path(a.training_summary, a.which)
    checkpoint_check(checkpoint, m)
    if m.get("runtime_protocol") != "aging_stage2_runtime_v1":
        raise ValueError("Corrected Stage 2 runtime is required")
    if digest(a.data/"references.json") != m["prepared_pair"]["references_sha256"]:
        raise ValueError("Prepared reference file changed")
    refs = json.loads((a.data / "references.json").read_text())
    if a.subset == "repeat":
        selected_rows = m["prepared_pair"]["repeat_query_rows"]
        refs = [refs[i] for i in selected_rows]
        dataset = a.data / "repeat.dataset"
    else:
        dataset = a.data / "paired.dataset"
    dataset_name = "repeat" if a.subset == "repeat" else "paired"
    for filename, expected in m["prepared_pair"]["dataset_file_sha256"][dataset_name].items():
        if digest(dataset/filename) != expected:
            raise ValueError(f"Prepared dataset file changed: {filename}")
    records = load_from_disk(str(dataset))
    if len(records) != 2 * len(refs):
        raise ValueError("Paired dataset/reference row count mismatch")
    if len({r["query_cell_id"] for r in refs}) != len(refs):
        raise ValueError("Expected unique query cells in this experiment")
    for i, ref in enumerate(refs):
        control = records[2*i]["input_ids"]
        perturbed = records[2*i+1]["input_ids"]
        if max(len(control), len(perturbed)) > m["prompt_token_budget"]:
            raise ValueError("Historical 8k input budget exceeded")
        if not ref["gene_present"] and control != perturbed:
            raise ValueError("Absent-gene input changed")
    identity = dataset_hash(dataset)
    scaling(m["label_scalar"])
    inference_config(m)
    from aging_stage2_runtime import install
    from predict_runner import run_headless_predict
    import bionemo.maxtoki.predict as bp
    original_datamodule_init = bp.MaxTokiDataModule.__init__
    def small_loader(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        original_datamodule_init(self, *args, **kwargs)
    bp.MaxTokiDataModule.__init__ = small_loader
    print("Runtime:", install(m["runtime_protocol"]), flush=True)
    print(f"Predicting {len(refs)} paired queries with {a.which} checkpoint ({a.subset}).", flush=True)
    run_headless_predict(
        ckpt_dir=checkpoint, tokenizer_path=a.data/"token_dictionary.json",
        data_path=dataset, output_dir=a.output/"predictions", variant="217m",
        seq_length=m["prompt_token_budget"], micro_batch_size=1, devices=1,
        work_dir=a.output/"work", write_interval="epoch",
        buffer_size_gb=4.0, buffer_overflow_factor=50.0)
    paths = list((a.output/"predictions").glob("predictions__rank_*.pt"))
    if len(paths) != 1:
        raise ValueError("Exactly one prediction rank file required")
    values = np.asarray(sink_predictions(a.output/"predictions"), dtype=float)
    if values.shape != (2*len(refs),) or not np.isfinite(values).all():
        raise ValueError("Nonfinite, missing, or misaligned paired predictions")
    values = values / m["time_scale"]
    control, perturbed = values[0::2], values[1::2]
    present = np.asarray([r["gene_present"] for r in refs], dtype=bool)
    delta = perturbed-control
    if not np.allclose(delta[~present], 0, atol=1e-6, rtol=0):
        raise RuntimeError("Identical absent-gene inputs produced different predictions")
    rows = [
        dict(row=int(ref["row"]), query_cell_id=ref["query_cell_id"],
             donor=ref["donor"], trajectory=ref["trajectory"],
             gene_present=bool(ref["gene_present"]),
             query_pseudotime=ref["query_pseudotime"],
             context_cell_ids=ref["context_cell_ids"],
             control_prediction=float(c), perturbed_prediction=float(t),
             delta_delta_t=float(t-c),
             target_delta_pseudotime=ref["delta_pseudotime"])
        for ref,c,t in zip(refs,control,perturbed)]
    dump(a.output/"rows.json", rows)
    np.savez(a.output/"scores.npz", baseline=control, perturbed=perturbed,
             delta_t=delta, gene_present=present)
    checkpoint_manifest = json.loads((checkpoint/"aging_temporal_manifest.json").read_text())
    gene = m["perturbation_protocol"]
    run = dict(
        task=f"{gene['gene_symbol']} inhibition paired TimeBetweenCells", checkpoint_selection=a.which,
        checkpoint=str(checkpoint.resolve()), checkpoint_steps=checkpoint_manifest["steps"],
        subset=a.subset, n_rows=len(refs), n_gene_present=int(present.sum()),
        perturbation=gene["operation"],
        gene_ensembl=gene["gene_ensembl"], gene_token=gene["gene_token"],
        mean_delta_delta_t=float(delta.mean()),
        mean_delta_delta_t_gene_present=float(delta[present].mean()) if present.any() else None,
        max_abs_absent_effect=float(np.max(np.abs(delta[~present]))) if (~present).any() else None,
        units="stored pseudotime coordinate; perturbed prediction minus control prediction",
        prompt_token_budget=m["prompt_token_budget"], seq_length=m["prompt_token_budget"],
        training_seq_length=m["seq_length"], dataloader_workers=0,
        label_scalar=m["label_scalar"], time_scale=m["time_scale"],
        numeric_expectation_normalized=m["numeric_expectation_normalized"],
        runtime_protocol=m["runtime_protocol"], rope_factor=m["rope_factor"],
        old_context_len=m["old_context_len"], attention_backend=m["attention_backend"],
        training_donors=m["splits"]["train"], validation_donors=m["splits"]["val"],
        query_donors=sorted({r["donor"] for r in refs}),
        context_protocol="Historical three fixed YM2 cells with stored pseudotime intervals",
        manifest_sha256=digest(a.data/"manifest.json"),
        references_sha256=digest(a.data/"references.json"),
        dataset_sha256=identity, tokenizer_sha256=m["tokenizer_sha256"],
        checkpoint_manifest_sha256=digest(checkpoint/"aging_temporal_manifest.json"),
        predictions_sha256=digest(paths[0]), rows_sha256=digest(a.output/"rows.json"),
        job_id=os.environ["SLURM_JOB_ID"], compute_node=socket.gethostname(),
        evaluation_role="Computational perturbation probe; OM6 training donor, OM9 validation donor")
    dump(a.output/"run.json", run)
    print(json.dumps(run, indent=2), flush=True)

def verify_repeat(full_dir, repeat_dir):
    import numpy as np
    full = json.loads((full_dir/"rows.json").read_text())
    repeated = json.loads((repeat_dir/"rows.json").read_text())
    indexed = {r["query_cell_id"]:r for r in full}
    fields = ("control_prediction","perturbed_prediction")
    actual, expected = [], []
    for row in repeated:
        reference = indexed[row["query_cell_id"]]
        for key in ("row","donor","gene_present"):
            if row[key] != reference[key]:
                raise ValueError("Repeat pairing changed")
        actual.append([row[k] for k in fields])
        expected.append([reference[k] for k in fields])
    actual, expected = np.asarray(actual), np.asarray(expected)
    report = dict(
        n_queries=len(repeated), n_predictions=int(actual.size),
        separate_model_load=True, micro_batch_size=1,
        max_absolute_difference=float(np.max(np.abs(actual-expected))),
        exactly_equal=bool(np.array_equal(actual,expected)),
        within_tolerance=bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-4)),
        rtol=1e-5, atol=1e-4)
    dump(full_dir/"repeatability.json", report)
    if not report["within_tolerance"]:
        raise RuntimeError(f"Independent repeat did not reproduce predictions: {report}")
    print("Repeatability:", json.dumps(report), flush=True)

def run(a):
    compute()
    from pathlib import Path
    a.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    run_paths = {}
    for which in ("selected", "final"):
        repeat_dir = a.output/f"{which}_repeat"
        full_dir = a.output/which
        for subset, destination in (("repeat",repeat_dir),("full",full_dir)):
            command = [sys.executable,"-u",str(script),"predict",
                       "--data",str(a.data),"--training-summary",str(a.training_summary),
                       "--which",which,"--subset",subset,"--output",str(destination)]
            print("Starting",which,subset,flush=True)
            subprocess.run(command, check=True)
        verify_repeat(full_dir, repeat_dir)
        run_paths[which] = str(full_dir.resolve())
    query_count = len(json.loads((a.data/"references.json").read_text()))
    dump(a.output/"run_summary.json",dict(
        status="completed", runs=run_paths, query_count=query_count,
        same_historical_queries=True, historical_context=True,
        stage2_temporal_prompt=True, job_id=os.environ["SLURM_JOB_ID"]))
    print("Both Stage 2 checkpoint evaluations completed.",flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for name, fn in (("predict",predict),("run",run)):
        p = sub.add_parser(name)
        p.add_argument("--data",type=Path,default=Path("out/pdk4_stage2_tbc_v1/data"))
        p.add_argument("--training-summary",type=Path,
                       default=Path("out/aging_skm_cross_donor_training_v1/training_summary.json"))
        p.add_argument("--output",type=Path,required=True)
        if name == "predict":
            p.add_argument("--which",choices=("selected","final"),required=True)
            p.add_argument("--subset",choices=("repeat","full"),default="full")
        p.set_defaults(func=fn)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
