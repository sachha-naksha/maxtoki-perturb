"""Run corrected Stage 2 and modest held-out-donor evaluation in one GPU job."""
from pathlib import Path
import json
import subprocess
import sys
from aging_temporal import compute, dump

def main():
    compute()
    data=Path("out/aging_skm_stage2_cross_donor_v1")
    trained=Path("out/aging_skm_cross_donor_training_v1")
    predicted=Path("out/aging_skm_cross_donor_eval_v1")
    if trained.exists() or predicted.exists():
        raise FileExistsError("Fresh training and prediction output directories required")
    base=[sys.executable, "-u", "scripts/torch_pipeline/aging_temporal.py"]
    def run(*args):
        print("Starting:", " ".join(map(str,args)), flush=True)
        subprocess.run(base+[str(x) for x in args],check=True)
    run("train","--data",data,"--checkpoint",
        "/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo",
        "--output",trained,"--steps",500)
    summary=json.loads((trained/"training_summary.json").read_text())
    checkpoint=Path(summary["best_checkpoint"])
    for task,limit in (("tbc",100),("nc",10)):
        run("predict","--data",data,"--checkpoint",checkpoint,"--output",
            predicted/f"{task}_val","--task",task,"--split","val","--limit",limit)
    dump(predicted/"run_summary.json",dict(
        training_summary=summary, evaluation_role="OM9 validation; no independent test",
        tbc_queries=100,nc_queries=10))
    print("Corrected Stage 2 training and validation inference completed.",flush=True)

if __name__=="__main__":
    main()
