"""Prepare young-to-old SKM evaluation prompts from a trained temporal protocol.

Run only inside a Slurm compute allocation. Donor age selects query direction;
stored pseudotime, including negative or zero differences, remains the target.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from aging_temporal import compose, compute, dictionary_check, digest, dump, manifest


def sample_context(pool, rng, n_context=3):
    """Sample distinct young cells across time bins without inspecting the query."""
    if n_context < 2 or len(pool) < n_context:
        raise ValueError("Insufficient young cells for the requested context")
    by_time = defaultdict(list)
    for cell in pool:
        by_time[cell["time"]].append(cell)
    times = sorted(by_time)
    if len(times) >= n_context:
        # Disjoint bins of distinct time values ensure temporal coverage even
        # when many cells share a pseudotime. Selection uses young data only.
        selected = []
        for i in range(n_context):
            values = times[i * len(times) // n_context:(i + 1) * len(times) // n_context]
            candidates = [cell for value in values for cell in by_time[value]]
            selected.append(rng.choice(sorted(candidates, key=lambda c: c["cell_id"])))
    else:
        selected = rng.sample(sorted(pool, key=lambda c: c["cell_id"]), n_context)
    return sorted(selected, key=lambda c: (c["time"], c["cell_id"]))


def early_context_pool(pool, quantile=0.25):
    """Select early young cells without using old-donor values or expression."""
    import numpy as np

    if not 0 < quantile <= 1 or len(pool) < 3:
        raise ValueError("Need at least three young cells and quantile in (0, 1]")
    times = sorted(cell["time"] for cell in pool)
    unique = sorted(set(times))
    threshold = float(np.quantile(times, quantile))
    # Expand only when the requested range cannot support three distinct cells
    # and, whenever present in the young pool, three distinct pseudotimes.
    required_time = unique[min(3, len(unique)) - 1]
    actual_threshold = max(threshold, required_time, times[2])
    selected = [cell for cell in pool if cell["time"] <= actual_threshold]
    return selected, dict(
        original_young_cells=len(pool), context_pool_cells=len(selected),
        requested_quantile=quantile, requested_threshold=threshold,
        actual_threshold=actual_threshold, expanded_threshold=actual_threshold > threshold,
        distinct_pseudotimes=len({cell["time"] for cell in selected}),
        fewer_than_three_distinct_young_times=len(unique) < 3,
    )


def stratified_queries(groups, count, rng):
    """Unique queries, approximately balanced by annotation, in round-robin order."""
    if count < 1 or count > sum(len(group) for group in groups.values()):
        raise ValueError("Requested query count exceeds eligible unique old cells")
    queues = {}
    for annotation, group in sorted(groups.items()):
        queues[annotation] = sorted(group, key=lambda c: c["cell_id"])
        rng.shuffle(queues[annotation])
    result = []
    while len(result) < count:
        for queue in queues.values():
            if queue:
                result.append(queue.pop())
                if len(result) == count:
                    break
    return result


def pseudotime_summary(cells):
    import numpy as np

    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["donor"], cell["trajectory"])].append(cell["time"])
    return [
        dict(donor=donor, trajectory=trajectory, n=len(values),
             min=float(min(values)), q25=float(np.quantile(values, .25)),
             median=float(np.median(values)), q75=float(np.quantile(values, .75)),
             max=float(max(values)))
        for (donor, trajectory), values in sorted(groups.items())
    ]


def prepare(args):
    compute()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset, load_from_disk

    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a fresh directed-evaluation output directory")
    queries_per_donor = args.queries_per_donor
    seed = args.seed
    young_quantile = getattr(args, "young_context_quantile", 0.25)
    if not 0 < young_quantile <= 1:
        raise ValueError("young_context_quantile must be in (0, 1]")
    if queries_per_donor < 1:
        raise ValueError("queries_per_donor must be positive")
    model = manifest(source)
    if model["splits"] != {"train": ["YM2"], "val": ["OM6"], "test": ["OM9"]}:
        raise ValueError("Expected YM2 training, OM6 validation and OM9 test provenance")
    tokens = {k: int(v) for k, v in json.loads((source / "token_dictionary.json").read_text()).items()}
    numeric = dictionary_check(tokens, {})
    genes = {v for k, v in tokens.items() if k.startswith("ENSG")}
    donor_col = model.get("donor_col", "sample")
    cols = [model["pseudotime_col"], donor_col, model["trajectory_col"]]
    h5ad = ad.read_h5ad(model["source_h5ad"], backed="r")
    try:
        obs = h5ad.obs.copy()
    finally:
        h5ad.file.close()
    if any(col not in obs for col in cols):
        raise ValueError(f"Missing required source metadata: {cols}")
    obs_hash = hashlib.sha256(obs[cols].to_csv().encode()).hexdigest()
    if obs_hash != model["source_obs_sha256"]:
        raise ValueError("Source obs changed since model training")
    dataset = load_from_disk(model["source_cells"])
    if not {"cell_id", "input_ids"} <= set(dataset.column_names):
        raise ValueError("Expected corrected single-cell tokenized dataset")
    records = list(dataset)
    ids = [str(cell["cell_id"]) for cell in records]
    if len(set(ids)) != len(ids) or obs.index.has_duplicates:
        raise ValueError("Duplicate source cell IDs")
    if set(ids) != set(obs.index.astype(str)):
        raise ValueError("H5AD and tokenized cell IDs differ")
    obs.index = obs.index.astype(str)
    pools = defaultdict(list)
    excluded = []
    for record in records:
        cell_id = str(record["cell_id"])
        row = obs.loc[cell_id]
        donor = str(row[donor_col])
        missing = [col for col in cols if pd.isna(row[col])]
        if missing:
            excluded.append(dict(cell_id=cell_id, donor=donor,
                                 reason="missing_obs", columns=missing))
            continue
        time = float(row[model["pseudotime_col"]])
        if not math.isfinite(time):
            excluded.append(dict(cell_id=cell_id, donor=donor,
                                 reason="nonfinite_pseudotime", columns=[cols[0]]))
            continue
        if donor_col in record and str(record[donor_col]) != donor:
            raise ValueError(f"Stale donor metadata: {cell_id}")
        if cols[0] in record and not np.isclose(float(record[cols[0]]), time):
            raise ValueError(f"Stale tokenized pseudotime: {cell_id}")
        seq = [int(token) for token in record["input_ids"]]
        if (len(seq) < 3 or (seq[0], seq[-1]) != (tokens["<bos>"], tokens["<eos>"])
                or not set(seq[1:-1]) <= genes):
            raise ValueError(f"Single-cell vocabulary mismatch: {cell_id}")
        if len(seq) > model["cell_cap"]:
            seq = seq[:model["cell_cap"] - 1] + [tokens["<eos>"]]
        cell = dict(cell_id=cell_id, donor=donor, trajectory=str(row[cols[2]]),
                    time=time, tokens=seq)
        pools[(donor, cell["trajectory"])].append(cell)
    young = {annotation: cells for (donor, annotation), cells in pools.items() if donor == "YM2"}
    eligible_young = {annotation: cells for annotation, cells in young.items() if len(cells) >= 3}
    if not eligible_young:
        raise ValueError("No young annotation has at least three valid cells")
    context_pools, context_pool_audit = {}, {}
    for annotation, pool in sorted(eligible_young.items()):
        context_pools[annotation], context_pool_audit[annotation] = early_context_pool(pool, young_quantile)

    def save(path, rows):
        table = pa.Table.from_pylist(rows)
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            fingerprint = hashlib.sha256(sink.getvalue()).hexdigest()
        Dataset(table, fingerprint=fingerprint).save_to_disk(str(path))

    all_cells = [cell for pool in pools.values() for cell in pool]
    protocol = dict(
        name="young_context_old_query_v1",
        biological_direction="YM2 -> {OM6, OM9}; no ordering between OM6 and OM9",
        context_donor="YM2", query_donors=["OM6", "OM9"], n_context=3,
        young_context_quantile=young_quantile, context_pool_by_annotation=context_pool_audit,
        context_policy="Restrict to the lower requested quantile of YM2 pseudotime within "
                       "the query annotation; expand the upper threshold only as necessary "
                       "to contain three distinct cells and three distinct times if available. "
                       "Sample one cell from each of three disjoint bins of distinct young "
                       "times; use three distinct cells when fewer times exist. Sort by young "
                       "pseudotime. Selection never reads query expression or query pseudotime.",
        query_policy="Unique queries balanced by annotation, emitted round-robin; subset "
                     "selection uses cell IDs, annotation and a fixed random seed only.",
        metric_weighting="Report annotation-balanced and per-annotation results; selected "
                         "queries are not sampled in population proportions.",
        target="Stored query pseudotime minus last context pseudotime; signed values retained.",
        sequence_protocol="Existing Stage 2 compose() with original tokenizer, units and budget.",
        checkpoint_status="Evaluation of existing checkpoint; no retraining or change to "
                          "model training donor splits.",
        seed=seed, queries_per_donor=queries_per_donor,
    )
    evaluation_splits = {
        "val": dict(context_donors=["YM2"], query_donors=["OM6"]),
        "test": dict(context_donors=["YM2"], query_donors=["OM9"]),
    }
    audit = dict(
        source_obs_sha256=obs_hash, excluded_cells=len(excluded),
        excluded_by_donor=dict(Counter(cell["donor"] for cell in excluded)),
        pseudotime_by_donor_annotation=pseudotime_summary(all_cells),
        unavailable_young_annotations=sorted(set(young) - set(eligible_young)),
        context_pool_by_annotation=context_pool_audit,
        ineligible_query_cells=[], splits={},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent))
    try:
        shutil.copyfile(source / "token_dictionary.json", staging / "token_dictionary.json")
        dump(staging / "young_reference_bank.json",
             sorted([cell for cells in young.values() for cell in cells], key=lambda c: c["cell_id"]))
        dump(staging / "excluded_cells.json", excluded)
        counts = {}
        for split_index, (split, donor) in enumerate((("val", "OM6"), ("test", "OM9"))):
            groups = {annotation: cells for (d, annotation), cells in pools.items()
                      if d == donor and annotation in eligible_young}
            audit["ineligible_query_cells"].extend(
                dict(cell_id=cell["cell_id"], donor=donor, trajectory=annotation,
                     reason="fewer_than_three_valid_young_context_cells")
                for (d, annotation), cells in pools.items()
                if d == donor and annotation not in eligible_young for cell in cells
            )
            queries = stratified_queries(groups, queries_per_donor, random.Random(seed + split_index))
            # Separate RNG ensures changing query values cannot affect young sampling.
            context_rng = random.Random(seed + 1000 + split_index)
            refs, tbc, nc = [], [], []
            for index, query in enumerate(queries):
                context = sample_context(context_pools[query["trajectory"]], context_rng)
                e = compose([cell["tokens"] for cell in context], [cell["time"] for cell in context],
                            query["tokens"], query["time"], tokens, numeric, model["time_scale"])
                if max(len(e["tbc_predict"]), len(e["nc_predict"]) + model["max_tokens"] - 1) > model["seq_length"]:
                    raise ValueError("Directed prompt exceeds original model sequence budget")
                tbc.append(dict(input_ids=e["tbc_predict"]))
                nc.append(dict(input_ids=e["nc_predict"]))
                refs.append({key: value for key, value in e.items()
                             if not key.endswith(("_train", "_predict"))} |
                            dict(row=index, query_cell_id=query["cell_id"], donor=donor,
                                 trajectory=query["trajectory"], query_pseudotime=query["time"],
                                 context_cell_ids=[cell["cell_id"] for cell in context],
                                 context_donors=[cell["donor"] for cell in context],
                                 context_pseudotimes=[cell["time"] for cell in context],
                                 context_tokens=[cell["tokens"] for cell in context],
                                 target_tokens=query["tokens"],
                                 copy_context_tokens=context[-1]["tokens"]))
            save(staging / f"{split}_tbc.dataset", tbc)
            save(staging / f"{split}_nc.dataset", nc)
            dump(staging / f"{split}_references.json", refs)
            signs = Counter("positive" if r["delta_pseudotime"] > 0 else
                            "negative" if r["delta_pseudotime"] < 0 else "zero" for r in refs)
            counts[split] = dict(trajectories=len(refs), tbc_rows=len(tbc), nc_rows=len(nc),
                                 eligible_query_cells=sum(map(len, groups.values())),
                                 eligible_annotations=sorted(groups),
                                 selected_by_annotation=dict(Counter(r["trajectory"] for r in refs)))
            audit["splits"][split] = dict(
                query_donor=donor, signed_pseudotime_direction_counts={s: signs[s] for s in ("positive", "zero", "negative")},
                interval_min=min(r["delta_pseudotime"] for r in refs),
                interval_max=max(r["delta_pseudotime"] for r in refs),
                selected_by_annotation=counts[split]["selected_by_annotation"])
        evaluation = model | dict(
            preparation_source=str(source.resolve()), preparation_source_manifest_sha256=digest(source / "manifest.json"),
            model_training_splits=model["splits"], model_training_counts=model["counts"],
            evaluation_protocol=protocol, evaluation_splits=evaluation_splits, counts=counts,
            eligible_young_reference_cells=sum(map(len, young.values())),
        )
        dump(staging / "audit.json", audit)
        dump(staging / "manifest.json", evaluation)
        if output.exists():
            raise FileExistsError("Output appeared during preparation; refusing to overwrite")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(dict(output=str(output), counts=counts, audit=audit["splits"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("out/aging_skm_stage2_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-donor", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--young-context-quantile", type=float, default=0.25)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
