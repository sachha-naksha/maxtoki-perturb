"""Prepare joint Stage 2 trajectories spanning young and old training donors.

Donor chronology orders contexts. Every interval remains a signed difference
of stored cell pseudotimes; donor age never replaces or rescales that value.
The held-out old donor supplies validation targets only, never training cells.
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
from prepare_aging_direction import pseudotime_summary, stratified_queries


def sample_time_bins(pool, count, rng):
    """Cover donor pseudotime bins, choosing distinct cells without query values."""
    if count < 1 or len(pool) < count:
        raise ValueError("Insufficient distinct donor cells for context")
    by_time = defaultdict(list)
    for cell in sorted(pool, key=lambda c: c["cell_id"]):
        by_time[cell["time"]].append(cell)
    times = sorted(by_time)
    if len(times) < count:
        selected = rng.sample(sorted(pool, key=lambda c: c["cell_id"]), count)
    else:
        selected = []
        for i in range(count):
            values = times[i * len(times) // count:(i + 1) * len(times) // count]
            selected.append(rng.choice([cell for time in values for cell in by_time[time]]))
    return sorted(selected, key=lambda c: (c["time"], c["cell_id"]))


def sample_context(pools, annotation, excluded_id, rng, young="YM2", old="OM6", row=0):
    """Three training-donor contexts; alternate 2 young/1 old and 1 young/2 old.

    Donor chronology, then within-donor pseudotime, defines sequence order.
    The query's ID is used only to prevent self-inclusion. Its pseudotime and
    expression are deliberately absent from this function's arguments.
    """
    counts = (2, 1) if row % 2 == 0 else (1, 2)
    context = []
    for donor, count in zip((young, old), counts):
        pool = [cell for cell in pools[(donor, annotation)] if cell["cell_id"] != excluded_id]
        context.extend(sample_time_bins(pool, count, rng))
    return context


def training_queries(groups, count, rng):
    """Balance donor and annotation; cycle each shuffled group when exhausted."""
    if count < 1 or not groups or any(not group for group in groups.values()):
        raise ValueError("Positive training count and nonempty donor/annotation groups required")
    queues = {key: [] for key in groups}
    result = []
    while len(result) < count:
        for key, group in sorted(groups.items()):
            if not queues[key]:
                queues[key] = sorted(group, key=lambda c: c["cell_id"])
                rng.shuffle(queues[key])
            result.append(queues[key].pop())
            if len(result) == count:
                break
    return result


def load_cells(model, tokens):
    """Read authoritative obs and reject stale IDs, times, donors and vocabulary."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from datasets import load_from_disk

    donor_col = model.get("donor_col", "sample")
    cols = [model["pseudotime_col"], donor_col, model["trajectory_col"]]
    h5ad = ad.read_h5ad(model["source_h5ad"], backed="r")
    try:
        obs = h5ad.obs.copy()
    finally:
        h5ad.file.close()
    if any(col not in obs for col in cols):
        raise ValueError(f"Missing source obs columns: {cols}")
    if hashlib.sha256(obs[cols].to_csv().encode()).hexdigest() != model["source_obs_sha256"]:
        raise ValueError("Source obs changed since original preparation")
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
    genes = {value for key, value in tokens.items() if key.startswith("ENSG")}
    pools, excluded = defaultdict(list), []
    for record in records:
        cell_id = str(record["cell_id"])
        row = obs.loc[cell_id]
        donor = str(row[donor_col])
        missing = [col for col in cols if pd.isna(row[col])]
        if missing:
            excluded.append(dict(cell_id=cell_id, donor=donor, reason="missing_obs", columns=missing))
            continue
        time = float(row[cols[0]])
        if not math.isfinite(time):
            excluded.append(dict(cell_id=cell_id, donor=donor, reason="nonfinite_pseudotime",
                                 columns=[cols[0]]))
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
    return pools, excluded


def prepare(args):
    compute()
    import pyarrow as pa
    from datasets import Dataset

    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a fresh Stage 2 trajectory output directory")
    young, old, validation = args.train_young, args.train_old, args.val_donor
    if len({young, old, validation}) != 3:
        raise ValueError("Young, old training and held-out validation donors must differ")
    if min(args.examples, args.eval_examples) < 1:
        raise ValueError("Positive training and validation example counts required")
    model = manifest(source)
    label_scalar = getattr(args, "label_scalar", None)
    if label_scalar is not None:
        if not math.isfinite(label_scalar) or label_scalar <= 0:
            raise ValueError("label_scalar must be finite and positive")
        model = model | dict(label_scalar=label_scalar)
    tokens = {key: int(value) for key, value in
              json.loads((source / "token_dictionary.json").read_text()).items()}
    numeric = dictionary_check(tokens, {})
    pools, excluded = load_cells(model, tokens)
    # Three valid cells in each training donor support both context allocations,
    # even after a same-donor query is excluded. Held-out metadata never affects
    # training annotation eligibility.
    annotations = sorted(annotation for donor, annotation in pools if donor == young
                         and len(pools[(young, annotation)]) >= 3
                         and len(pools.get((old, annotation), [])) >= 3)
    if not annotations:
        raise ValueError("No annotation has three valid cells in both training donors")
    train_groups = {(donor, annotation): pools[(donor, annotation)]
                    for donor in (young, old) for annotation in annotations}
    val_groups = {annotation: pools[(validation, annotation)] for annotation in annotations
                  if pools.get((validation, annotation))}
    if not val_groups:
        raise ValueError("No held-out validation cell matches an eligible training trajectory")
    queries = {
        "train": training_queries(train_groups, args.examples, random.Random(args.seed)),
        "val": stratified_queries(val_groups, args.eval_examples, random.Random(args.seed + 1)),
    }
    training_bank = sorted([cell for group in train_groups.values() for cell in group],
                           key=lambda c: c["cell_id"])
    protocol = dict(
        name="young_old_cross_donor_stage2_v1",
        biological_direction=f"{young} -> {old}; {validation} is the held-out old donor",
        context_donors=[young, old], training_query_donors=[young, old],
        validation_query_donor=validation, n_context=3,
        context_policy="Every context includes both training donors, ordered young then old, "
                       "and within donor by stored pseudotime. Alternate two young/one old "
                       "and one young/two old. Sample donor-specific pseudotime bins; exclude "
                       "the query cell. Query expression and query pseudotime never select context.",
        query_policy="Training queries balance donor and annotation and cycle shuffled cells. "
                     "Validation queries are unique and annotation-balanced; no query values "
                     "select them. Training eligibility depends only on training donors.",
        target="Stored query pseudotime minus last context pseudotime. Signed differences "
               "are retained even when donor chronology and pseudotime order disagree.",
        tasks=["tbc", "nc"], task_ratio="1:1",
        evaluation_status="Held-out-donor validation only; no independent test set.",
        metric_weighting="Report per-annotation and annotation-balanced validation metrics.",
    )
    all_cells = [cell for group in pools.values() for cell in group]
    audit = dict(
        excluded_cells=len(excluded),
        excluded_by_donor=dict(Counter(cell["donor"] for cell in excluded)),
        pseudotime_by_donor_annotation=pseudotime_summary(all_cells),
        eligible_training_annotations=annotations,
        ineligible_cells=[dict(cell_id=cell["cell_id"], donor=cell["donor"],
                               trajectory=cell["trajectory"],
                               reason="fewer_than_three_cells_in_a_training_donor")
                          for cell in all_cells if cell["trajectory"] not in annotations],
        splits={},
    )

    def save(path, rows):
        table = pa.Table.from_pylist(rows)
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            fingerprint = hashlib.sha256(sink.getvalue()).hexdigest()
        Dataset(table, fingerprint=fingerprint).save_to_disk(str(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent))
    try:
        shutil.copyfile(source / "token_dictionary.json", staging / "token_dictionary.json")
        dump(staging / "excluded_cells.json", excluded)
        dump(staging / "training_reference_bank.json", training_bank)
        counts = {}
        for split_index, split in enumerate(("train", "val")):
            context_rng = random.Random(args.seed + 1000 + split_index)
            mixed, tbc, nc, refs = [], [], [], []
            for index, query in enumerate(queries[split]):
                context = sample_context(pools, query["trajectory"], query["cell_id"],
                                         context_rng, young, old, index)
                e = compose([cell["tokens"] for cell in context], [cell["time"] for cell in context],
                            query["tokens"], query["time"], tokens, numeric, model["time_scale"])
                if max(len(e["tbc_train"]), len(e["nc_train"]),
                       len(e["nc_predict"]) + model["max_tokens"] - 1) > model["seq_length"]:
                    raise ValueError("Cross-donor trajectory exceeds model sequence budget")
                mixed.extend([dict(input_ids=e["tbc_train"]), dict(input_ids=e["nc_train"])])
                tbc.append(dict(input_ids=e["tbc_predict"]))
                nc.append(dict(input_ids=e["nc_predict"]))
                refs.append({key: value for key, value in e.items()
                             if not key.endswith(("_train", "_predict"))} |
                            dict(row=index, query_cell_id=query["cell_id"], donor=query["donor"],
                                 trajectory=query["trajectory"], query_pseudotime=query["time"],
                                 context_cell_ids=[cell["cell_id"] for cell in context],
                                 context_donors=[cell["donor"] for cell in context],
                                 context_pseudotimes=[cell["time"] for cell in context],
                                 context_tokens=[cell["tokens"] for cell in context],
                                 target_tokens=query["tokens"],
                                 copy_context_tokens=context[-1]["tokens"]))
            random.Random(args.seed + 2000 + split_index).shuffle(mixed)
            save(staging / f"{split}.dataset", mixed)
            dump(staging / f"{split}_references.json", refs)
            if split == "val":
                save(staging / "val_tbc.dataset", tbc)
                save(staging / "val_nc.dataset", nc)
            signs = Counter("positive" if r["delta_pseudotime"] > 0 else
                            "negative" if r["delta_pseudotime"] < 0 else "zero" for r in refs)
            counts[split] = dict(
                trajectories=len(refs), training_rows=len(mixed),
                unique_query_cells=len({r["query_cell_id"] for r in refs}),
                selected_by_donor=dict(Counter(r["donor"] for r in refs)),
                selected_by_annotation=dict(Counter(r["trajectory"] for r in refs)))
            audit["splits"][split] = dict(
                signed_interval_counts={sign: signs[sign] for sign in ("positive", "zero", "negative")},
                interval_min=min(r["delta_pseudotime"] for r in refs),
                interval_max=max(r["delta_pseudotime"] for r in refs),
                every_context_spans_training_donors=all(set(r["context_donors"]) == {young, old} for r in refs),
                negative_context_interval_count=sum(dt < 0 for r in refs for dt in r["context_intervals"]),
                training_cell_coverage=len({cell_id for r in refs for cell_id in r["context_cell_ids"]} |
                                           ({r["query_cell_id"] for r in refs} if split == "train" else set())),
            )
        result = model | dict(
            preparation_source=str(source.resolve()),
            preparation_source_manifest_sha256=digest(source / "manifest.json"),
            splits={"train": [young, old], "val": [validation]},
            runtime_protocol="aging_stage2_runtime_v1",
            numeric_expectation_normalized=True,
            nc_full_answer_loss=True,
            dataset_paths={"train": "train.dataset", "val": "val.dataset", "test": "val.dataset"},
            test_data_role="Validation alias required by upstream CLI; no independent test evaluation.",
            stage2_protocol=protocol, evaluation_protocol=protocol,
            evaluation_splits={"val": dict(context_donors=[young, old], query_donors=[validation])},
            counts=counts, seed=args.seed, n_context=3,
            training_reference_bank="training_reference_bank.json",
            baseline_fit_donors=[young, old],
            eligible_training_reference_cells=len(training_bank),
            excluded_cells=len(excluded), valid_cells=len(all_cells),
            excluded_by_donor=audit["excluded_by_donor"],
        )
        dump(staging / "audit.json", audit)
        dump(staging / "manifest.json", result)
        if output.exists():
            raise FileExistsError("Output appeared during preparation; refusing overwrite")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(dict(output=str(output), splits=result["splits"],
                          counts=counts, audit=audit["splits"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("out/aging_skm_stage2_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-young", default="YM2")
    parser.add_argument("--train-old", default="OM6")
    parser.add_argument("--val-donor", default="OM9")
    parser.add_argument("--examples", type=int, default=2000)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--label-scalar", type=float,
                        help="Override regression normalization; default inherits the source manifest")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
