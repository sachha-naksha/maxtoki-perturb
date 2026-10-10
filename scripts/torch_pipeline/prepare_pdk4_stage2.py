"""Rebuild a historical gene-inhibition TBC probe with current Stage 2 prompt grammar.

Retains every historical context/query gene rank and the query-only inhibition
operation. Original numeric token IDs are translated through their gene/token
names; context pseudotime intervals are then added without query-time input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from aging_temporal import compose, compute, dictionary_check, digest, dump, manifest

PDK4 = "ENSG00000004799"
GENES = {
    "PDK4": dict(ensembl=PDK4, token=63, n_present=983),
    "IRS2": dict(ensembl="ENSG00000185950", token=15393, n_present=256),
    # No historical IRS1 run: perturbation is derived from stored baseline prompts.
    "IRS1": dict(ensembl="ENSG00000169047", token=12277, n_present=None),
}
EXPECTED_ROWS = 2000
EXPECTED_CONTEXT_IDS = [
    "CELL1761_N1_2_1_11_1", "CELL3002_N1_2_1_11_1", "CELL1480_N1_1_11_1"]
EXPECTED_CONTEXT_TIMES = [1.0, 57.0, 100.0]
PROMPT_TOKEN_BUDGET = 8192
DEFAULT_LEGACY_DICTIONARY = Path(
    "/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/"
    "token_dictionary.before-id-fix.json")


def token_remapping(legacy, current):
    if len(set(legacy.values())) != len(legacy):
        raise ValueError("Ambiguous legacy dictionary")
    missing = set(legacy) - set(current)
    if missing:
        raise ValueError(f"Legacy token names absent from current vocabulary: {sorted(missing)[:5]}")
    return {int(old): int(current[name]) for name, old in legacy.items()}


def remap_sequence(sequence, remap):
    try:
        return [remap[int(token)] for token in sequence]
    except KeyError as error:
        raise ValueError(f"Unknown legacy token ID: {error.args[0]}") from error


def parse_legacy_tbc(sequence, tokens):
    """Strictly parse three untimed complete contexts and a query gene ranking."""
    seq = list(map(int, sequence))
    bos, eos, boq, eoq = (tokens[key] for key in ("<bos>", "<eos>", "<boq>", "<eoq>"))
    genes = {value for key, value in tokens.items() if key.startswith("ENSG")}
    numeric = set(dictionary_check(tokens, {}).values())
    if seq.count(boq) != 1 or seq.count(eoq) != 1:
        raise ValueError("Historical TBC prompt must have one BOQ and EOQ")
    query_start, query_end = seq.index(boq), seq.index(eoq)
    if query_end <= query_start or query_end != len(seq) - 2 or seq[-1] not in numeric:
        raise ValueError("Expected gene query and one numeric inference sentinel")
    contexts, position = [], 0
    while position < query_start:
        if seq[position] != bos:
            raise ValueError("Historical contexts must be contiguous BOS/genes/EOS cells")
        try:
            end = seq.index(eos, position + 1)
        except ValueError as error:
            raise ValueError("Context has no EOS") from error
        cell = seq[position:end + 1]
        if end >= query_start or not set(cell[1:-1]) <= genes:
            raise ValueError("Invalid context gene vocabulary")
        if len(set(cell[1:-1])) != len(cell[1:-1]):
            raise ValueError("Duplicate context gene")
        contexts.append(cell)
        position = end + 1
    query = seq[query_start + 1:query_end]
    if len(contexts) != 3 or not query or not set(query) <= genes:
        raise ValueError("Expected three contexts and a nonempty gene-only query")
    if len(set(query)) != len(query):
        raise ValueError("Duplicate query gene")
    return contexts, [bos, *query, eos]


def inhibit_query(query, gene, bos, eos):
    if not query or query[0] != bos or query[-1] != eos:
        raise ValueError("Expected a complete query cell")
    genes = list(query[1:-1])
    return [bos, *([x for x in genes if x != gene] + [gene] if gene in genes else genes), eos]


def official_record(old, row, remap, token):
    """Official-vocabulary twin of a legacy record when no corrected rerun exists."""
    return dict(old, input_ids=remap_sequence(old["input_ids"], remap), gene_token=token, row_index=row)


def derive_legacy_pair(old_base, legacy, ensembl):
    """Inhibit a gene with no stored run inside a historical baseline prompt."""
    context, query = parse_legacy_tbc(old_base["input_ids"], legacy)
    gene = legacy[ensembl]
    sentinel = old_base["input_ids"][-1]

    def prompt(cell):
        return [t for item in context for t in item] + [
            legacy["<boq>"], *cell[1:-1], legacy["<eoq>"], sentinel]

    if prompt(query) != list(old_base["input_ids"]):
        raise ValueError("Historical baseline prompt did not round-trip")
    meta = dict(old_base, gene_ensembl=ensembl, gene_token=gene,
                gene_present_in_query=gene in query[1:-1])
    inhibited = inhibit_query(query, gene, legacy["<bos>"], legacy["<eos>"])
    return (dict(meta, input_ids=prompt(query), condition="baseline"),
            dict(meta, input_ids=prompt(inhibited), condition="perturbed"))


def validate_pair(old_base, old_pert, base, pert, remap, tokens, row, gene=PDK4):
    """Reject changed query order, metadata, gene ranks, or perturbation semantics."""
    identity_fields = ("cell_id", "group", "query_pseudotime", "context_pseudotimes",
                       "context_cell_ids", "n_context_cells", "gene_ensembl", "direction",
                       "apply_to", "gene_present_in_query")
    for key in identity_fields:
        if any(record.get(key) != base.get(key) for record in (old_base, old_pert, pert)):
            raise ValueError(f"Row {row}: paired metadata mismatch: {key}")
    for record in (base, pert):
        if record.get("row_index", row) != row or record["gene_token"] != tokens[gene]:
            raise ValueError(f"Row {row}: incorrect row index or official {gene} ID")
    if any(remap.get(record["gene_token"]) != tokens[gene] for record in (old_base, old_pert)):
        raise ValueError(f"Row {row}: legacy {gene} ID does not remap correctly")
    for old, corrected, label in ((old_base, base, "baseline"), (old_pert, pert, "perturbed")):
        if remap_sequence(old["input_ids"], remap) != list(corrected["input_ids"]):
            raise ValueError(f"Row {row}: {label} gene ranking differs after vocabulary remapping")
        if old.get("condition") != label or corrected.get("condition") != label:
            raise ValueError(f"Row {row}: incorrect condition label")
    if (base["gene_ensembl"] != gene or base["direction"] != "inhibit"
            or base["apply_to"] != "query" or base["n_context_cells"] != 3):
        raise ValueError(f"Row {row}: not the historical query-only {gene} inhibition protocol")
    context, query = parse_legacy_tbc(base["input_ids"], tokens)
    pert_context, pert_query = parse_legacy_tbc(pert["input_ids"], tokens)
    expected = inhibit_query(query, tokens[gene], tokens["<bos>"], tokens["<eos>"])
    if pert_context != context or pert_query != expected:
        raise ValueError(f"Row {row}: inhibition must only move existing query {gene} to ranking end")
    present = tokens[gene] in query[1:-1]
    if bool(base["gene_present_in_query"]) != present:
        raise ValueError(f"Row {row}: stale gene-presence metadata")
    return context, query, pert_query, present


def timed_pair(context, times, query, pert_query, query_time, tokens, numeric, scale, budget):
    baseline = compose(context, times, query, query_time, tokens, numeric, scale)
    perturbed = compose(context, times, pert_query, query_time, tokens, numeric, scale)
    base_seq, pert_seq = baseline["tbc_predict"], perturbed["tbc_predict"]
    if max(len(base_seq), len(pert_seq)) > budget:
        raise ValueError("Historical gene rankings plus temporal tokens exceed prompt budget")
    if len(base_seq) != len(pert_seq):
        raise ValueError("Inhibition changed prompt length")
    # compose receives query_time for reference labels, but prediction sentinel
    # is always 0. No query-time label is passed to the inference prompt.
    if base_seq[-2:] != [tokens["<eoq>"], tokens["0"]] or pert_seq[-2:] != base_seq[-2:]:
        raise ValueError("TBC query-time target leaked into inference sentinel")
    return base_seq, pert_seq, {key: value for key, value in baseline.items()
                               if not key.endswith(("_train", "_predict"))}


def repeat_query_indices(references):
    present = [r["row"] for r in references if r["gene_present"]][:8]
    absent = [r["row"] for r in references if not r["gene_present"]][:4]
    return sorted(present + absent)


def dataset_digest(path):
    """Digest saved files and relative names, independent of absolute location."""
    files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    value = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value, files


def save_dataset(path, rows):
    import pyarrow as pa
    from datasets import Dataset
    table = pa.Table.from_pylist(rows)
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        fingerprint = hashlib.sha256(sink.getvalue()).hexdigest()
    Dataset(table, fingerprint=fingerprint).save_to_disk(str(path))
    return fingerprint


def prepare(args):
    compute()
    import anndata as ad
    from datasets import load_from_disk

    symbol = args.gene
    gene = GENES[symbol]
    original, training, output = map(Path, (args.original, args.training_data, args.output))
    corrected = Path(args.corrected) if args.corrected else None
    if output.exists():
        raise FileExistsError(f"Choose a fresh {symbol} Stage 2 output directory")
    model = manifest(training)
    if model["splits"] != {"train": ["YM2", "OM6"], "val": ["OM9"]}:
        raise ValueError("Expected corrected whole-trajectory Stage 2 training provenance")
    if model["seq_length"] != 16384:
        raise ValueError("Expected native Stage 2 16k configuration")
    tokens = {k: int(v) for k, v in json.loads((training / "token_dictionary.json").read_text()).items()}
    if tokens.get(gene["ensembl"]) != gene["token"]:
        raise ValueError(f"Official {symbol} token must be {gene['token']}")
    numeric = dictionary_check(tokens, {})
    legacy_path = Path(args.legacy_dictionary)
    legacy = json.loads(legacy_path.read_text())
    remap = token_remapping(legacy, tokens)
    sources = [
        load_from_disk(str(directory / f"{condition}.dataset"))
        for directory in (original, *([corrected] if corrected else []))
        for condition in ("baseline", "perturbed")]
    if any(len(dataset) != EXPECTED_ROWS for dataset in sources):
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} historical rows in all source datasets")
    old_base, old_pert = sources[:2]
    row_manifest = None
    if corrected:
        row_manifest = json.loads((corrected / "row_manifest.json").read_text())
        if row_manifest["n_rows"] != EXPECTED_ROWS or len(row_manifest["rows"]) != EXPECTED_ROWS:
            raise ValueError("Corrected source row manifest has inconsistent count")
    h5ad = ad.read_h5ad(model["source_h5ad"], backed="r")
    try:
        obs = h5ad.obs.copy()
    finally:
        h5ad.file.close()
    columns = [model["pseudotime_col"], model.get("donor_col", "sample"), model["trajectory_col"]]
    if hashlib.sha256(obs[columns].to_csv().encode()).hexdigest() != model["source_obs_sha256"]:
        raise ValueError("Authoritative obs changed since Stage 2 training")
    obs.index = obs.index.astype(str)
    if obs.index.has_duplicates:
        raise ValueError("Duplicate authoritative cell IDs")
    baseline_rows, perturbed_rows, paired_rows, refs = [], [], [], []
    fixed_context = None
    for row in range(EXPECTED_ROWS):
        records = [dataset[row] for dataset in sources]
        derived = records[0]["gene_ensembl"] != gene["ensembl"]
        if derived:
            if corrected:
                raise ValueError("A corrected rerun cannot belong to a different gene")
            records = list(derive_legacy_pair(records[0], legacy, gene["ensembl"]))
        if not corrected:
            # No official-vocabulary rerun exists: derive it from the legacy rows.
            records += [official_record(old, row, remap, gene["token"]) for old in records]
        base = records[2]
        context, query, pert_query, present = validate_pair(
            *records, remap, tokens, row, gene=gene["ensembl"])
        if base["context_cell_ids"] != EXPECTED_CONTEXT_IDS:
            raise ValueError(f"Row {row}: historical context IDs changed")
        if fixed_context is None:
            fixed_context = context
        elif context != fixed_context:
            raise ValueError("Historical fixed context gene rankings changed across queries")
        context_obs = obs.loc[EXPECTED_CONTEXT_IDS]
        times = [float(value) for value in context_obs[columns[0]]]
        if (times != EXPECTED_CONTEXT_TIMES or base["context_pseudotimes"] != times
                or set(context_obs[columns[1]].astype(str)) != {"YM2"}):
            raise ValueError("Historical contexts no longer match authoritative YM2 pseudotimes")
        query_id = str(base["cell_id"])
        if query_id in EXPECTED_CONTEXT_IDS:
            raise ValueError("Query appears in fixed contexts")
        query_obs = obs.loc[query_id]
        query_time = float(query_obs[columns[0]])
        donor, annotation = str(query_obs[columns[1]]), str(query_obs[columns[2]])
        if not math.isfinite(query_time) or base["query_pseudotime"] != query_time:
            raise ValueError(f"Row {row}: stale or missing query pseudotime")
        if donor != base["group"] or donor not in {"OM6", "OM9"}:
            raise ValueError(f"Row {row}: stale or unexpected query donor")
        expected_meta = dict(row_index=row, cell_id=query_id, group=donor,
            query_pseudotime=query_time, context_cell_ids=EXPECTED_CONTEXT_IDS,
            gene_present_in_query=present)
        if row_manifest and row_manifest["rows"][row] != expected_meta:
            raise ValueError(f"Row {row}: corrected row manifest does not match dataset")
        base_seq, pert_seq, interval_meta = timed_pair(context, times, query, pert_query,
            query_time, tokens, numeric, model["time_scale"], PROMPT_TOKEN_BUDGET)
        if not present and base_seq != pert_seq:
            raise ValueError(f"Row {row}: absent {symbol} must produce identical paired prompts")
        baseline_rows.append(dict(input_ids=base_seq))
        perturbed_rows.append(dict(input_ids=pert_seq))
        paired_rows.extend([baseline_rows[-1], perturbed_rows[-1]])
        refs.append(dict(row=row, legacy_row=row, query_cell_id=query_id, donor=donor,
            trajectory=annotation, gene_present=present, query_pseudotime=query_time,
            context_cell_ids=list(EXPECTED_CONTEXT_IDS), context_pseudotimes=times,
            context_donors=["YM2"] * 3, raw_target_delta=query_time - times[-1],
            paired_baseline_row=2 * row, paired_perturbed_row=2 * row + 1, **interval_meta))
    if len({r["query_cell_id"] for r in refs}) != EXPECTED_ROWS:
        raise ValueError("Historical queries are not unique")
    if gene["n_present"] is not None and sum(r["gene_present"] for r in refs) != gene["n_present"]:
        raise ValueError(f"Historical {symbol}-present count changed")
    repeats = repeat_query_indices(refs)
    repeat_rows = [paired_rows[2 * row + condition] for row in repeats for condition in (0, 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".preparing.", dir=output.parent))
    try:
        shutil.copyfile(training / "token_dictionary.json", temporary / "token_dictionary.json")
        dump(temporary / "references.json", refs)
        dump(temporary / "repeat_references.json", [refs[row] for row in repeats])
        fingerprints, checksums, files = {}, {}, {}
        for name, rows in (("baseline", baseline_rows), ("perturbed", perturbed_rows),
                           ("paired", paired_rows), ("repeat", repeat_rows)):
            path = temporary / f"{name}.dataset"
            fingerprints[name] = save_dataset(path, rows)
            checksums[name], files[name] = dataset_digest(path)
        prepared = deepcopy(model)
        prepared["prompt_token_budget"] = PROMPT_TOKEN_BUDGET
        prepared["perturbation_protocol"] = dict(
            name=f"historical_{symbol.lower()}_inhibit_timed_stage2_v1", gene_symbol=symbol,
            gene_ensembl=gene["ensembl"], gene_token=tokens[gene["ensembl"]], direction="inhibit",
            apply_to="query",
            operation=f"Move existing {symbol} token to the end of query gene ranking; absent gene unchanged.",
            n_queries=len(refs), n_gene_present=sum(r["gene_present"] for r in refs),
            query_donors=dict(Counter(r["donor"] for r in refs)),
            context_cell_ids=EXPECTED_CONTEXT_IDS, context_pseudotimes=EXPECTED_CONTEXT_TIMES,
            context_intervals=[56.0, 43.0], context_donors=["YM2"] * 3,
            context_policy="Exact historical fixed YM2 context; not annotation matched. "
                           "This is a historical perturbation probe, not the Stage 2 donor split.",
            query_time_input=False, inference_sentinel="0",
            pairing="paired.dataset rows [baseline0, perturbed0, baseline1, perturbed1, ...]",
            interpretation="OM6 is a training donor; OM9 is a checkpoint-selection validation donor. "
                           "Neither donor provides measured intervention outcomes.",
            changes_from_original=["Official token vocabulary", "Known inter-context intervals inserted",
                                   "Current Stage 2 checkpoint/runtime", "Numeric zero inference sentinel"],
            max_prompt_tokens=max(len(row["input_ids"]) for row in paired_rows))
        prepared["prepared_pair"] = dict(
            original=str(original.resolve()),
            corrected=str(corrected.resolve()) if corrected else None,
            official_sequences=("corrected rerun cross-checked against legacy remap" if corrected
                                else "legacy rows remapped through token names; no official rerun"),
            perturbation_source=("derived: gene moved to ranking end within the stored baseline "
                                 "prompts of a different historical run" if derived
                                 else "stored historical perturbed rows"),
            training_data=str(training.resolve()), legacy_dictionary=str(legacy_path.resolve()),
            training_manifest_sha256=digest(training / "manifest.json"),
            legacy_dictionary_sha256=digest(legacy_path),
            corrected_row_manifest_sha256=(
                digest(corrected / "row_manifest.json") if corrected else None),
            original_spec_sha256=digest(original / "spec.resolved.json"),
            references_sha256=digest(temporary / "references.json"),
            repeat_references_sha256=digest(temporary / "repeat_references.json"),
            dataset_fingerprints=fingerprints, dataset_sha256=checksums, dataset_file_sha256=files,
            legacy_remap_exact=True, rows_and_gene_presence_exact=True,
            paired_reference_order="baseline then perturbed for each references.json row",
            repeat_query_rows=repeats, repeat_paired_source_rows=[
                2 * row + condition for row in repeats for condition in (0, 1)])
        dump(temporary / "manifest.json", prepared)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(dict(output=str(output), n_queries=len(refs),
        n_gene_present=sum(r["gene_present"] for r in refs),
        max_prompt_tokens=prepared["perturbation_protocol"]["max_prompt_tokens"],
        context_intervals=[56, 43], repeat_query_rows=repeats), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "training-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corrected", type=Path, default=None,
                        help="Official-vocabulary rerun with row_manifest.json; optional cross-check")
    parser.add_argument("--gene", choices=sorted(GENES), default="PDK4")
    parser.add_argument("--legacy-dictionary", type=Path, default=DEFAULT_LEGACY_DICTIONARY)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
