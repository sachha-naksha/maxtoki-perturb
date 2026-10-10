"""Evaluate young-to-old SKM trajectories using prompt-matched baselines.

The fitted baselines use only the donors assigned to model training. Query expression is used for TBC,
and only the supplied time interval is used for NC. Absolute query pseudotime
is used solely to score predictions. CLI workloads require a Slurm allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from aging_baselines import (context_linear_nc, context_linear_tbc,
                            nearest_expression_tbc, rank_features)
from aging_temporal import compute, dump, interval, manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gene_matrix(cells, gene_ids, max_genes=2046):
    """Sparse fixed-cap rank features in deterministic vocabulary order."""
    genes = sorted(set(map(int, gene_ids)))
    columns = {g: i for i, g in enumerate(genes)}
    rows, cols, values = [], [], []
    for i, tokens in enumerate(cells):
        for gene, value in rank_features(tokens, genes, max_genes).items():
            rows.append(i)
            cols.append(columns[gene])
            values.append(value)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(cells), len(genes)))


def fit_young_bank(bank, gene_ids, max_genes=2046, alpha=10.0, allowed_donors=("YM2",)):
    """Fit per-Annotation clocks/trends on declared training donors only.

    The default retains the original YM2-only evaluation protocol.
    """
    from sklearn.linear_model import Ridge

    allowed = set(allowed_donors)
    if not allowed or not bank or any(r["donor"] not in allowed for r in bank):
        raise ValueError(f"Baseline fit must contain training donors {sorted(allowed)} only")
    if {r["donor"] for r in bank} != allowed:
        raise ValueError("Baseline bank must represent every declared training donor")
    if len({r["cell_id"] for r in bank}) != len(bank):
        raise ValueError("Duplicate cells in training reference bank")
    genes = sorted(set(map(int, gene_ids)))
    groups = defaultdict(list)
    for row in bank:
        if not np.isfinite(row["time"]):
            raise ValueError("Nonfinite training reference pseudotime")
        groups[row["trajectory"]].append(row)
    fits, provenance = {}, {}
    for annotation, rows in sorted(groups.items()):
        times = np.asarray([r["time"] for r in rows], dtype=float)
        record = dict(n_cells=len(rows), donors=sorted({r["donor"] for r in rows}),
                      cell_ids=[r["cell_id"] for r in rows],
                      min_pseudotime=float(times.min()), max_pseudotime=float(times.max()),
                      clock=dict(estimator="Ridge", alpha=alpha, solver="lsqr", tol=1e-4),
                      trajectory="per-gene ordinary least squares of rank score on pseudotime")
        fits[annotation] = dict(available=False, reason="insufficient_training_time_variation")
        if len(times) < 2 or np.ptp(times) <= 1e-12:
            provenance[annotation] = record | fits[annotation]
            continue
        matrix = gene_matrix([r["tokens"] for r in rows], genes, max_genes)
        if np.any(np.asarray(matrix.getnnz(axis=1)) == 0):
            raise ValueError("Training reference bank contains an empty gene profile")
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-4).fit(matrix, times)
        center = times - times.mean()
        mean_features = np.asarray(matrix.mean(axis=0)).ravel()
        slope = np.asarray(matrix.T @ center).ravel() / float(center @ center)
        intercept = mean_features - slope * float(times.mean())
        available = bool(np.linalg.norm(slope) > 1e-12 and
                         np.linalg.norm(estimator.coef_) > 1e-12)
        reason = None if available else "no_training_gene_time_trend"
        fits[annotation] = dict(available=available, reason=reason, gene_ids=genes,
                                clock_coef=np.asarray(estimator.coef_),
                                clock_intercept=float(estimator.intercept_),
                                trend_intercept=intercept, trend_slope=slope,
                                time_min=float(times.min()), time_max=float(times.max()),
                                max_genes=max_genes)
        provenance[annotation] = record | dict(available=available, reason=reason)
    return fits, provenance


def bank_predict(fit, context_tokens, relative_times, query_tokens, query_delta):
    """Clock alignment is inferred from context; query time never enters TBC."""
    if not fit or not fit["available"]:
        reason = fit["reason"] if fit else "annotation_absent_from_training_fit"
        return (dict(available=False, prediction=None, reason=reason, diagnostics={}),
                dict(available=False, gene_tokens=[], reason=reason, diagnostics={}))
    times = np.asarray(relative_times, dtype=float)
    if (len(times) != len(context_tokens) or not np.isfinite(times).all() or
            not np.isfinite(query_delta)):
        raise ValueError("Finite prompt times and matching context required")
    context = gene_matrix(context_tokens, fit["gene_ids"], fit["max_genes"])
    context_clocks = np.asarray(context @ fit["clock_coef"]).ravel() + fit["clock_intercept"]
    origin = float(np.mean(context_clocks - times))
    query = gene_matrix([query_tokens], fit["gene_ids"], fit["max_genes"])
    query_clock = float(np.asarray(query @ fit["clock_coef"]).ravel()[0] + fit["clock_intercept"])
    tbc = dict(available=True, prediction=query_clock-origin, reason=None,
               diagnostics=dict(estimated_absolute_query_time=query_clock,
                                estimated_context_time_origin=origin,
                                estimated_query_outside_fit_range=bool(
                                    query_clock < fit["time_min"] or query_clock > fit["time_max"])))
    target_time = origin + float(query_delta)
    scores = np.maximum(fit["trend_intercept"] + fit["trend_slope"] * target_time, 0.)
    ranked = sorted(((int(g), float(s)) for g, s in zip(fit["gene_ids"], scores) if s > 0),
                    key=lambda item: (-item[1], item[0]))
    nc = dict(available=bool(ranked), gene_tokens=[g for g, _ in ranked[:fit["max_genes"]]],
              reason=None if ranked else "no_positive_extrapolated_gene_scores",
              diagnostics=dict(estimated_context_time_origin=origin,
                               supplied_query_delta=float(query_delta),
                               estimated_absolute_query_time=target_time,
                               estimated_query_outside_fit_range=bool(
                                   target_time < fit["time_min"] or target_time > fit["time_max"]),
                               positive_gene_count=len(ranked)))
    return tbc, nc


def prompt_times(ref, token_dict, time_scale):
    """Reconstruct times from exactly the numeric tokens visible in the prompt."""
    numeric = {}
    for key, token in token_dict.items():
        try:
            number = float(key)
        except ValueError:
            continue
        numeric[number] = int(token)
    times = np.asarray(ref["context_pseudotimes"], dtype=float)
    if len(times) != len(ref["context_tokens"]) or len(times) < 2:
        raise ValueError("Malformed context reference")
    intervals = [interval(float(t), numeric, time_scale)[1] for t in np.diff(times)]
    relative = np.concatenate(([0.], np.cumsum(intervals)))
    relative -= relative[-1]
    return relative.tolist(), float(ref["represented_delta_pseudotime"])


def predict_baselines(ref, fits, gene_ids, token_dict, time_scale, max_genes=2046,
                      fitted_prefix="young"):
    """Separate task inputs so NC cannot observe target expression."""
    context = ref["context_tokens"]
    relative, query_delta = prompt_times(ref, token_dict, time_scale)
    tbc_bank, nc_bank = bank_predict(fits.get(ref["trajectory"]), context, relative,
                                    ref["target_tokens"], query_delta)
    return dict(row=ref["row"], query_cell_id=ref["query_cell_id"],
                tbc={
                    "context_linear": context_linear_tbc(context, relative, ref["target_tokens"],
                                                          gene_ids, max_genes),
                    f"{fitted_prefix}_ridge_clock": tbc_bank,
                    "nearest_expression_diagnostic": nearest_expression_tbc(
                        context, relative, ref["target_tokens"], gene_ids, max_genes)},
                nc={
                    "context_linear": context_linear_nc(context, relative, query_delta,
                                                        gene_ids, max_genes, max_genes),
                    f"{fitted_prefix}_linear_trend": nc_bank})


def ref_metadata(ref, split):
    delta = float(ref["delta_pseudotime"])
    return dict(row=int(ref["row"]), query_cell_id=ref["query_cell_id"],
                donor=ref["donor"], trajectory=ref["trajectory"], split=split,
                target_delta=delta, delta_sign="positive" if delta > 0 else
                "negative" if delta < 0 else "zero")


def nc_metrics(predicted, target):
    """Jaccard at fixed cutoffs plus true Spearman on shared full-list ranks."""
    from scipy.stats import spearmanr
    p, t = list(dict.fromkeys(predicted)), list(dict.fromkeys(target))
    out = {}
    for k in (100, 500):
        a, b = set(p[:k]), set(t[:k])
        out[f"jaccard_at_{k}"] = len(a & b)/len(a | b) if a or b else None
    pi, ti = {g:i for i,g in enumerate(p)}, {g:i for i,g in enumerate(t)}
    shared = sorted(set(pi) & set(ti))
    rho = float(spearmanr([pi[g] for g in shared], [ti[g] for g in shared]).statistic) if len(shared)>1 else None
    out.update(spearman_shared=rho, shared_genes=len(shared), predicted_genes=len(p))
    return out


def baseline_records(refs, predictions, split, gene_ids):
    if len(refs) != len(predictions):
        raise ValueError("Baseline/reference count mismatch")
    records = []
    for ref, pred in zip(refs, predictions):
        if (ref["row"], ref["query_cell_id"]) != (pred["row"], pred["query_cell_id"]):
            raise ValueError("Baseline/reference row alignment mismatch")
        metadata = ref_metadata(ref, split)
        target_genes = list(dict.fromkeys(int(g) for g in ref["target_tokens"] if int(g) in gene_ids))
        for task in ("tbc", "nc"):
            for method, result in pred[task].items():
                record = metadata | dict(task=task, method=method, available=result["available"],
                                         reason=result["reason"])
                if result["available"]:
                    if task == "tbc":
                        value = float(result["prediction"])
                        if not np.isfinite(value):
                            raise ValueError(f"Nonfinite baseline: {method}")
                        error = value - metadata["target_delta"]
                        record.update(predicted_delta=value, absolute_error=abs(error), squared_error=error**2)
                    else:
                        record.update(nc_metrics(result["gene_tokens"], target_genes))
                records.append(record)
    return records


def aggregate(records, task):
    available = [r for r in records if r["available"]]
    result = dict(n_total=len(records), n_available=len(available),
                  coverage=len(available)/len(records) if records else None,
                  n_unique_queries=len({r["query_cell_id"] for r in records}),
                  n_available_unique_queries=len({r["query_cell_id"] for r in available}),
                  unavailable_reasons=dict(Counter(r["reason"] for r in records if not r["available"])))
    if task == "tbc":
        p = np.asarray([r["predicted_delta"] for r in available], dtype=float)
        t = np.asarray([r["target_delta"] for r in available], dtype=float)
        result.update(mae=float(np.abs(p-t).mean()) if len(p) else None,
                      mse=float(np.square(p-t).mean()) if len(p) else None,
                      pearson=float(np.corrcoef(p,t)[0,1]) if len(p)>1 and p.std()>0 and t.std()>0 else None)
    else:
        for metric in ("jaccard_at_100","jaccard_at_500","spearman_shared"):
            values = [r[metric] for r in available if r.get(metric) is not None]
            result[metric] = float(np.mean(values)) if values else None
            result[f"{metric}_n_defined"] = len(values)
        for name in ("invalid_tokens", "duplicate_tokens"):
            result[name] = sum(r.get(name,0) for r in available)
        for name in ("saw_eos", "finished_naturally", "hit_generation_limit"):
            values = [r[name] for r in available if name in r]
            result[f"{name}_count"] = sum(values) if values else None
    return result


def grouped_report(records, task):
    def grouped(keys):
        buckets = defaultdict(list)
        for record in records:
            buckets[" / ".join(str(record[k]) for k in keys)].append(record)
        return {key: aggregate(rows, task) for key, rows in sorted(buckets.items())}
    by_annotation = grouped(("trajectory",))
    by_donor_annotation = grouped(("donor","trajectory"))
    metrics = ("mae","mse") if task == "tbc" else ("jaccard_at_100","jaccard_at_500","spearman_shared")
    def macro(groups):
        out = dict(n_annotations=len(groups))
        for key in metrics:
            values = [g[key] for g in groups if g.get(key) is not None]
            out[key] = float(np.mean(values)) if values else None
            out[f"{key}_n_annotations"] = len(values)
        return out
    donor_macro = {}
    for donor in sorted({r["donor"] for r in records}):
        donor_macro[donor] = macro([v for k,v in by_donor_annotation.items() if k.startswith(donor+" / ")])
    return dict(overall=aggregate(records, task), by_donor=grouped(("donor",)),
                by_annotation=by_annotation, annotation_macro=macro(list(by_annotation.values())),
                by_delta_sign=grouped(("delta_sign",)), by_donor_annotation=by_donor_annotation,
                by_donor_delta_sign=grouped(("donor","delta_sign")), annotation_macro_by_donor=donor_macro)


def summarize(records):
    result = {}
    for task in ("tbc","nc"):
        result[task] = {}
        for method in sorted({r["method"] for r in records if r["task"]==task}):
            selected = [r for r in records if r["task"]==task and r["method"]==method]
            result[task][method] = grouped_report(selected,task)
    return result


def evaluation_protocol(m):
    """Resolve explicit fitting/evaluation donors; never infer them from targets."""
    bank_name = m.get("training_reference_bank")
    if bank_name is not None:
        if not isinstance(bank_name, str) or Path(bank_name).name != bank_name:
            raise ValueError("Training reference bank must be a local manifest filename")
        splits = m.get("splits", {})
        training = splits.get("train", [])
        evaluation = m.get("evaluation_splits") or {
            split: dict(context_donors=training, query_donors=donors)
            for split, donors in splits.items() if split != "train" and donors
        }
        fitted_prefix = "training"
    else:
        # Compatibility with the original directed-evaluation manifests and fixtures.
        bank_name, training, fitted_prefix = "young_reference_bank.json", ["YM2"], "young"
        declared = m.get("model_training_splits", m.get("splits"))
        if declared is not None and set(declared.get("train", [])) != {"YM2"}:
            raise ValueError("Multidonor training requires an explicit training reference bank")
        evaluation = m.get("evaluation_splits") or {
            "val": dict(context_donors=["YM2"], query_donors=["OM6"]),
            "test": dict(context_donors=["YM2"], query_donors=["OM9"]),
        }
    if not training or len(training) != len(set(training)) or not evaluation:
        raise ValueError("Nonempty unique training donors and evaluation splits required")
    seen = set(training)
    for split, specification in evaluation.items():
        if split not in ("val", "test"):
            raise ValueError("Evaluation splits must be val or test")
        contexts = specification.get("context_donors", [])
        queries = specification.get("query_donors", [])
        if not contexts or not set(contexts) <= set(training):
            raise ValueError("Evaluation contexts must come from training donors")
        if not queries or len(queries) != len(set(queries)) or seen.intersection(queries):
            raise ValueError("Training and held-out query donor splits must be disjoint")
        seen.update(queries)
        if fitted_prefix == "training" and set(queries) != set(m["splits"].get(split, [])):
            raise ValueError("Evaluation query donors differ from model split")
    return dict(bank_name=bank_name, training_donors=list(training),
                evaluation_splits=evaluation, fitted_prefix=fitted_prefix)


def read_data(data):
    m = manifest(data)
    protocol = evaluation_protocol(m)
    tokens = {k:int(v) for k,v in json.loads((data/"token_dictionary.json").read_text()).items()}
    genes = {v for k,v in tokens.items() if k.startswith("ENSG")}
    refs = {split:json.loads((data/f"{split}_references.json").read_text())
            for split in protocol["evaluation_splits"]}
    for split, rows in refs.items():
        specification = protocol["evaluation_splits"][split]
        if not rows:
            raise ValueError("Evaluation reference split is empty")
        for row in rows:
            if (row["donor"] not in specification["query_donors"] or
                    not row["context_donors"] or
                    not set(row["context_donors"]) <= set(specification["context_donors"])):
                raise ValueError(f"Evaluation reference donor violates {split} protocol")
            ids = row["context_cell_ids"]
            if (row["query_cell_id"] in ids or len(set(ids)) != len(ids) or
                    len(ids) != len(row["context_donors"]) or
                    len(ids) != len(row["context_tokens"]) or
                    len(ids) != len(row["context_pseudotimes"])):
                raise ValueError("Query/context leakage or malformed context cells")
        if [r["row"] for r in rows] != list(range(len(rows))):
            raise ValueError("Noncontiguous reference rows")
    return m,tokens,genes,refs


def reference_bank(data, m, refs):
    protocol = evaluation_protocol(m)
    bank_path = data/protocol["bank_name"]
    bank = json.loads(bank_path.read_text())
    by_id = {r["cell_id"]: r for r in bank}
    for rows in refs.values():
        for row in rows:
            if row["query_cell_id"] in by_id:
                raise ValueError("Evaluation query leaked into baseline fit")
            for i, cell_id in enumerate(row["context_cell_ids"]):
                cell = by_id.get(cell_id)
                if cell is None:
                    raise ValueError("Evaluation context is absent from training reference bank")
                if (cell["donor"] != row["context_donors"][i] or
                        cell["trajectory"] != row["trajectory"] or
                        cell["tokens"] != row["context_tokens"][i] or
                        float(cell["time"]) != float(row["context_pseudotimes"][i])):
                    raise ValueError("Evaluation context differs from training reference bank")
    if len(by_id) != len(bank):
        raise ValueError("Duplicate cells in training reference bank")
    if any(r["donor"] not in protocol["training_donors"] for r in bank):
        raise ValueError("Reference bank contains held-out donors")
    return bank_path, bank, protocol


def fitting_bank(data, bank, protocol):
    """Match the exact cells appearing in supervised model training rows."""
    by_id = {cell["cell_id"]: cell for cell in bank}
    audit = dict(source_split="train", eligible_bank_cells=len(bank))
    if protocol["fitted_prefix"] == "training":
        path = data/"train_references.json"
        rows = json.loads(path.read_text())
        if not rows:
            raise ValueError("Training references are empty")
        selected = set()
        for row in rows:
            query_id = row["query_cell_id"]
            if query_id in row["context_cell_ids"]:
                raise ValueError("Training query occurs in its own context")
            observed = [(query_id, row["donor"], row["query_pseudotime"], row["target_tokens"])]
            lengths = [len(row[key]) for key in
                       ("context_cell_ids", "context_donors", "context_pseudotimes", "context_tokens")]
            if len(set(lengths)) != 1:
                raise ValueError("Malformed training reference context")
            observed.extend(zip(row["context_cell_ids"], row["context_donors"],
                                row["context_pseudotimes"], row["context_tokens"]))
            for cell_id, donor, time, tokens in observed:
                cell = by_id.get(cell_id)
                if (cell is None or donor not in protocol["training_donors"] or
                        cell["donor"] != donor or cell["trajectory"] != row["trajectory"] or
                        float(cell["time"]) != float(time) or cell["tokens"] != tokens):
                    raise ValueError("Training reference cell differs from training bank")
                selected.add(cell_id)
        audit.update(selection="union_of_training_query_and_context_cells",
                     training_references_sha256=sha256(path), n_training_trajectories=len(rows))
    else:
        selected = set(by_id)
        audit.update(selection="all_legacy_young_reference_cells")
    cell_ids = sorted(selected)
    audit.update(n_fit_cells=len(cell_ids),
                 fit_cell_ids_sha256=hashlib.sha256(
                     json.dumps(cell_ids, separators=(",", ":")).encode()).hexdigest())
    return [by_id[cell_id] for cell_id in cell_ids], audit


def baselines(a):
    compute()
    if a.output.exists():
        raise FileExistsError("Choose a fresh baseline output directory")
    m,tokens,genes,refs = read_data(a.data)
    bank_path, bank, protocol = reference_bank(a.data, m, refs)
    fit_bank, fit_audit = fitting_bank(a.data, bank, protocol)
    max_genes = int(m["cell_cap"])-2
    fits, fitted_provenance = fit_young_bank(
        fit_bank, genes, max_genes, allowed_donors=protocol["training_donors"])
    predictions = {split:[predict_baselines(
        r, fits, genes, tokens, m["time_scale"], max_genes, protocol["fitted_prefix"])
        for r in rows] for split,rows in refs.items()}
    records = [r for split in refs for r in baseline_records(refs[split],predictions[split],split,genes)]
    a.output.mkdir(parents=True)
    arrays = {}
    for i,(annotation,fit) in enumerate(sorted(fits.items())):
        if fit["available"]:
            prefix = f"annotation_{i}"
            fitted_provenance[annotation]["array_prefix"] = prefix
            for key in ("gene_ids","clock_coef","clock_intercept","trend_intercept","trend_slope"):
                arrays[f"{prefix}_{key}"] = np.asarray(fit[key])
    np.savez_compressed(a.output/"fitted_baselines.npz",**arrays)
    provenance = dict(data=str(a.data.resolve()), manifest_sha256=sha256(a.data/"manifest.json"),
                      tokenizer_sha256=m["tokenizer_sha256"], bank_sha256=sha256(bank_path),
                      references_sha256={s:sha256(a.data/f"{s}_references.json") for s in refs},
                      bank_filename=protocol["bank_name"],
                      training_donors=protocol["training_donors"],
                      evaluation_splits=protocol["evaluation_splits"],
                      training_fit=fit_audit,
                      fitted_annotations=fitted_provenance,
                      max_genes=max_genes, rank_score="1 - one_based_rank / (max_genes + 1); absent=0",
                      timing="quantized prompt context intervals anchored to last context; NC uses supplied quantized interval",
                      target="exact stored query pseudotime minus last context pseudotime",
                      task_inputs=dict(tbc="context genes, relative times, query genes",
                                       nc="context genes, relative times, supplied query interval"),
                      evaluation="Donor-held-out evaluation with fitting and context cells restricted to model training donors",
                      secondary_methods=["nearest_expression_diagnostic"])
    dump(a.output/"provenance.json",provenance)
    dump(a.output/"baseline_predictions.json",predictions)
    dump(a.output/"baseline_per_row.json",records)
    report = dict(provenance=provenance, by_split={
        split:summarize([r for r in records if r["split"]==split]) for split in refs})
    dump(a.output/"baseline_metrics.json",report)
    print(json.dumps({s:{t:{k:v["overall"] for k,v in methods.items()}
                         for t,methods in values.items()} for s,values in report["by_split"].items()},indent=2))


def flatten_tbc(value):
    import torch
    if isinstance(value,(list,tuple)):
        return [v for item in value for v in flatten_tbc(item)]
    if isinstance(value,dict):
        if "regression_preds" in value:
            return torch.as_tensor(value["regression_preds"]).float().reshape(-1).tolist()
        if "predictions" in value:
            return flatten_tbc(value["predictions"])
    raise ValueError("Unrecognized TBC prediction payload")


def model_records(predictions_dir, data, m, tokens, genes, refs, task, split):
    run = json.loads((predictions_dir/"run.json").read_text())
    if "invalid_reason" in run:
        raise ValueError(run["invalid_reason"])
    for key,expected in (("task",task),("split",split),("tokenizer_sha256",m["tokenizer_sha256"]),
                         ("rope_factor",m["rope_factor"]),("attention_backend",m["attention_backend"]),
                         ("time_scale",m["time_scale"]),("label_scalar",m["label_scalar"])):
        if run.get(key)!=expected:
            raise ValueError(f"Prediction setting mismatch: {key}")
    if m.get("runtime_protocol") and run.get("runtime_protocol") != m["runtime_protocol"]:
        raise ValueError("Prediction setting mismatch: runtime_protocol")
    if run.get("references_sha256") != sha256(data/f"{split}_references.json"):
        raise ValueError("Prediction reference hash mismatch")
    # Container bind aliases can name identical data with different absolute paths.
    if run.get("manifest_sha256") != sha256(data/"manifest.json"):
        raise ValueError("Prediction manifest hash mismatch")
    count = run.get("limit")
    selected = refs if count is None else refs[:count]
    paths = list(predictions_dir.glob("predictions__rank_*.pt"))
    if len(paths)!=1:
        raise ValueError("Require exactly one GPU rank file for deterministic row alignment")
    if task=="tbc":
        import torch
        values = [v/m["time_scale"] for v in flatten_tbc(torch.load(paths[0],map_location="cpu",weights_only=False))]
    else:
        from score_nextcell import (_extract_per_row_tokens, decode_generation,
                                    _build_id_to_ensg, _special_token_ids, _numeric_token_ids)
        values = _extract_per_row_tokens(predictions_dir)
        id_to_ensg, specials, numeric = _build_id_to_ensg(tokens), _special_token_ids(tokens), _numeric_token_ids(tokens)
    if len(values)!=len(selected):
        raise ValueError(f"Model/reference count mismatch: {len(values)} vs {len(selected)}")
    records=[]
    for ref,value in zip(selected,values):
        record=ref_metadata(ref,split)|dict(task=task,method="maxtoki_stage2",available=True,reason=None)
        if task=="tbc":
            if not np.isfinite(value):
                record.update(available=False,reason="nonfinite_model_prediction")
            else:
                error=value-record["target_delta"]
                record.update(predicted_delta=value,absolute_error=abs(error),squared_error=error**2)
        else:
            decoded=decode_generation(value["tokens"],id_to_ensg,specials,numeric)
            target=[id_to_ensg[g] for g in ref["target_tokens"] if g in id_to_ensg]
            record.update(nc_metrics(decoded.ensg_order,target),
                          invalid_tokens=decoded.n_invalid, duplicate_tokens=decoded.n_duplicates,
                          saw_eos=decoded.saw_eos, finished_naturally=bool(value["finished"]),
                          hit_generation_limit=bool(not decoded.saw_eos and value["length"]>=m["max_tokens"]),
                          generated_length=int(value["length"]))
        records.append(record)
    return records,selected,run


def compare(a):
    compute()
    if a.output.exists():
        raise FileExistsError("Choose a fresh comparison output directory")
    m,tokens,genes,refs=read_data(a.data)
    baseline_provenance=json.loads((a.baselines/"provenance.json").read_text())
    if baseline_provenance["manifest_sha256"]!=sha256(a.data/"manifest.json"):
        raise ValueError("Baseline data manifest mismatch")
    if baseline_provenance["tokenizer_sha256"]!=m["tokenizer_sha256"]:
        raise ValueError("Baseline tokenizer mismatch")
    bank_path, bank, protocol = reference_bank(a.data, m, refs)
    if baseline_provenance.get("training_donors") != protocol["training_donors"]:
        raise ValueError("Baseline training donor mismatch")
    if baseline_provenance.get("bank_filename", "young_reference_bank.json") != protocol["bank_name"]:
        raise ValueError("Baseline training reference bank filename mismatch")
    if baseline_provenance["bank_sha256"]!=sha256(bank_path):
        raise ValueError("Baseline training reference bank mismatch")
    if protocol["fitted_prefix"] == "training":
        _, fit_audit = fitting_bank(a.data, bank, protocol)
        if baseline_provenance.get("training_fit") != fit_audit:
            raise ValueError("Baseline fitting cells differ from model training cells")
    for split in refs:
        if baseline_provenance["references_sha256"].get(split)!=sha256(a.data/f"{split}_references.json"):
            raise ValueError(f"Baseline reference hash mismatch: {split}")
    baseline_predictions=json.loads((a.baselines/"baseline_predictions.json").read_text())
    records,report,runs=[],{},{}
    for split in refs:
        split_records=[]
        paired={}
        for task in ("tbc","nc"):
            model,selected,run=model_records(a.predictions/f"{task}_{split}",a.data,m,tokens,genes,refs[split],task,split)
            runs[f"{task}_{split}"]=run
            matching=[r for r in baseline_records(selected,baseline_predictions[split][:len(selected)],split,genes)
                      if r["task"]==task]
            split_records.extend(model+matching)
            paired[task]={}
            for method in sorted({r["method"] for r in matching}):
                by_row={r["row"]:r for r in matching if r["method"]==method and r["available"]}
                joint=[r for r in model if r["available"] and r["row"] in by_row]
                base=[by_row[r["row"]] for r in joint]
                paired[task][method]=dict(n_matched_available=len(joint),
                    model=grouped_report(joint,task),baseline=grouped_report(base,task))
        records.extend(split_records)
        report[split]=dict(methods=summarize(split_records),paired_comparisons=paired)
    a.output.mkdir(parents=True)
    dump(a.output/"comparison_per_row.json",records)
    result=dict(evaluation="Donor-held-out evaluation using exactly matched model and baseline queries",
                baseline_provenance=baseline_provenance,prediction_runs=runs,by_split=report)
    dump(a.output/"comparison_metrics.json",result)
    print(json.dumps({s:{t:{method:value["overall"] for method,value in methods.items()}
                         for t,methods in value["methods"].items()} for s,value in report.items()},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command,func in (("baselines",baselines),("compare",compare)):
        p=sub.add_parser(command)
        p.add_argument("--data",type=Path,required=True)
        p.add_argument("--output",type=Path,required=True)
        if command=="compare":
            p.add_argument("--baselines",type=Path,required=True)
            p.add_argument("--predictions",type=Path,required=True)
        p.set_defaults(func=func)
    a=parser.parse_args()
    a.func(a)


if __name__=="__main__":
    main()
