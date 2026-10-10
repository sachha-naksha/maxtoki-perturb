"""Baselines using only MaxToki's timed context and task-specific query inputs.

The trajectories use linear rank scores, not expression counts. No absolute
pseudotime or donor age is needed: the final context cell defines time zero.
These functions do no I/O. Run their tests and data evaluation under Slurm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def rank_features(
    tokens: Sequence[int], gene_ids: Iterable[int], max_genes: int = 2046,
) -> dict[int, float]:
    """Encode unique gene ranks with a fixed scale, ignoring special tokens.

    Rank r (one based) has score 1 - r / (max_genes + 1).
    A gene absent from the capped cell has score zero. The fixed denominator
    makes scores comparable across cells with different numbers of genes.
    """
    if not isinstance(max_genes, (int, np.integer)) or max_genes < 1:
        raise ValueError("max_genes must be a positive integer")
    genes = set(map(int, gene_ids))
    result = {}
    for token in tokens:
        token = int(token)
        if token in genes and token not in result:
            result[token] = 1.0 - (len(result) + 1) / (max_genes + 1)
            if len(result) == max_genes:
                break
    return result


def _context_inputs(context_tokens, context_times, gene_ids, max_genes):
    times = np.asarray(context_times, dtype=float)
    if (times.ndim != 1 or len(times) != len(context_tokens) or
            len(times) < 2 or not np.isfinite(times).all()):
        raise ValueError("At least two context cells with matching finite times required")
    genes = set(map(int, gene_ids))
    features = [rank_features(cell, genes, max_genes) for cell in context_tokens]
    return times - times[-1], features, genes


@dataclass
class _Trajectory:
    times: np.ndarray
    genes: tuple[int, ...]
    intercept: np.ndarray
    slope: np.ndarray
    available: bool
    reason: str | None
    diagnostics: dict


def _fit_context(context_tokens, context_times, gene_ids, max_genes):
    times, features, _ = _context_inputs(
        context_tokens, context_times, gene_ids, max_genes)
    genes = tuple(sorted(set().union(*(set(f) for f in features))))
    matrix = np.asarray([[f.get(g, 0.0) for g in genes] for f in features], dtype=float)
    centered_time = times - times.mean()
    time_ss = float(centered_time @ centered_time)
    diagnostics = dict(n_context=len(times), context_gene_count=len(genes),
                       relative_context_times=times.tolist(),
                       time_spread=float(np.ptp(times)))
    if not genes or any(not f for f in features):
        return _Trajectory(times, genes, np.zeros(len(genes)), np.zeros(len(genes)),
                           False, "empty_context_gene_profile", diagnostics)
    if time_ss <= 1e-12:
        return _Trajectory(times, genes, matrix.mean(axis=0), np.zeros(len(genes)),
                           False, "no_context_time_variation", diagnostics)
    slope = centered_time @ (matrix - matrix.mean(axis=0)) / time_ss
    intercept = matrix.mean(axis=0) - slope * times.mean()
    slope_ss = float(slope @ slope)
    diagnostics["slope_norm_squared"] = slope_ss
    if slope_ss <= 1e-18:
        return _Trajectory(times, genes, intercept, slope, False,
                           "no_context_gene_trend", diagnostics)
    return _Trajectory(times, genes, intercept, slope, True, None, diagnostics)


def context_linear_tbc(
    context_tokens, context_times, query_tokens, gene_ids, max_genes=2046,
) -> dict:
    """Project query rank expression onto the context's fitted time direction.

    Fits x(t) = intercept + slope * t to known context cells, with the final
    context at t=0. Returns the unconstrained least-squares query interval.
    Positive extrapolation is allowed. Query pseudotime is never an input.
    """
    genes = set(map(int, gene_ids))
    fit = _fit_context(context_tokens, context_times, genes, max_genes)
    result = dict(prediction=None, available=False, reason=fit.reason,
                  diagnostics=fit.diagnostics)
    if not fit.available:
        return result
    features = rank_features(query_tokens, genes, max_genes)
    if not features:
        result["reason"] = "empty_query_gene_profile"
        return result
    query = np.asarray([features.get(g, 0.0) for g in fit.genes])
    delta = float((query - fit.intercept) @ fit.slope / (fit.slope @ fit.slope))
    result.update(prediction=delta, available=True, reason=None)
    result["diagnostics"]["query_context_gene_overlap"] = len(set(features).intersection(fit.genes))
    result["diagnostics"]["extrapolated"] = bool(delta < fit.times.min() or delta > fit.times.max())
    return result


def context_linear_nc(
    context_tokens, context_times, query_delta, gene_ids,
    max_genes=2046, top_n=2046,
) -> dict:
    """Extrapolate a ranked gene profile to the supplied relative query time.

    Uses no query expression. Negative fitted rank scores are clipped to zero.
    Gene-ID order breaks score ties deterministically. Only genes observed in
    context can be generated by this baseline.
    """
    if not np.isfinite(query_delta):
        raise ValueError("query_delta must be finite")
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    fit = _fit_context(context_tokens, context_times, gene_ids, max_genes)
    result = dict(gene_tokens=[], available=False, reason=fit.reason,
                  diagnostics=fit.diagnostics)
    if not fit.available:
        return result
    scores = np.maximum(fit.intercept + fit.slope * float(query_delta), 0.0)
    ranked = sorted(((gene, float(score)) for gene, score in zip(fit.genes, scores)
                     if score > 0.0), key=lambda item: (-item[1], item[0]))
    if not ranked:
        result["reason"] = "no_positive_extrapolated_gene_scores"
        return result
    result.update(gene_tokens=[gene for gene, _ in ranked[:top_n]],
                  available=True, reason=None)
    result["diagnostics"]["extrapolated"] = bool(query_delta < fit.times.min() or
                                                query_delta > fit.times.max())
    result["diagnostics"]["positive_gene_count"] = len(ranked)
    return result


def nearest_expression_tbc(
    context_tokens, context_times, query_tokens, gene_ids, max_genes=2046,
) -> dict:
    """Use relative time of the context closest to query by rank-score cosine.

    Exact similarity ties average the corresponding known relative times.
    This secondary baseline interpolates within context and cannot extrapolate
    into a future that is later than every context cell.
    """
    times, contexts, genes = _context_inputs(
        context_tokens, context_times, gene_ids, max_genes)
    query = rank_features(query_tokens, genes, max_genes)
    result = dict(prediction=None, available=False, reason=None, diagnostics={})
    if not query or any(not context for context in contexts):
        result["reason"] = "empty_gene_profile"
        return result
    query_norm = np.sqrt(sum(value * value for value in query.values()))
    similarities = []
    for context in contexts:
        norm = np.sqrt(sum(value * value for value in context.values()))
        similarities.append(sum(value * query.get(gene, 0.0)
                                for gene, value in context.items()) / (norm * query_norm))
    similarities = np.asarray(similarities)
    best = float(similarities.max())
    if best <= 0.0:
        result["reason"] = "no_shared_query_context_genes"
        return result
    tied = np.flatnonzero(np.isclose(similarities, best, rtol=1e-12, atol=1e-12))
    result.update(prediction=float(times[tied].mean()), available=True,
                  diagnostics=dict(best_cosine=best, tied_context_indices=tied.tolist()))
    return result


def nearest_time_nc(
    context_tokens, context_times, query_delta, gene_ids,
    max_genes=2046, top_n=2046,
) -> dict:
    """Secondary interpolation check: use the context nearest supplied time.

    For future extrapolation this generally copies the last context and should
    not be treated as the primary aging baseline.
    """
    if not np.isfinite(query_delta):
        raise ValueError("query_delta must be finite")
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    times, contexts, _ = _context_inputs(
        context_tokens, context_times, gene_ids, max_genes)
    chosen = min(range(len(times)), key=lambda i: (abs(times[i] - query_delta), i))
    tokens = list(contexts[chosen])[:top_n]
    return dict(gene_tokens=tokens, available=bool(tokens),
                reason=None if tokens else "empty_context_gene_profile",
                diagnostics=dict(context_index=chosen, relative_context_time=float(times[chosen])))
