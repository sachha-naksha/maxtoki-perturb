import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/torch_pipeline"))

import numpy as np
import pytest

from aging_baselines import (
    context_linear_nc, context_linear_tbc, nearest_expression_tbc,
    nearest_time_nc, rank_features,
)


def test_rank_features_use_fixed_scale_and_ignore_specials_and_duplicates():
    assert rank_features([99, 1, 1, 2, 3, 4, 5, 98], {1, 2, 3, 4, 5}, 4) == pytest.approx({
        1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2,
    })
    assert rank_features([1, 2], {1, 2}, 4) == pytest.approx({1: 0.8, 2: 0.6})


def test_tbc_exact_forward_projection_without_query_time():
    # Linear rank vectors: x0=(.8,.4,.6), x1=(.6,.4,.8).
    # Query=(.4,.6,.8); projection extends halfway past x1.
    result = context_linear_tbc([[1, 3, 2], [3, 1, 2]], [10, 20],
                                [3, 2, 1], {1, 2, 3}, max_genes=4)
    assert result["available"]
    assert result["prediction"] == pytest.approx(5.0)
    assert result["diagnostics"]["extrapolated"]
    assert "query_pseudotime" not in inspect.signature(context_linear_tbc).parameters


def test_baselines_are_invariant_to_absolute_time_offset():
    contexts = [[1, 3, 2], [3, 1, 2]]
    original = context_linear_tbc(contexts, [10, 20], [3, 2, 1], {1, 2, 3}, 4)
    shifted = context_linear_tbc(contexts, [1010, 1020], [3, 2, 1], {1, 2, 3}, 4)
    assert original == shifted
    assert context_linear_nc(contexts, [10, 20], 20, {1, 2, 3}, 4) == (
        context_linear_nc(contexts, [1010, 1020], 20, {1, 2, 3}, 4))


def test_nc_extrapolates_profile_using_only_time_and_context():
    result = context_linear_nc([[1, 3, 2], [3, 1, 2]], [10, 20], 20,
                               {1, 2, 3, 4}, max_genes=4)
    assert result["available"]
    assert result["gene_tokens"] == [3, 2, 1]
    assert result["diagnostics"]["extrapolated"]
    assert 4 not in result["gene_tokens"]


@pytest.mark.parametrize("times,cells,reason", [
    ([10, 10], [[1, 2], [2, 1]], "no_context_time_variation"),
    ([10, 20], [[1, 2], [1, 2]], "no_context_gene_trend"),
    ([10, 20], [[], [1, 2]], "empty_context_gene_profile"),
])
def test_degenerate_trajectories_are_unavailable_not_zero(times, cells, reason):
    tbc = context_linear_tbc(cells, times, [2, 1], {1, 2})
    nc = context_linear_nc(cells, times, 3, {1, 2})
    assert not tbc["available"] and tbc["prediction"] is None
    assert not nc["available"] and nc["gene_tokens"] == []
    assert tbc["reason"] == nc["reason"] == reason


def test_query_can_change_tbc_prediction_but_nc_requires_no_query_expression():
    contexts = [[1, 3, 2], [3, 1, 2]]
    for index, query in enumerate(contexts):
        result = context_linear_tbc(contexts, [10, 20], query, {1, 2, 3}, 4)
        assert result["prediction"] == pytest.approx([-10, 0][index])
    signature = inspect.signature(context_linear_nc)
    assert "query_tokens" not in signature.parameters


def test_nearest_expression_uses_known_relative_time_and_reports_no_overlap():
    contexts = [[1, 2], [3, 4]]
    result = nearest_expression_tbc(contexts, [50, 60], [1, 2], {1, 2, 3, 4}, 4)
    assert result["prediction"] == pytest.approx(-10)
    missing = nearest_expression_tbc(contexts, [50, 60], [5], {1, 2, 3, 4, 5}, 4)
    assert not missing["available"] and missing["prediction"] is None


def test_nearest_time_is_secondary_and_uses_supplied_relative_time():
    contexts = [[1, 2], [3, 4]]
    assert nearest_time_nc(contexts, [50, 60], -9, {1, 2, 3, 4})["gene_tokens"] == [1, 2]
    assert nearest_time_nc(contexts, [50, 60], 20, {1, 2, 3, 4})["gene_tokens"] == [3, 4]


def test_validation_and_deterministic_gene_tie_breaking():
    with pytest.raises(ValueError):
        context_linear_tbc([[1], [2]], [0, np.nan], [1], {1, 2})
    with pytest.raises(ValueError):
        context_linear_nc([[1], [2]], [0, 1], np.inf, {1, 2})
    with pytest.raises(ValueError):
        rank_features([1], {1}, max_genes=0)
    result = context_linear_nc([[2, 1], [1, 2]], [0, 2], -1, {1, 2}, 4)
    assert result["gene_tokens"] == [1, 2]
