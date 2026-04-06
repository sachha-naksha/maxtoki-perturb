"""Calibrated perturbation scoring.

Uses a Ridge regression from pretraining embeddings → chronological age
(fitted via LODO cross-validation on CELLxGENE cardiomyocytes) to convert
perturbation embedding shifts into estimated years of aging change.

The direction vector w = Ridge.coef_ defines the "aging axis" in embedding
space. For a perturbation:
    delta_years = w · (perturbed_embedding - baseline_embedding)

This measures how far the cell moved along the aging axis, in year-equivalent
units. Positive = aged, negative = rejuvenated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .inference import predict_cell_embedding
from .perturbation import perturb_tokens

Direction = Literal["delete", "overexpress", "inhibit"]


class AgingCalibrator:
    """Loads a fitted Ridge regressor and scores perturbations in year-equivalents."""

    def __init__(self, regressor_path: str | Path) -> None:
        data = np.load(regressor_path)
        self.coef: np.ndarray = data["coef"]  # (hidden_size,)
        self.intercept: float = float(data["intercept"])
        self.alpha: float = float(data["alpha"][0]) if "alpha" in data else 1000.0

    def predict_age(self, embedding: np.ndarray) -> float:
        return float(np.dot(self.coef, embedding) + self.intercept)

    def delta_years(
        self, baseline_emb: np.ndarray, perturbed_emb: np.ndarray
    ) -> float:
        return float(np.dot(self.coef, perturbed_emb - baseline_emb))


def calibrated_perturb(
    model,
    calibrator: AgingCalibrator,
    token_ids: list[int],
    gene_token: int,
    direction: Direction,
    bos_id: int,
    eos_id: int,
) -> dict:
    """Run a perturbation and score it in calibrated year-equivalents.

    Returns dict with:
        delta_years: float - estimated aging shift (positive = aged)
        baseline_age_pred: float - predicted baseline age
        perturbed_age_pred: float - predicted perturbed age
        gene_present: bool
        direction: str
    """
    baseline_emb = predict_cell_embedding(model, token_ids, layer="last_hidden")
    perturbed_tokens = perturb_tokens(
        token_ids, gene_token, direction, bos_id, eos_id
    )
    perturbed_emb = predict_cell_embedding(
        model, perturbed_tokens, layer="last_hidden"
    )

    baseline_age = calibrator.predict_age(baseline_emb)
    perturbed_age = calibrator.predict_age(perturbed_emb)
    delta = calibrator.delta_years(baseline_emb, perturbed_emb)

    return {
        "delta_years": delta,
        "baseline_age_pred": baseline_age,
        "perturbed_age_pred": perturbed_age,
        "gene_present": gene_token in token_ids,
        "direction": direction,
    }
