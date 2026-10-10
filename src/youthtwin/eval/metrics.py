"""Performance metrics for predicted risk.

M6 needs only discrimination and overall accuracy to run the ablation grid; time-dependent AUC,
calibration slope and intercept, ICI/ECE and PR-AUC arrive with the evaluation suite in M7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _binary(outcome: ArrayLike, risk: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(outcome, dtype=np.float64).ravel()
    p = np.asarray(risk, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError("outcome and risk must have the same length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcome must be 0 or 1")
    return y, p


def auc(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Area under the ROC curve: P(a random event row scores above a random non-event row).

    Ties count one half. NaN if only one class is present.
    """
    y, p = _binary(outcome, risk)
    n_events, n_non_events = int(y.sum()), int((1 - y).sum())
    if n_events == 0 or n_non_events == 0:
        return float("nan")
    ranks = stats.rankdata(p)  # average ranks handle ties
    return float((ranks[y == 1].sum() - n_events * (n_events + 1) / 2) / (n_events * n_non_events))


def brier_score(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Mean squared difference between predicted risk and outcome (lower is better)."""
    y, p = _binary(outcome, risk)
    return float(np.mean((p - y) ** 2))
