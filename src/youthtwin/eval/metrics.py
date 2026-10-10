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


def pr_auc(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Average precision: area under the precision-recall curve (useful for rare events).

    Sum over distinct thresholds of (recall step) x precision. Its chance level is the event
    prevalence, not 0.5. NaN if there are no events.
    """
    y, p = _binary(outcome, risk)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    true_positives = np.cumsum(y_sorted)[last_of_tie]
    predicted_positive = np.arange(1, y.size + 1)[last_of_tie]
    recall = true_positives / y.sum()
    precision = true_positives / predicted_positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def sensitivity_specificity(
    outcome: ArrayLike, risk: ArrayLike, threshold: float
) -> dict[str, float]:
    """Classification measures when everyone with risk >= ``threshold`` is flagged."""
    y, p = _binary(outcome, risk)
    flagged = p >= threshold
    tp, fp = float(np.sum(flagged & (y == 1))), float(np.sum(flagged & (y == 0)))
    fn, tn = float(np.sum(~flagged & (y == 1))), float(np.sum(~flagged & (y == 0)))

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "flagged_fraction": float(flagged.mean()),
    }


def time_dependent_auc(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Cumulative/dynamic AUC at the prediction horizon.

    On the athlete-week table every row is an athlete at risk at a prediction time, the outcome
    is "event within the horizon", and rows whose horizon is not fully observed have already
    been removed. Under those conditions the cumulative/dynamic time-dependent AUC at the
    horizon is the ordinary AUC of that binary outcome, which is what this returns. Rows from
    the same athlete are correlated: take confidence intervals from the athlete-level bootstrap.
    """
    return auc(outcome, risk)
