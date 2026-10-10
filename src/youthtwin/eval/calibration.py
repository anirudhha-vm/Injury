"""Calibration of predicted risk.

- **Calibration intercept** (calibration-in-the-large): ``a`` in ``logit(P(y)) = a + logit(p)``.
  Zero when risks are right on average; negative when risks are too high.
- **Calibration slope**: ``b`` in ``logit(P(y)) = a + b * logit(p)``. One is ideal; below one
  means predictions are too extreme (the usual sign of overfitting), above one too timid.
- **ICI** (integrated calibration index, Austin & Steyerberg 2019): mean absolute difference
  between predicted risk and a loess-smoothed estimate of the observed risk.
- **ECE**: the same idea with equal-size bins instead of a smoother.

The logistic recalibration models are fitted by Newton's method.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.nonparametric.smoothers_lowess import lowess

FloatArray = NDArray[np.float64]
_CLIP = 1e-6
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-10


def _prepare(outcome: ArrayLike, risk: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(outcome, dtype=np.float64).ravel()
    p = np.asarray(risk, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError("outcome and risk must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration needs both events and non-events")
    clipped = np.clip(p, _CLIP, 1.0 - _CLIP)
    return y, p, np.log(clipped / (1.0 - clipped))


def _logistic_fit(design: FloatArray, y: FloatArray, offset: FloatArray) -> FloatArray:
    """Maximum-likelihood coefficients of ``logit(P(y)) = offset + design @ beta``."""
    beta = np.zeros(design.shape[1])
    for _ in range(_MAX_ITERATIONS):
        mean = 1.0 / (1.0 + np.exp(-(offset + design @ beta)))
        gradient = design.T @ (y - mean)
        hessian = (design * (mean * (1.0 - mean))[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < _TOLERANCE:
            break
    return beta


def calibration_intercept(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Calibration-in-the-large: intercept with the slope fixed at one."""
    y, _, logit = _prepare(outcome, risk)
    return float(_logistic_fit(np.ones((y.size, 1)), y, logit)[0])


def calibration_slope(outcome: ArrayLike, risk: ArrayLike) -> float:
    """Slope of the outcome on the logit of predicted risk (intercept free)."""
    y, _, logit = _prepare(outcome, risk)
    design = np.column_stack([np.ones(y.size), logit])
    return float(_logistic_fit(design, y, np.zeros(y.size))[1])


def smoothed_observed_risk(
    outcome: ArrayLike, risk: ArrayLike, *, frac: float = 0.75
) -> FloatArray:
    """Loess estimate of the observed event rate at each predicted risk (same order as input)."""
    y, p, _ = _prepare(outcome, risk)
    delta = 0.01 * float(np.ptp(p))  # skip refits for nearly equal risks: large speed-up
    smooth = lowess(y, p, frac=frac, it=0, delta=delta, return_sorted=False)
    return np.clip(np.asarray(smooth, dtype=np.float64), 0.0, 1.0)


def integrated_calibration_index(
    outcome: ArrayLike, risk: ArrayLike, *, frac: float = 0.75
) -> float:
    """ICI: mean |predicted risk - smoothed observed risk|."""
    _, p, _ = _prepare(outcome, risk)
    return float(np.mean(np.abs(p - smoothed_observed_risk(outcome, risk, frac=frac))))


def calibration_bins(
    outcome: ArrayLike, risk: ArrayLike, n_bins: int = 10
) -> dict[str, FloatArray]:
    """Mean predicted risk, observed rate and size of equal-size bins of predicted risk."""
    y, p, _ = _prepare(outcome, risk)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, min(n_bins, y.size))
    return {
        "predicted": np.array([p[g].mean() for g in groups]),
        "observed": np.array([y[g].mean() for g in groups]),
        "n": np.array([g.size for g in groups], dtype=np.float64),
    }


def expected_calibration_error(outcome: ArrayLike, risk: ArrayLike, n_bins: int = 10) -> float:
    """ECE: size-weighted mean |predicted - observed| over equal-size bins."""
    bins = calibration_bins(outcome, risk, n_bins)
    gap = np.abs(bins["predicted"] - bins["observed"])
    return float(np.sum(gap * bins["n"]) / np.sum(bins["n"]))
