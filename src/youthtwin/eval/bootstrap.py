"""Athlete-level (cluster) bootstrap.

Athlete-weeks from the same athlete are correlated, and overlapping outcome windows make
consecutive weeks nearly duplicates. Resampling rows would treat them as independent and give
intervals that are far too narrow. Here whole athletes are resampled with replacement; an
athlete drawn twice contributes all their rows twice.

Used for confidence intervals of any metric and, as the primary comparison method, for the
difference in a metric between two models evaluated on the same athletes (paired: both models
see the same resampled athletes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Metric = Callable[[FloatArray, FloatArray], float]


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float  # the metric on the original data
    ci_low: float
    ci_high: float
    n_resamples: int  # resamples that produced a finite value
    level: float


def _row_groups(groups: ArrayLike) -> list[NDArray[np.int64]]:
    labels = np.asarray(groups)
    _, inverse = np.unique(labels, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return [np.asarray(part, dtype=np.int64) for part in np.split(order, boundaries)]


def _resampled_rows(
    clusters: list[NDArray[np.int64]], rng: np.random.Generator
) -> NDArray[np.int64]:
    chosen = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[k] for k in chosen])


def _interval(values: list[float], estimate: float, level: float) -> BootstrapResult:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"), 0, level)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return BootstrapResult(estimate, float(low), float(high), int(finite.size), level)


def cluster_bootstrap(
    metric: Metric,
    outcome: ArrayLike,
    risk: ArrayLike,
    groups: ArrayLike,
    *,
    n_resamples: int,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval for ``metric(outcome, risk)``, resampling whole athletes."""
    y = np.asarray(outcome, dtype=np.float64)
    p = np.asarray(risk, dtype=np.float64)
    clusters = _row_groups(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        rows = _resampled_rows(clusters, rng)
        values.append(metric(y[rows], p[rows]))
    return _interval(values, metric(y, p), level)


def cluster_bootstrap_difference(
    metric: Metric,
    outcome: ArrayLike,
    risk_a: ArrayLike,
    risk_b: ArrayLike,
    groups: ArrayLike,
    *,
    n_resamples: int,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval for ``metric(model A) - metric(model B)`` on the same athletes."""
    y = np.asarray(outcome, dtype=np.float64)
    a = np.asarray(risk_a, dtype=np.float64)
    b = np.asarray(risk_b, dtype=np.float64)
    clusters = _row_groups(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        rows = _resampled_rows(clusters, rng)
        values.append(metric(y[rows], a[rows]) - metric(y[rows], b[rows]))
    return _interval(values, metric(y, a) - metric(y, b), level)
