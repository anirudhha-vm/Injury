"""Uncertainty in predicted risk (contribution C3, non-negotiable N5).

Two sources are separated:

- **Growth-state uncertainty**: the model is evaluated once per draw of the growth posterior, so
  uncertainty about when an athlete's growth spurt is becomes uncertainty about their risk.
- **Model uncertainty**: spread across the members of a deep ensemble.

``RiskDistribution`` holds the resulting samples of risk per athlete-week and summarises them.
The reported risk is the mean over samples (the posterior predictive probability), not the risk
at the mean growth state: for a non-linear model these differ, which is the point of
propagating rather than plugging in.

Not implemented: split-conformal intervals (listed as optional in the brief). A conformal
interval for an event *probability* needs a calibration construction (e.g. Venn-Abers) that is a
design decision for the team, not a default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RiskDistribution:
    """Samples of predicted risk: (samples, rows)."""

    samples: FloatArray

    @property
    def mean(self) -> FloatArray:
        return np.asarray(self.samples.mean(axis=0))

    @property
    def sd(self) -> FloatArray:
        ddof = 1 if self.samples.shape[0] > 1 else 0
        return np.asarray(self.samples.std(axis=0, ddof=ddof))

    def interval(self, level: float = 0.90) -> tuple[FloatArray, FloatArray]:
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.samples, [tail, 1.0 - tail], axis=0)
        return np.asarray(low), np.asarray(high)


def propagate_growth_draws(
    predict: Callable[[FloatArray], FloatArray], growth_draws: FloatArray
) -> RiskDistribution:
    """Evaluate ``predict`` on every draw of the growth vector.

    ``growth_draws`` is (draws, rows, growth_dim). ``predict`` maps a (rows, growth_dim) array to
    risks of shape (rows,) or, for an ensemble, (members, rows); in the latter case the result
    combines growth and model uncertainty as draws x members samples.
    """
    samples = [np.atleast_2d(predict(draw)) for draw in growth_draws]
    return RiskDistribution(np.concatenate(samples, axis=0))


def decompose(
    predict_members: Callable[[FloatArray], FloatArray], growth_draws: FloatArray
) -> dict[str, FloatArray]:
    """Variance of risk split into a growth-state part and a model (ensemble) part.

    Law of total variance over ensemble members ``m`` and growth draws ``s``:
    ``total = Var_s(E_m[risk]) + E_s(Var_m[risk])``. Returns per-row ``growth``, ``model`` and
    ``total`` variances.
    """
    stacked = np.stack([np.atleast_2d(predict_members(draw)) for draw in growth_draws])
    # stacked: (draws, members, rows)
    growth = stacked.mean(axis=1).var(axis=0)
    model = stacked.var(axis=1).mean(axis=0)
    return {"growth": growth, "model": model, "total": growth + model}
