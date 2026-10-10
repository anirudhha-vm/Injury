"""Decision-curve analysis (Vickers & Elkin 2006).

If athletes whose predicted risk is at least the threshold probability ``t`` are selected for
review, the **net benefit** is

    true positives / n  -  false positives / n * t / (1 - t)

i.e. the benefit of correctly selected athletes minus the cost of unnecessary reviews, where
``t`` encodes how many unnecessary reviews one is willing to accept per correctly selected
athlete. A model is useful at ``t`` only if its net benefit exceeds both reference strategies:
review everyone ("treat all") and review no one ("treat none", net benefit zero).

This addresses the critique that discrimination alone does not show a screening test is worth
using (Bahr 2016).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def net_benefit(outcome: ArrayLike, risk: ArrayLike, threshold: float) -> float:
    """Net benefit of selecting everyone with risk >= ``threshold``."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    y = np.asarray(outcome, dtype=np.float64).ravel()
    selected = np.asarray(risk, dtype=np.float64).ravel() >= threshold
    true_positive = float(np.sum(selected & (y == 1))) / y.size
    false_positive = float(np.sum(selected & (y == 0))) / y.size
    return true_positive - false_positive * threshold / (1.0 - threshold)


def net_benefit_treat_all(outcome: ArrayLike, threshold: float) -> float:
    """Net benefit of selecting everyone: prevalence - (1 - prevalence) * t / (1 - t)."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    prevalence = float(np.mean(np.asarray(outcome, dtype=np.float64)))
    return prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)


def decision_curve(
    outcome: ArrayLike, risks: Mapping[str, ArrayLike], thresholds: ArrayLike
) -> pd.DataFrame:
    """Net benefit of each model and of the two reference strategies at each threshold.

    Returns a tidy table: ``threshold, strategy, net_benefit``.
    """
    rows: list[dict[str, float | str]] = []
    for threshold in np.asarray(thresholds, dtype=np.float64):
        t = float(threshold)
        rows.append(
            {
                "threshold": t,
                "strategy": "treat_all",
                "net_benefit": net_benefit_treat_all(outcome, t),
            }
        )
        rows.append({"threshold": t, "strategy": "treat_none", "net_benefit": 0.0})
        for name, risk in risks.items():
            rows.append(
                {"threshold": t, "strategy": name, "net_benefit": net_benefit(outcome, risk, t)}
            )
    return pd.DataFrame(rows)
