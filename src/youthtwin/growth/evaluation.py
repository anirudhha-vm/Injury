"""Scoring the growth twin against a known age at PHV (simulation study, hypothesis H1).

Only meaningful where the true APHV is known: synthetic cohorts, or real athletes whose PHV was
observed during follow-up. Results from synthetic data carry ``synthetic=True`` and may be
reported only as a simulation study.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.growth.growth_state import SitarPosterior

FloatArray = NDArray[np.float64]

STRATA: tuple[str, ...] = ("pre_phv", "circa_phv", "post_phv")


def maturity_stratum(maturity_offset: FloatArray, band_years: float) -> NDArray[np.str_]:
    """Pre / circa / post-PHV by maturity offset, with ``circa`` = within ``band_years``."""
    return np.where(
        maturity_offset < -band_years,
        STRATA[0],
        np.where(maturity_offset > band_years, STRATA[2], STRATA[1]),
    ).astype(np.str_)


def score_aphv(
    posterior: SitarPosterior,
    true_aphv: Mapping[str, float],
    point_estimate: Mapping[str, float],
    *,
    level: float,
) -> pd.DataFrame:
    """Per-athlete comparison of posterior APHV and a point estimate against the truth."""
    draws = posterior.aphv()
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail], axis=0)
    truth = np.asarray([true_aphv[a] for a in posterior.athlete_ids], dtype=np.float64)
    point = np.asarray([point_estimate[a] for a in posterior.athlete_ids], dtype=np.float64)
    mean = draws.mean(axis=0)
    return pd.DataFrame(
        {
            "athlete_id": list(posterior.athlete_ids),
            "sex": posterior.sex,
            "true_aphv": truth,
            "posterior_mean": mean,
            "posterior_sd": draws.std(axis=0, ddof=1),
            "interval_low": low,
            "interval_high": high,
            "covered": (truth >= low) & (truth <= high),
            "posterior_abs_error": np.abs(mean - truth),
            "point_estimate": point,
            "point_abs_error": np.abs(point - truth),
            "synthetic": posterior.synthetic,
        }
    )


def summarise_scores(
    scores: pd.DataFrame, *, level: float, coverage_range: Sequence[float], by: str | None = None
) -> pd.DataFrame:
    """Coverage and mean absolute errors, overall or per group of column ``by``."""

    def one(group: pd.DataFrame) -> dict[str, Any]:
        coverage = float(group["covered"].mean())
        posterior_mae = float(group["posterior_abs_error"].mean())
        point_mae = float(group["point_abs_error"].mean())
        return {
            "n_athletes": len(group),
            "interval_level": level,
            "coverage": coverage,
            "coverage_in_range": bool(coverage_range[0] <= coverage <= coverage_range[1]),
            "posterior_mae_years": posterior_mae,
            "point_mae_years": point_mae,
            "posterior_beats_point": bool(posterior_mae < point_mae),
            "mean_interval_width_years": float(
                (group["interval_high"] - group["interval_low"]).mean()
            ),
            "posterior_bias_years": float((group["posterior_mean"] - group["true_aphv"]).mean()),
            "point_bias_years": float((group["point_estimate"] - group["true_aphv"]).mean()),
        }

    if by is None:
        return pd.DataFrame([{"group": "all", **one(scores)}])
    rows = [{"group": str(key), **one(group)} for key, group in scores.groupby(by)]
    return pd.DataFrame(rows)
