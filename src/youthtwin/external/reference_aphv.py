"""Reference age at peak height velocity from a child's full, dense growth record.

On real data the true age at PHV is unknown. The standard practical reference is the peak of a
curve fitted to a dense longitudinal record covering the growth spurt: here Preece-Baines
model 1 (non-linear least squares), cross-checked against the velocity peak of a smoothing
spline. Children whose record does not cover the spurt, whose fit is poor, or whose two
estimates disagree are excluded with a stated reason. The twin is then scored against this
reference using only a short window of the same child's data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares

from youthtwin.sim.growth_sim import pb1_height, pb1_peak_offset, pb1_velocity

FloatArray = NDArray[np.float64]
_THETA_STARTS_YEARS = (-2.0, -1.0, 0.0, 1.0, 2.0)  # around the initial guess
_GRID_STEP_YEARS = 0.01


@dataclass(frozen=True)
class PreeceBainesFit:
    h1: float
    h_theta: float
    s0: float
    s1: float
    theta: float
    aphv: float
    peak_velocity_cm_per_year: float
    rmse_cm: float


def fit_preece_baines(age: ArrayLike, stature: ArrayLike, initial_theta: float) -> PreeceBainesFit:
    """Least-squares Preece-Baines model 1 fit, best of several starting ages for theta."""
    t = np.asarray(age, dtype=np.float64)
    h = np.asarray(stature, dtype=np.float64)

    def residual(p: FloatArray) -> FloatArray:
        h1, h_theta, s0, s1, theta = (np.array([v]) for v in p)
        return pb1_height(t, h1, h_theta, s0, s1, theta) - h

    best = None
    for shift in _THETA_STARTS_YEARS:
        start = np.array([h.max() + 1.0, h.max() - 12.0, 0.1, 1.2, initial_theta + shift])
        lower = [h.max() - 5.0, 50.0, 0.0, 0.2, 6.0]
        upper = [h.max() + 30.0, h.max(), 1.0, 5.0, 20.0]
        start = np.clip(start, np.add(lower, 1e-6), np.subtract(upper, 1e-6))
        result = least_squares(residual, start, bounds=(lower, upper))
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    h1, h_theta, s0, s1, theta = (float(v) for v in best.x)
    rmse = float(np.sqrt(np.mean(best.fun**2)))
    try:
        aphv = theta + float(pb1_peak_offset(np.array([s0]), np.array([s1]))[0])
    except ValueError:  # the fitted curve has no pubertal spurt
        return PreeceBainesFit(h1, h_theta, s0, s1, theta, float("nan"), float("nan"), rmse)
    one = np.array([1.0])
    peak = float(
        pb1_velocity(np.array([aphv]), h1 * one, h_theta * one, s0 * one, s1 * one, theta * one)[0]
    )
    return PreeceBainesFit(h1, h_theta, s0, s1, theta, aphv, peak, rmse)


def spline_peak_age(
    age: ArrayLike, stature: ArrayLike, search: tuple[float, float] = (8.0, 18.0)
) -> float:
    """Age of the highest interior velocity peak of a GCV smoothing spline, within ``search``."""
    t = np.asarray(age, dtype=np.float64)
    h = np.asarray(stature, dtype=np.float64)
    order = np.argsort(t)
    spline = make_smoothing_spline(t[order], h[order])
    low, high = max(search[0], t.min()), min(search[1], t.max())
    grid = np.arange(low, high + _GRID_STEP_YEARS, _GRID_STEP_YEARS)
    velocity = spline.derivative()(grid)
    interior = np.zeros(grid.size, dtype=bool)
    interior[1:-1] = (velocity[1:-1] > velocity[:-2]) & (velocity[1:-1] >= velocity[2:])
    if not interior.any():
        return float("nan")
    return float(grid[np.argmax(np.where(interior, velocity, -np.inf))])


def reference_table(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """One row per child: reference APHV, fit quality, and whether the child is included.

    ``frame`` has columns ``athlete_id, sex, age, stature`` (complete records, any ages).
    """
    rows: list[dict[str, Any]] = []
    for athlete, record in frame.dropna(subset=["stature"]).groupby("athlete_id", sort=True):
        record = record[record["age"] >= float(config["min_age_years"])].sort_values("age")
        sex = str(record["sex"].iloc[0])
        fit = fit_preece_baines(
            record["age"], record["stature"], float(config["initial_theta_years"][sex])
        )
        spline = spline_peak_age(record["age"], record["stature"])
        reasons = []
        margin = float(config["coverage_years"])
        if not np.isfinite(fit.aphv):
            reasons.append("no pubertal peak in the fitted curve")
        elif record["age"].min() > fit.aphv - margin or record["age"].max() < fit.aphv + margin:
            reasons.append("coverage: record does not span the peak")
        if fit.rmse_cm > float(config["max_rmse_cm"]):
            reasons.append("poor Preece-Baines fit")
        if not np.isfinite(spline) or abs(spline - fit.aphv) > float(
            config["max_method_disagreement_years"]
        ):
            reasons.append("reference methods disagree")
        rows.append(
            {
                "athlete_id": str(athlete),
                "sex": sex,
                "reference_aphv": fit.aphv,
                "spline_aphv": spline,
                "peak_velocity_cm_per_year": fit.peak_velocity_cm_per_year,
                "rmse_cm": fit.rmse_cm,
                "n_measurements": len(record),
                "included": not reasons,
                "exclusion_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)
