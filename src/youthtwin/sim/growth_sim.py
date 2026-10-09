"""Synthetic growth: Preece-Baines model 1 height curves with an exactly known age at PHV.

Height:   h(t) = h1 - 2 (h1 - h_theta) / (exp(s0 (t - theta)) + exp(s1 (t - theta)))
Velocity: h'(t) = 2 (h1 - h_theta) (s0 e^{s0 u} + s1 e^{s1 u}) / (e^{s0 u} + e^{s1 u})^2
          with u = t - theta

The curve depends on ``t - theta`` only, and the location of the velocity peak relative to
``theta`` depends on ``s0`` and ``s1`` only. So a target true APHV is sampled first and ``theta``
is set to ``APHV - peak_offset(s0, s1)``; the true APHV of every synthetic athlete is then known
exactly, which is what the growth twin (M2) is scored against.

Sitting height is stature times a ratio that falls through childhood and rises slightly after
PHV (legs grow first, trunk later). Mass comes from a simple BMI trajectory.

All parameter values come from ``configs/sim/cohort.yaml`` and are ASSUMPTIONS for pipeline
development, not estimates. Outputs of this module are synthetic by definition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_PEAK_SEARCH_HALF_WIDTH_YEARS = 4.0
_PEAK_SEARCH_STEP_YEARS = 0.001
_DERIVATIVE_STEP_YEARS = 1e-4


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def pb1_height(
    age: FloatArray,
    h1: FloatArray,
    h_theta: FloatArray,
    s0: FloatArray,
    s1: FloatArray,
    theta: FloatArray,
) -> FloatArray:
    """Preece-Baines model 1 stature (cm)."""
    u = age - theta
    return np.asarray(
        h1 - 2.0 * (h1 - h_theta) / (np.exp(s0 * u) + np.exp(s1 * u)), dtype=np.float64
    )


def _pb1_velocity_shape(u: FloatArray, s0: FloatArray, s1: FloatArray) -> FloatArray:
    e0, e1 = np.exp(s0 * u), np.exp(s1 * u)
    return np.asarray((s0 * e0 + s1 * e1) / (e0 + e1) ** 2, dtype=np.float64)


def pb1_velocity(
    age: FloatArray,
    h1: FloatArray,
    h_theta: FloatArray,
    s0: FloatArray,
    s1: FloatArray,
    theta: FloatArray,
) -> FloatArray:
    """Preece-Baines model 1 height velocity (cm/year), analytic."""
    return 2.0 * (h1 - h_theta) * _pb1_velocity_shape(age - theta, s0, s1)


def pb1_peak_offset(s0: FloatArray, s1: FloatArray) -> FloatArray:
    """Age of peak height velocity minus ``theta`` (years) for each (s0, s1) pair."""
    s0 = np.atleast_1d(np.asarray(s0, dtype=np.float64))
    s1 = np.atleast_1d(np.asarray(s1, dtype=np.float64))
    grid = np.arange(
        -_PEAK_SEARCH_HALF_WIDTH_YEARS,
        _PEAK_SEARCH_HALF_WIDTH_YEARS + _PEAK_SEARCH_STEP_YEARS,
        _PEAK_SEARCH_STEP_YEARS,
    )
    shape = _pb1_velocity_shape(grid[None, :], s0[:, None], s1[:, None])
    k = np.argmax(shape, axis=1)
    if np.any(k == 0) or np.any(k == grid.size - 1):
        raise ValueError("Velocity peak outside the search window; check s0/s1")
    rows = np.arange(s0.size)
    left, mid, right = shape[rows, k - 1], shape[rows, k], shape[rows, k + 1]
    # Parabolic refinement of the grid maximum.
    shift = 0.5 * (left - right) / (left - 2.0 * mid + right)
    return np.asarray(grid[k] + shift * _PEAK_SEARCH_STEP_YEARS, dtype=np.float64)


@dataclass(frozen=True)
class GrowthParams:
    """True per-athlete growth parameters (arrays of equal length)."""

    is_female: BoolArray
    h1: FloatArray
    h_theta: FloatArray
    s0: FloatArray
    s1: FloatArray
    theta: FloatArray
    aphv: FloatArray
    sitting_ratio_adult: FloatArray
    bmi_offset: FloatArray

    def __len__(self) -> int:
        return int(self.aphv.size)

    def at(self, index: IntArray) -> GrowthParams:
        """Parameters of the athletes at ``index`` (repeats allowed), aligned with a long table."""
        return GrowthParams(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


def _by_sex(
    rng: np.random.Generator, is_female: BoolArray, spec: Mapping[str, Any], *, lognormal: bool
) -> FloatArray:
    out = np.empty(is_female.size, dtype=np.float64)
    for sex, mask in (("female", is_female), ("male", ~is_female)):
        n = int(mask.sum())
        if lognormal:
            out[mask] = spec[sex]["median"] * np.exp(rng.normal(0.0, spec[sex]["sigma_log"], n))
        else:
            out[mask] = rng.normal(spec[sex]["mean"], spec[sex]["sd"], n)
    return out


def sample_growth_params(
    rng: np.random.Generator, is_female: BoolArray, config: Mapping[str, Any]
) -> GrowthParams:
    """Draw true growth parameters; ``config`` is the ``growth`` block of the cohort config."""
    aphv = _by_sex(rng, is_female, config["aphv_years"], lognormal=False)
    h1 = _by_sex(rng, is_female, config["adult_height_cm"], lognormal=False)
    gain = np.abs(_by_sex(rng, is_female, config["height_gain_after_theta_cm"], lognormal=False))
    s0 = _by_sex(rng, is_female, config["s0"], lognormal=True)
    s1 = _by_sex(rng, is_female, config["s1"], lognormal=True)
    ratio = config["sitting_ratio"]["adult"]
    return GrowthParams(
        is_female=is_female,
        h1=h1,
        h_theta=h1 - gain,
        s0=s0,
        s1=s1,
        theta=aphv - pb1_peak_offset(s0, s1),
        aphv=aphv,
        sitting_ratio_adult=rng.normal(ratio["mean"], ratio["sd"], is_female.size),
        bmi_offset=rng.normal(0.0, config["bmi"]["between_athlete_sd"], is_female.size),
    )


class GrowthCurves:
    """True (error-free) growth trajectories. ``age`` arrays must align with ``params``."""

    def __init__(self, params: GrowthParams, config: Mapping[str, Any]) -> None:
        self.params = params
        self._config = config
        self._ratio = config["sitting_ratio"]
        self._bmi = config["bmi"]

    def at(self, index: IntArray) -> GrowthCurves:
        """Curves of the athletes at ``index`` (repeats allowed), aligned with a long table."""
        return GrowthCurves(self.params.at(index), self._config)

    def maturity_offset(self, age: FloatArray) -> FloatArray:
        """True years from PHV (negative before)."""
        return np.asarray(age - self.params.aphv, dtype=np.float64)

    def stature(self, age: FloatArray) -> FloatArray:
        p = self.params
        return pb1_height(age, p.h1, p.h_theta, p.s0, p.s1, p.theta)

    def height_velocity(self, age: FloatArray) -> FloatArray:
        p = self.params
        return pb1_velocity(age, p.h1, p.h_theta, p.s0, p.s1, p.theta)

    def sitting_ratio(self, age: FloatArray) -> FloatArray:
        r, u = self._ratio, self.maturity_offset(age)
        child = r["child_excess"] * _sigmoid(
            -(u - r["child_centre_years"]) / r["child_width_years"]
        )
        late = r["late_trunk_gain"] * _sigmoid((u - r["late_centre_years"]) / r["late_width_years"])
        return np.asarray(self.params.sitting_ratio_adult + child + late, dtype=np.float64)

    def sitting_height(self, age: FloatArray) -> FloatArray:
        return self.stature(age) * self.sitting_ratio(age)

    def leg_length(self, age: FloatArray) -> FloatArray:
        return self.stature(age) - self.sitting_height(age)

    def mass(self, age: FloatArray) -> FloatArray:
        b = self._bmi
        bmi = (
            b["at_age_8"]
            + b["slope_per_year"] * (age - 8.0)
            + b["maturity_gain"] * _sigmoid(self.maturity_offset(age))
            + self.params.bmi_offset
        )
        return np.asarray(bmi * (self.stature(age) / 100.0) ** 2, dtype=np.float64)

    def _derivative(self, curve: Callable[[FloatArray], FloatArray], age: FloatArray) -> FloatArray:
        step = _DERIVATIVE_STEP_YEARS
        return np.asarray((curve(age + step) - curve(age - step)) / (2.0 * step), dtype=np.float64)

    def leg_length_velocity(self, age: FloatArray) -> FloatArray:
        """cm/year."""
        return self._derivative(self.leg_length, age)

    def sitting_height_velocity(self, age: FloatArray) -> FloatArray:
        """cm/year."""
        return self._derivative(self.sitting_height, age)

    def mass_velocity(self, age: FloatArray) -> FloatArray:
        """kg/year."""
        return self._derivative(self.mass, age)


@dataclass(frozen=True)
class VisitSchedule:
    """Long-format visit table: one entry per attended visit."""

    athlete_index: IntArray
    day: IntArray  # days since study start


def sample_visits(
    rng: np.random.Generator, n_athletes: int, follow_up_days: int, config: Mapping[str, Any]
) -> VisitSchedule:
    """Sparse visits every 6-8 weeks with randomly missed visits (at least one kept each)."""
    low, high = config["interval_days"]
    athlete_index: list[int] = []
    days: list[int] = []
    for i in range(n_athletes):
        scheduled: list[int] = []
        day = int(rng.integers(0, config["first_visit_within_days"] + 1))
        while day < follow_up_days:
            scheduled.append(day)
            day += int(rng.integers(low, high + 1))
        attended = [d for d in scheduled if rng.random() >= config["missed_visit_probability"]]
        if not attended:
            attended = [scheduled[0]]
        athlete_index.extend([i] * len(attended))
        days.extend(attended)
    return VisitSchedule(
        np.asarray(athlete_index, dtype=np.int64), np.asarray(days, dtype=np.int64)
    )


def measure_twice(
    rng: np.random.Generator, true_value: FloatArray, error_sd: float, decimals: int = 1
) -> tuple[FloatArray, FloatArray]:
    """Two independent noisy measurements, rounded as they would be recorded."""
    first = np.round(true_value + rng.normal(0.0, error_sd, true_value.shape), decimals)
    second = np.round(true_value + rng.normal(0.0, error_sd, true_value.shape), decimals)
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
