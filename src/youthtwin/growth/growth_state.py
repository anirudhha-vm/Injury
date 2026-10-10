"""Growth twin state: what the fitted growth model says about one athlete at one age.

``SitarPosterior`` holds posterior draws of the SITAR-style model

    stature = intercept + alpha_i + h((age - beta_i) * exp(gamma_i))

and is plain NumPy, so everything downstream (risk models, dashboard) can use a saved posterior
without importing the sampler.

The full model adds a per-athlete velocity deviation ``delta_i * (age - mean_age_i)``; a constant
added to velocity does not move the age at which velocity peaks.

**Age at PHV is defined as the age of maximum velocity of the athlete's fitted curve**, computed
per posterior draw: ``aphv_i = beta_i + z_peak * exp(-gamma_i)``, where ``z_peak`` maximises
``h'(z)`` in that draw. ``beta_i`` alone is not APHV, because nothing forces the spline's
velocity peak to sit at ``z = 0``.

Every quantity is returned as posterior draws plus mean and SD, so uncertainty can be propagated
to the risk model (non-negotiable N5). M2 covers stature; leg-length and mass velocities are
added when those outcomes are modelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from youthtwin.growth import splines

FloatArray = NDArray[np.float64]

DAYS_PER_YEAR = 365.25
_MID_MONTH_DAY = 15
_PEAK_GRID_STEP_YEARS = 0.01
_ARRAY_FIELDS = (
    "intercept",
    "linear",
    "coefs",
    "alpha",
    "beta",
    "gamma",
    "sigma",
    "sigma_alpha",
    "sigma_gamma",
    "delta",
    "sigma_delta",
    "timing_offset",
    "sigma_beta",
    "cohort_mean",
    "cohort_sd",
)


def age_on(dob_month: str, when: date) -> float:
    """Decimal age on ``when`` for a month-precision date of birth (taken as mid-month)."""
    year, month = (int(part) for part in dob_month.split("-"))
    return (when - date(year, month, _MID_MONTH_DAY)).days / DAYS_PER_YEAR


@dataclass(frozen=True)
class Summary:
    """Posterior draws of one quantity with their mean and SD."""

    draws: FloatArray

    @property
    def mean(self) -> float:
        return float(np.mean(self.draws))

    @property
    def sd(self) -> float:
        return float(np.std(self.draws, ddof=1)) if self.draws.size > 1 else 0.0

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        """Central posterior interval."""
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.draws, [tail, 1.0 - tail])
        return float(low), float(high)


@dataclass(frozen=True)
class GrowthState:
    """Growth state of one athlete at one age (stature model)."""

    athlete_id: str
    age: float
    aphv: Summary  # years
    maturity_offset: Summary  # years; negative before PHV
    height_velocity: Summary  # cm/year
    pct_peak_velocity: Summary  # current velocity as % of this athlete's peak velocity
    stature: Summary  # cm, fitted
    forecast_stature_gain: Summary  # cm over ``forecast_horizon_days``
    forecast_horizon_days: int
    synthetic: bool

    def as_features(self) -> dict[str, float]:
        """Posterior means and SDs as a flat feature dictionary."""
        features: dict[str, float] = {}
        for name in (
            "aphv",
            "maturity_offset",
            "height_velocity",
            "pct_peak_velocity",
            "forecast_stature_gain",
        ):
            summary: Summary = getattr(self, name)
            features[f"{name}_mean"] = summary.mean
            features[f"{name}_sd"] = summary.sd
        return features


@dataclass(frozen=True)
class SitarPosterior:
    """Posterior draws (S draws, n athletes, K knots)."""

    athlete_ids: tuple[str, ...]
    knots: tuple[float, ...]
    intercept: FloatArray  # (S,)
    linear: FloatArray  # (S,)
    coefs: FloatArray  # (S, K - 2)
    alpha: FloatArray  # (S, n)
    beta: FloatArray  # (S, n)
    gamma: FloatArray  # (S, n)
    sigma: FloatArray  # (S,)
    sigma_alpha: FloatArray  # (S,) SD of the size effect
    sigma_gamma: FloatArray  # (S,) SD of the intensity effect (zeros if disabled)
    delta: FloatArray  # (S, n) velocity deviation, cm/year (zeros if disabled)
    sigma_delta: FloatArray  # (S,) SD of the velocity deviation
    timing_offset: FloatArray  # (S,) cohort-level shift of the timing prior centre, years
    sigma_beta: FloatArray  # (S,) SD of the timing prior, years
    cohort_mean: FloatArray  # (S,) cohort mean timing, years (NaN if the cohort prior is off)
    cohort_sd: FloatArray  # (S,) cohort SD of timing, years (NaN if off)
    mean_age: FloatArray  # (n,) each athlete's mean age at measurement (pivot for delta)
    sex: str
    synthetic: bool

    def __post_init__(self) -> None:
        n_draws, n_athletes = self.alpha.shape
        if len(self.athlete_ids) != n_athletes:
            raise ValueError("athlete_ids does not match the athlete dimension")
        if self.coefs.shape != (n_draws, splines.n_basis(self.knots)):
            raise ValueError("coefs does not match draws x (knots - 2)")
        if len(set(self.athlete_ids)) != n_athletes:
            raise ValueError("athlete_ids must be unique")

    @property
    def n_draws(self) -> int:
        return int(self.alpha.shape[0])

    def index_of(self, athlete_id: str) -> int:
        try:
            return self.athlete_ids.index(athlete_id)
        except ValueError:
            raise KeyError(f"Athlete {athlete_id} is not in this posterior") from None

    # --- mean curve -------------------------------------------------------------------

    def _coef_columns(self) -> list[FloatArray]:
        return [self.coefs[:, j : j + 1] for j in range(self.coefs.shape[1])]

    def mean_curve_velocity(self, z: FloatArray) -> FloatArray:
        """``h'(z)`` for every draw: shape (S, len(z))."""
        return np.asarray(
            splines.spline_derivative(
                z[None, :], self.linear[:, None], self._coef_columns(), self.knots
            ),
            dtype=np.float64,
        )

    def z_peak(self) -> tuple[FloatArray, FloatArray]:
        """Location and height of the velocity peak of the mean curve, per draw.

        Searched between the boundary knots for the highest interior local maximum. A draw
        without one has no growth spurt in range; its location is then the global maximum,
        normally a boundary (and should be rare in a healthy fit).
        """
        grid = np.arange(
            self.knots[0], self.knots[-1] + _PEAK_GRID_STEP_YEARS, _PEAK_GRID_STEP_YEARS
        )
        velocity = self.mean_curve_velocity(grid)
        # The growth spurt is the highest *interior local* maximum of velocity. Childhood
        # velocity at the young edge of the knot range can exceed the pubertal peak, so the
        # global maximum would sometimes be the edge. Draws with no interior local maximum fall
        # back to the global maximum and are reported by ``has_interior_peak``.
        local = np.zeros_like(velocity, dtype=np.bool_)
        local[:, 1:-1] = (velocity[:, 1:-1] > velocity[:, :-2]) & (
            velocity[:, 1:-1] >= velocity[:, 2:]
        )
        k = np.where(
            local.any(axis=1),
            np.argmax(np.where(local, velocity, -np.inf), axis=1),
            np.argmax(velocity, axis=1),
        )
        rows = np.arange(self.n_draws)
        interior = (k > 0) & (k < grid.size - 1)
        ki = np.clip(k, 1, grid.size - 2)
        left, mid, right = velocity[rows, ki - 1], velocity[rows, ki], velocity[rows, ki + 1]
        curvature = left - 2.0 * mid + right
        safe = interior & (curvature < 0)
        shift = np.where(safe, 0.5 * (left - right) / np.where(safe, curvature, 1.0), 0.0)
        location = grid[k] + shift * _PEAK_GRID_STEP_YEARS
        return np.asarray(location, dtype=np.float64), np.asarray(velocity[rows, k], np.float64)

    def has_interior_peak(self) -> NDArray[np.bool_]:
        """Per draw: is the velocity maximum strictly inside the knot range?"""
        location, _ = self.z_peak()
        return np.asarray(
            (location > self.knots[0] + _PEAK_GRID_STEP_YEARS)
            & (location < self.knots[-1] - _PEAK_GRID_STEP_YEARS)
        )

    # --- per athlete ------------------------------------------------------------------

    def aphv(self) -> FloatArray:
        """Age at peak height velocity for every draw and athlete: shape (S, n)."""
        location, _ = self.z_peak()
        return np.asarray(self.beta + location[:, None] * np.exp(-self.gamma), dtype=np.float64)

    def _z(self, index: int, age: float) -> FloatArray:
        return np.asarray((age - self.beta[:, index]) * np.exp(self.gamma[:, index]))

    def stature(self, athlete_id: str, age: float) -> FloatArray:
        """Fitted stature (cm) at ``age``, per draw."""
        i = self.index_of(athlete_id)
        z = self._z(i, age)
        curve = splines.spline_value(z, self.linear, self.coefs.T, self.knots)
        deviation = self.delta[:, i] * (age - self.mean_age[i])
        return np.asarray(self.intercept + self.alpha[:, i] + curve + deviation, dtype=np.float64)

    def height_velocity(self, athlete_id: str, age: float) -> FloatArray:
        """Height velocity (cm/year) at ``age``, per draw."""
        i = self.index_of(athlete_id)
        slope = splines.spline_derivative(self._z(i, age), self.linear, self.coefs.T, self.knots)
        return np.asarray(slope * np.exp(self.gamma[:, i]) + self.delta[:, i], dtype=np.float64)

    def growth_state(
        self, athlete_id: str, age: float, forecast_horizon_days: int = 90
    ) -> GrowthState:
        """Growth state of ``athlete_id`` at ``age`` with a short-term stature forecast."""
        i = self.index_of(athlete_id)
        aphv = self.aphv()[:, i]
        _, peak_slope = self.z_peak()
        velocity = self.height_velocity(athlete_id, age)
        peak_velocity = peak_slope * np.exp(self.gamma[:, i]) + self.delta[:, i]
        now = self.stature(athlete_id, age)
        later = self.stature(athlete_id, age + forecast_horizon_days / DAYS_PER_YEAR)
        return GrowthState(
            athlete_id=athlete_id,
            age=age,
            aphv=Summary(aphv),
            maturity_offset=Summary(age - aphv),
            height_velocity=Summary(velocity),
            pct_peak_velocity=Summary(100.0 * velocity / peak_velocity),
            stature=Summary(now),
            forecast_stature_gain=Summary(later - now),
            forecast_horizon_days=forecast_horizon_days,
            synthetic=self.synthetic,
        )

    def features_at(self, athlete_id: str, ages: FloatArray) -> dict[str, FloatArray]:
        """Posterior mean and SD of the growth state at many ages for one athlete.

        Vectorised counterpart of ``growth_state(...).as_features()`` for building athlete-week
        tables. Keys: ``aphv, maturity_offset, height_velocity, pct_peak_velocity``, each with
        ``_mean`` and ``_sd``; arrays align with ``ages``.
        """
        i = self.index_of(athlete_id)
        ages = np.asarray(ages, dtype=np.float64)
        location, peak_slope = self.z_peak()
        stretch = np.exp(self.gamma[:, i])
        aphv = self.beta[:, i] + location / stretch  # (S,)
        z = (ages[None, :] - self.beta[:, i : i + 1]) * stretch[:, None]  # (S, A)
        slope = splines.spline_derivative(z, self.linear[:, None], self._coef_columns(), self.knots)
        velocity = slope * stretch[:, None] + self.delta[:, i : i + 1]
        peak = (peak_slope * stretch + self.delta[:, i])[:, None]
        quantities = {
            "aphv": np.broadcast_to(aphv[:, None], z.shape),
            "maturity_offset": ages[None, :] - aphv[:, None],
            "height_velocity": velocity,
            "pct_peak_velocity": 100.0 * velocity / peak,
        }
        ddof = 1 if self.n_draws > 1 else 0
        features: dict[str, FloatArray] = {}
        for name, draws in quantities.items():
            features[f"{name}_mean"] = np.asarray(draws.mean(axis=0), dtype=np.float64)
            features[f"{name}_sd"] = np.asarray(draws.std(axis=0, ddof=ddof), dtype=np.float64)
        return features

    # --- persistence ------------------------------------------------------------------

    def thin(self, n_draws: int, seed: int = 0) -> SitarPosterior:
        """A random subset of ``n_draws`` draws (all draws if fewer are available)."""
        if n_draws >= self.n_draws:
            return self
        keep = np.sort(np.random.default_rng(seed).choice(self.n_draws, n_draws, replace=False))
        arrays = {name: getattr(self, name)[keep] for name in _ARRAY_FIELDS}
        return SitarPosterior(
            athlete_ids=self.athlete_ids,
            knots=self.knots,
            mean_age=self.mean_age,
            sex=self.sex,
            synthetic=self.synthetic,
            **arrays,
        )

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        """Write ``<path>.npz`` (draws) and ``<path>.json`` (ids, knots, flags, metadata)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"), **{name: getattr(self, name) for name in _ARRAY_FIELDS}
        )
        info = {
            "athlete_ids": list(self.athlete_ids),
            "knots": list(self.knots),
            "mean_age": self.mean_age.tolist(),
            "sex": self.sex,
            "synthetic": self.synthetic,
            "metadata": metadata or {},
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path) -> SitarPosterior:
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz")) as stored:
            arrays = {name: np.asarray(stored[name], dtype=np.float64) for name in _ARRAY_FIELDS}
        return cls(
            athlete_ids=tuple(info["athlete_ids"]),
            knots=tuple(info["knots"]),
            mean_age=np.asarray(info["mean_age"], dtype=np.float64),
            sex=info["sex"],
            synthetic=bool(info["synthetic"]),
            **arrays,
        )
