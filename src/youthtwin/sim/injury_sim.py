"""Synthetic weekly exposure and cause-specific injuries from a KNOWN process.

Weekly hazards (logit scale), coefficients from ``configs/sim/cohort.yaml``:

- growth-related:  intercept + leg-length velocity + load proxy + velocity x load + hours
- soft-tissue:     intercept + post-PHV + exposure spike + hours
                   (post-PHV = sigmoid of 2 x true maturity offset)
- other non-contact: intercept only

An injury removes the athlete from the risk set for the following ``ceil(time_loss_days / 7)``
weeks, during which no injury can occur and exposure is zero. At most one injury per athlete
per week; if several are drawn, one is kept with probability proportional to its hazard.

Because the process is known, later milestones can check that the modelling pipeline recovers
the effect directions. These are simulated labels for method testing only; they are never used
as, or mixed with, real outcomes (non-negotiable N1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

OUTCOME_GROUPS: tuple[str, ...] = (
    "growth_related",
    "non_contact_soft_tissue",
    "other_non_contact",
)


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@dataclass(frozen=True)
class Exposure:
    """Planned weekly exposure, shape (n_athletes, n_weeks)."""

    hours_training: FloatArray
    hours_match: FloatArray
    spike: BoolArray


def sample_exposure(
    rng: np.random.Generator, age: FloatArray, config: Mapping[str, Any]
) -> Exposure:
    """Weekly hours rising with age, gamma-distributed, with occasional spikes."""
    t = config["training_hours"]
    mean = t["at_age_8"] + t["slope_per_year"] * (age - 8.0)
    shape = t["gamma_shape"]
    hours = rng.gamma(shape, mean / shape)
    spike = rng.random(age.shape) < config["spike_probability"]
    hours = np.where(spike, hours * config["spike_multiplier"], hours)
    match = hours * config["match_hours_fraction"] * rng.uniform(0.5, 1.5, age.shape)
    return Exposure(np.asarray(hours, dtype=np.float64), np.asarray(match, dtype=np.float64), spike)


def weekly_hazards(
    *,
    leg_length_velocity: FloatArray,
    load_proxy: FloatArray,
    maturity_offset: FloatArray,
    hours: FloatArray,
    spike: BoolArray,
    config: Mapping[str, Any],
) -> dict[str, FloatArray]:
    """True weekly hazard per outcome group (probability of onset in an at-risk week)."""
    std = config["standardisation"]
    v_spec, h_spec = std["leg_length_velocity_cm_per_year"], std["weekly_hours"]
    velocity_z = (leg_length_velocity - v_spec["centre"]) / v_spec["scale"]
    hours_z = (hours - h_spec["centre"]) / h_spec["scale"]
    g, s, o = (config[name] for name in OUTCOME_GROUPS)
    growth = (
        g["intercept"]
        + g["leg_length_velocity"] * velocity_z
        + g["load_proxy"] * load_proxy
        + g["velocity_x_load"] * velocity_z * load_proxy
        + g["hours"] * hours_z
    )
    soft = (
        s["intercept"]
        + s["post_phv"] * _sigmoid(2.0 * maturity_offset)
        + s["exposure_spike"] * spike.astype(np.float64)
        + s["hours"] * hours_z
    )
    other = np.full_like(growth, o["intercept"])
    return dict(
        zip(OUTCOME_GROUPS, (_sigmoid(growth), _sigmoid(soft), _sigmoid(other)), strict=True)
    )


@dataclass(frozen=True)
class InjuryEvents:
    """Simulated injuries (long format) plus the risk-set indicator."""

    athlete_index: IntArray
    week: IntArray
    outcome_group: NDArray[np.str_]
    time_loss_days: IntArray
    onset_day_in_week: IntArray
    at_risk: BoolArray  # (n_athletes, n_weeks)


def simulate_injuries(
    rng: np.random.Generator, hazards: Mapping[str, FloatArray], config: Mapping[str, Any]
) -> InjuryEvents:
    """Draw injuries week by week, respecting time-loss exclusion from the risk set."""
    stacked = np.stack([hazards[name] for name in OUTCOME_GROUPS], axis=-1)  # (n, weeks, k)
    n_athletes, n_weeks, n_groups = stacked.shape
    loss = config["time_loss_days"]
    out_until = np.zeros(n_athletes, dtype=np.int64)  # first week back in the risk set
    at_risk = np.zeros((n_athletes, n_weeks), dtype=np.bool_)
    rows: list[tuple[int, int, int, int, int]] = []
    for week in range(n_weeks):
        at_risk[:, week] = week >= out_until
        drawn = (rng.random((n_athletes, n_groups)) < stacked[:, week, :]) & at_risk[:, week, None]
        for athlete in np.flatnonzero(drawn.any(axis=1)):
            candidates = np.flatnonzero(drawn[athlete])
            weights = stacked[athlete, week, candidates]
            group = int(rng.choice(candidates, p=weights / weights.sum()))
            days = max(1, round(float(loss["median"] * np.exp(rng.normal(0.0, loss["sigma_log"])))))
            out_until[athlete] = week + 1 + int(np.ceil(days / 7.0))
            rows.append((int(athlete), week, group, days, int(rng.integers(0, 7))))
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 5)
    return InjuryEvents(
        athlete_index=table[:, 0],
        week=table[:, 1],
        outcome_group=np.asarray([OUTCOME_GROUPS[k] for k in table[:, 2]], dtype=np.str_),
        time_loss_days=table[:, 3],
        onset_day_in_week=table[:, 4],
        at_risk=at_risk,
    )
