"""Synthetic growth curves: shape, known APHV, plausibility."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from youthtwin.sim import growth_sim
from youthtwin.sim.cohort import SyntheticCohort
from youthtwin.utils.seeding import make_rng

AGES = np.linspace(7.0, 19.0, 1201)


@pytest.fixture(scope="module")
def curves(cohort_config: dict[str, Any]) -> growth_sim.GrowthCurves:
    rng = make_rng(1, "test_growth")
    is_female = rng.random(200) < 0.5
    params = growth_sim.sample_growth_params(rng, is_female, cohort_config["growth"])
    return growth_sim.GrowthCurves(params, cohort_config["growth"])


def _on_grid(curves: growth_sim.GrowthCurves) -> growth_sim.GrowthCurves:
    """Curves aligned to a flat (athlete x age) grid."""
    index = np.repeat(np.arange(len(curves.params)), AGES.size)
    return curves.at(index)


def _grid_ages(curves: growth_sim.GrowthCurves) -> np.ndarray:
    return np.tile(AGES, len(curves.params))


def test_pb1_height_reference_at_theta() -> None:
    # At t = theta: h = h1 - 2 (h1 - h_theta) / (1 + 1) = h_theta
    one = np.array([1.0])
    height = growth_sim.pb1_height(
        14.0 * one, 175.0 * one, 163.0 * one, 0.11 * one, 1.2 * one, 14.0 * one
    )
    assert height[0] == pytest.approx(163.0)
    # ... and velocity = 2 (h1 - h_theta) (s0 + s1) / 4 = 12 * 1.31 / 2 = 7.86
    velocity = growth_sim.pb1_velocity(
        14.0 * one, 175.0 * one, 163.0 * one, 0.11 * one, 1.2 * one, 14.0 * one
    )
    assert velocity[0] == pytest.approx(7.86)


def test_analytic_velocity_matches_numerical_derivative() -> None:
    args = tuple(np.array([v]) for v in (175.0, 163.0, 0.11, 1.2, 14.0))
    age, step = np.array([12.5]), 1e-5
    numeric = (
        growth_sim.pb1_height(age + step, *args) - growth_sim.pb1_height(age - step, *args)
    ) / (2 * step)
    assert growth_sim.pb1_velocity(age, *args)[0] == pytest.approx(numeric[0], rel=1e-6)


def test_height_is_monotone_and_velocity_positive(curves: growth_sim.GrowthCurves) -> None:
    grid = _on_grid(curves)
    ages = _grid_ages(curves)
    stature = grid.stature(ages).reshape(len(curves.params), -1)
    assert np.all(np.diff(stature, axis=1) > 0)
    assert np.all(grid.height_velocity(ages) > 0)


def test_true_aphv_is_exactly_the_velocity_peak(curves: growth_sim.GrowthCurves) -> None:
    params = curves.params
    step = 1e-3
    at_peak = curves.height_velocity(params.aphv)
    assert np.all(at_peak > curves.height_velocity(params.aphv - step))
    assert np.all(at_peak > curves.height_velocity(params.aphv + step))
    # Peak velocity in a plausible adolescent range (cm/year).
    assert float(at_peak.min()) > 4.5 and float(at_peak.max()) < 13.0


def test_peak_offset_rejects_degenerate_rates() -> None:
    with pytest.raises(ValueError, match="No pubertal velocity peak"):
        growth_sim.pb1_peak_offset(np.array([0.001]), np.array([0.002]))


def test_peak_offset_finds_the_pubertal_peak_not_the_childhood_edge() -> None:
    """Regression: with s0 = 0.148, s1 = 0.905 the childhood velocity 4 years before theta is
    marginally higher than the pubertal peak, and a global maximum picked the window edge."""
    s0, s1 = np.array([0.148]), np.array([0.905])
    offset = growth_sim.pb1_peak_offset(s0, s1)[0]
    assert -2.5 < offset < 0.5  # an interior point, not the edge at -4
    shape = growth_sim._pb1_velocity_shape
    here = shape(np.array([offset]), s0, s1)[0]
    assert here > shape(np.array([offset - 0.01]), s0, s1)[0]
    assert here > shape(np.array([offset + 0.01]), s0, s1)[0]
    assert shape(np.array([-4.0]), s0, s1)[0] > here  # the edge really is higher


def test_aphv_distribution_matches_config(
    curves: growth_sim.GrowthCurves, cohort_config: dict[str, Any]
) -> None:
    spec = cohort_config["growth"]["aphv_years"]
    params = curves.params
    for sex, mask in (("female", params.is_female), ("male", ~params.is_female)):
        standard_error = spec[sex]["sd"] / np.sqrt(mask.sum())
        assert abs(params.aphv[mask].mean() - spec[sex]["mean"]) < 4 * standard_error


def test_body_proportions_are_plausible(curves: growth_sim.GrowthCurves) -> None:
    grid = _on_grid(curves)
    ages = _grid_ages(curves)
    ratio = grid.sitting_ratio(ages)
    assert float(ratio.min()) > 0.47 and float(ratio.max()) < 0.59
    np.testing.assert_allclose(
        grid.leg_length(ages) + grid.sitting_height(ages), grid.stature(ages)
    )
    # Schema lower bound, over the cohort's age range (baseline 8-17 y). Before 8 y a very late
    # maturer's Preece-Baines curve can dip below it (one simulated boy at 7.0-7.15 y).
    assert np.all(grid.mass(ages)[ages >= 8.0] > 10.0)


def test_legs_peak_before_trunk(curves: growth_sim.GrowthCurves) -> None:
    """Around PHV, leg-length velocity peaks earlier than sitting-height velocity."""
    grid = _on_grid(curves)
    ages = _grid_ages(curves)
    n = len(curves.params)
    near_phv = np.abs(AGES[None, :] - curves.params.aphv[:, None]) <= 2.0
    leg = np.where(near_phv, grid.leg_length_velocity(ages).reshape(n, -1), -np.inf)
    trunk = np.where(near_phv, grid.sitting_height_velocity(ages).reshape(n, -1), -np.inf)
    leg_peak, trunk_peak = AGES[np.argmax(leg, axis=1)], AGES[np.argmax(trunk, axis=1)]
    assert np.mean(leg_peak < trunk_peak) > 0.9
    assert np.all(np.isfinite(grid.mass_velocity(ages)))


def test_visits_are_sparse_and_within_follow_up(cohort_config: dict[str, Any]) -> None:
    visits = growth_sim.sample_visits(make_rng(3, "v"), 100, 364, cohort_config["visits"])
    assert visits.day.min() >= 0 and visits.day.max() < 364
    assert set(np.unique(visits.athlete_index)) == set(range(100))  # everyone keeps >= 1 visit
    low, high = cohort_config["visits"]["interval_days"]
    for athlete in range(100):
        gaps = np.diff(visits.day[visits.athlete_index == athlete])
        assert np.all(gaps >= low)
        assert np.all(gaps % 1 == 0)
        # a gap is one interval, or several if visits were missed
        assert np.all(gaps <= high * 8)


def test_all_visits_missed_still_keeps_one() -> None:
    config = {
        "interval_days": [42, 56],
        "first_visit_within_days": 14,
        "missed_visit_probability": 1.0,
    }
    visits = growth_sim.sample_visits(make_rng(3, "v"), 5, 364, config)
    assert np.array_equal(np.bincount(visits.athlete_index), np.ones(5, dtype=np.int64))


def test_measure_twice_adds_independent_error() -> None:
    truth = np.full(20000, 150.0)
    first, second = growth_sim.measure_twice(make_rng(5, "m"), truth, error_sd=0.3)
    assert np.std(first - truth) == pytest.approx(0.3, abs=0.02)
    assert abs(np.corrcoef(first - truth, second - truth)[0, 1]) < 0.03
    assert np.allclose(first, np.round(first, 1))


def test_cohort_measurements_track_the_truth(cohort: SyntheticCohort) -> None:
    measured = cohort.tables["anthropometrics"]
    assert (measured["stature_1"] - measured["stature_2"]).abs().max() < 2.0
    assert measured["stature_1"].between(90.0, 210.0).all()


def test_every_simulated_child_has_a_spurt(cohort_config: dict[str, Any]) -> None:
    """Wide (real-data) spreads of s0 and s1 occasionally give no spurt; those are redrawn."""
    from youthtwin.sim.growth_sim import _has_spurt, pb1_peak_offset, sample_growth_params

    assert not _has_spurt(np.array([0.135]), np.array([0.76]))[0]  # a known no-spurt pair
    is_female = np.arange(20000) % 2 == 0
    params = sample_growth_params(np.random.default_rng(0), is_female, cohort_config["growth"])
    assert _has_spurt(params.s0, params.s1).all()
    np.testing.assert_allclose(params.theta + pb1_peak_offset(params.s0, params.s1), params.aphv)
