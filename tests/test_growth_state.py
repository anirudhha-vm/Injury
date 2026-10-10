"""Growth state from a hand-built posterior whose mean curve and athlete effects are known."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from youthtwin.growth import evaluation, splines
from youthtwin.growth.growth_state import SitarPosterior, Summary, age_on

KNOTS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
N_DRAWS = 50
IDS = ("early", "late", "intense")
BETA = np.array([12.0, 15.0, 14.0])
GAMMA = np.array([0.0, 0.0, 0.3])
ALPHA = np.array([3.0, -2.0, 0.0])
DELTA = np.zeros(3)
MEAN_AGE = np.array([12.0, 14.0, 13.0])


def _curve() -> tuple[float, np.ndarray]:
    """Least-squares spline for h(z) = 5 z + 6 tanh(z): velocity 5 + 6 sech^2(z), peak at 0."""
    z = np.linspace(-4.0, 4.0, 400)
    design = np.column_stack([z, *splines.basis(z, KNOTS)])
    solution, *_ = np.linalg.lstsq(design, 5.0 * z + 6.0 * np.tanh(z), rcond=None)
    return float(solution[0]), solution[1:]


def _posterior(spread: float = 0.0, synthetic: bool = True) -> SitarPosterior:
    linear, coefs = _curve()
    rng = np.random.default_rng(0)
    noise = spread * rng.standard_normal((N_DRAWS, len(IDS)))
    ones = np.ones(N_DRAWS)
    return SitarPosterior(
        athlete_ids=IDS,
        knots=KNOTS,
        intercept=150.0 * ones,
        linear=linear * ones,
        coefs=np.tile(coefs, (N_DRAWS, 1)),
        alpha=np.tile(ALPHA, (N_DRAWS, 1)),
        beta=BETA + noise,
        gamma=np.tile(GAMMA, (N_DRAWS, 1)),
        sigma=0.2 * ones,
        sigma_alpha=5.0 * ones,
        sigma_gamma=0.1 * ones,
        delta=np.tile(DELTA, (N_DRAWS, 1)),
        sigma_delta=0.5 * ones,
        timing_offset=0.0 * ones,
        sigma_beta=1.0 * ones,
        cohort_mean=np.nan * ones,
        cohort_sd=np.nan * ones,
        mean_age=MEAN_AGE,
        sex="male",
        synthetic=synthetic,
    )


def test_age_on_uses_mid_month() -> None:
    assert age_on("2014-03", date(2026, 3, 15)) == pytest.approx(12.0, abs=0.01)
    assert age_on("2014-03", date(2026, 9, 15)) == pytest.approx(12.5, abs=0.01)


def test_summary_statistics() -> None:
    summary = Summary(np.arange(101, dtype=np.float64))
    assert summary.mean == pytest.approx(50.0)
    assert summary.sd == pytest.approx(np.std(np.arange(101), ddof=1))
    assert summary.interval(0.90) == pytest.approx((5.0, 95.0))
    assert Summary(np.array([3.0])).sd == 0.0


def test_mean_curve_peak_is_found() -> None:
    posterior = _posterior()
    location, height = posterior.z_peak()
    # Independent dense search of the fitted spline's derivative.
    linear, coefs = _curve()
    grid = np.linspace(-4.0, 4.0, 800001)
    slope = splines.spline_derivative(grid, linear, coefs, KNOTS)
    assert np.allclose(location, grid[np.argmax(slope)], atol=1e-3)
    assert np.allclose(height, slope.max(), rtol=1e-6)
    # The spline approximates 5 + 6 sech^2(z): peak near z = 0 and near 11 cm/year.
    assert abs(location[0]) < 0.4 and height[0] == pytest.approx(11.0, abs=0.8)
    assert posterior.has_interior_peak().all()


def test_aphv_is_the_age_of_maximum_velocity_of_each_athletes_curve() -> None:
    """Independent check: search each athlete's own velocity curve over age."""
    posterior = _posterior()
    aphv = posterior.aphv()
    ages = np.arange(9.0, 19.0, 0.002)
    for k, athlete in enumerate(IDS):
        velocity = np.array([posterior.height_velocity(athlete, a)[0] for a in ages])
        assert aphv[0, k] == pytest.approx(ages[np.argmax(velocity)], abs=0.01)
    # aphv = beta + z_peak * exp(-gamma): intensity shrinks the distance from beta to the peak.
    location, _ = posterior.z_peak()
    np.testing.assert_allclose(aphv[0], BETA + location[0] * np.exp(-GAMMA))
    assert not np.allclose(aphv[0], BETA)  # beta alone is not APHV


def test_intensity_raises_peak_velocity_and_size_shifts_stature() -> None:
    posterior = _posterior()
    late = posterior.height_velocity("late", 15.0)[0]
    intense = posterior.height_velocity("intense", 14.0)[0]
    assert intense == pytest.approx(late * np.exp(0.3), rel=0.02)
    early_at_peak = posterior.stature("early", 12.0)[0]
    late_at_peak = posterior.stature("late", 15.0)[0]
    assert early_at_peak - late_at_peak == pytest.approx(ALPHA[0] - ALPHA[1], abs=1e-6)


def test_velocity_deviation_shifts_velocity_but_not_age_at_phv() -> None:
    base = _posterior()
    faster = SitarPosterior(
        **{**base.__dict__, "delta": np.tile(np.array([0.0, 0.8, 0.0]), (N_DRAWS, 1))}
    )
    np.testing.assert_allclose(
        faster.height_velocity("late", 13.5), base.height_velocity("late", 13.5) + 0.8
    )
    # The deviation pivots on the athlete's mean age: no change there, 0.8 cm one year later.
    np.testing.assert_allclose(faster.stature("late", 14.0), base.stature("late", 14.0))
    np.testing.assert_allclose(faster.stature("late", 15.0), base.stature("late", 15.0) + 0.8)
    np.testing.assert_allclose(faster.aphv(), base.aphv())
    peak_age = float(base.aphv()[0, 1])
    assert faster.growth_state("late", peak_age).pct_peak_velocity.mean == pytest.approx(100.0)
    np.testing.assert_allclose(
        faster.height_velocity("early", 12.0), base.height_velocity("early", 12.0)
    )


def test_growth_state_quantities() -> None:
    posterior = _posterior()
    peak_age = float(posterior.aphv()[0, IDS.index("late")])
    before = posterior.growth_state("late", peak_age - 2.0)
    at_peak = posterior.growth_state("late", peak_age)
    after = posterior.growth_state("late", peak_age + 2.0)
    assert before.maturity_offset.mean == pytest.approx(-2.0)
    assert after.maturity_offset.mean == pytest.approx(2.0)
    assert at_peak.pct_peak_velocity.mean == pytest.approx(100.0, abs=0.01)
    assert before.pct_peak_velocity.mean < 100.0 and after.pct_peak_velocity.mean < 100.0
    for state in (before, at_peak, after):
        assert state.height_velocity.mean > 0
        assert state.forecast_stature_gain.mean > 0  # still growing in 90 days
        assert state.synthetic is True
    # A 90-day gain at peak velocity is close to velocity x 90 days.
    assert at_peak.forecast_stature_gain.mean == pytest.approx(
        at_peak.height_velocity.mean * 90 / 365.25, rel=0.05
    )
    longer = posterior.growth_state("late", peak_age, forecast_horizon_days=180)
    assert longer.forecast_stature_gain.mean > at_peak.forecast_stature_gain.mean


def test_uncertainty_is_carried_into_the_state() -> None:
    """Non-negotiable N5: a wider timing posterior gives wider state uncertainty."""
    certain = _posterior(spread=0.0).growth_state("early", 12.5)
    uncertain = _posterior(spread=0.5).growth_state("early", 12.5)
    assert certain.aphv.sd == pytest.approx(0.0, abs=1e-9)
    assert uncertain.aphv.sd == pytest.approx(0.5, rel=0.3)
    assert uncertain.maturity_offset.sd == pytest.approx(uncertain.aphv.sd)
    assert uncertain.height_velocity.sd > 0
    features = uncertain.as_features()
    assert features["aphv_sd"] == pytest.approx(uncertain.aphv.sd)
    assert {name.rsplit("_", 1)[1] for name in features} == {"mean", "sd"}


def test_unknown_athlete_and_bad_shapes() -> None:
    posterior = _posterior()
    with pytest.raises(KeyError, match="not in this posterior"):
        posterior.growth_state("nobody", 12.0)
    with pytest.raises(ValueError, match="athlete_ids"):
        SitarPosterior(**{**posterior.__dict__, "athlete_ids": ("a", "b")})
    with pytest.raises(ValueError, match="unique"):
        SitarPosterior(**{**posterior.__dict__, "athlete_ids": ("a", "a", "b")})
    with pytest.raises(ValueError, match="coefs"):
        SitarPosterior(**{**posterior.__dict__, "coefs": posterior.coefs[:, :2]})


def test_boundary_peak_is_reported_as_not_interior() -> None:
    """A curve that only accelerates has no growth spurt inside the knot range."""
    posterior = _posterior()
    accelerating = SitarPosterior(
        **{**posterior.__dict__, "coefs": np.abs(posterior.coefs) * 0 + 5.0}
    )
    assert not accelerating.has_interior_peak().any()


def test_thin_and_save_round_trip(tmp_path: Path) -> None:
    posterior = _posterior(spread=0.3)
    thinned = posterior.thin(10, seed=1)
    assert thinned.n_draws == 10 and posterior.thin(500) is posterior
    stored = thinned.save(tmp_path / "fit" / "posterior_male", {"run_id": "test"})
    assert stored.suffix == ".npz"
    loaded = SitarPosterior.load(tmp_path / "fit" / "posterior_male")
    assert loaded.athlete_ids == IDS and loaded.synthetic is True and loaded.sex == "male"
    np.testing.assert_allclose(loaded.beta, thinned.beta)
    np.testing.assert_allclose(loaded.aphv(), thinned.aphv())


# ------------------------------------------------------------------------------- scoring


def test_scoring_against_known_truth() -> None:
    posterior = _posterior(spread=0.2)
    truth = dict(zip(IDS, [12.0, 15.0, 17.0], strict=True))  # third athlete is far off
    point = dict(zip(IDS, [12.5, 14.0, 14.0], strict=True))
    scores = evaluation.score_aphv(posterior, truth, point, level=0.90)
    assert list(scores["covered"]) == [True, True, False]
    assert scores["point_abs_error"].tolist() == pytest.approx([0.5, 1.0, 3.0])
    assert scores["posterior_abs_error"].iloc[2] == pytest.approx(3.0, abs=0.15)
    assert scores["synthetic"].all()

    summary = evaluation.summarise_scores(scores, level=0.90, coverage_range=[0.85, 0.95])
    row = summary.iloc[0]
    assert row["group"] == "all" and row["n_athletes"] == 3
    assert row["coverage"] == pytest.approx(2 / 3)
    assert not row["coverage_in_range"]
    assert row["point_mae_years"] == pytest.approx(1.5)
    assert row["posterior_beats_point"]

    by_sex = evaluation.summarise_scores(scores, level=0.90, coverage_range=[0.5, 0.95], by="sex")
    assert by_sex["group"].tolist() == ["male"] and by_sex["coverage_in_range"].all()


def test_maturity_strata() -> None:
    strata = evaluation.maturity_stratum(np.array([-2.0, -1.0, 0.0, 1.0, 2.5]), band_years=1.0)
    assert strata.tolist() == ["pre_phv", "circa_phv", "circa_phv", "circa_phv", "post_phv"]
