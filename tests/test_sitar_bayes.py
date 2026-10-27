"""Growth model: data preparation, the marginal likelihood, and exact athlete-effect draws.

The sampler itself is exercised by one small ``slow`` test; acceptance numbers for the growth
twin come from ``scripts/sim_study_growth.py``, not from unit tests.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from youthtwin.growth import sitar_bayes, splines
from youthtwin.growth.sitar_bayes import GrowthData, SharedDraws
from youthtwin.sim.cohort import SyntheticCohort
from youthtwin.utils.config import load_config
from youthtwin.utils.plotting import is_watermarked, save_figure, watermark_synthetic

KNOTS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    settings = load_config("growth/sitar.yaml")
    settings["spline"]["knots_years"] = list(KNOTS)
    settings["marginalisation"].update(timing_points=81, intensity_points=5)
    return settings


def _true_curve() -> tuple[float, np.ndarray]:
    z = np.linspace(-4.0, 4.0, 400)
    design = np.column_stack([z, *splines.basis(z, KNOTS)])
    solution, *_ = np.linalg.lstsq(design, 5.0 * z + 6.0 * np.tanh(z), rcond=None)
    return float(solution[0]), solution[1:]


def _simulate(
    n_athletes: int, seed: int, *, sigma: float = 0.2, visits: int = 7
) -> tuple[GrowthData, dict[str, np.ndarray]]:
    """Data generated from the model itself, with known athlete effects."""
    rng = np.random.default_rng(seed)
    linear, coefs = _true_curve()
    prior = rng.uniform(12.5, 15.5, n_athletes)
    beta = prior + rng.normal(0.0, 0.6, n_athletes)
    alpha = rng.normal(0.0, 5.0, n_athletes)
    start = beta + rng.uniform(-2.5, 1.5, n_athletes)
    age = (start[:, None] + np.arange(visits)[None, :] * 0.14).ravel()
    index = np.repeat(np.arange(n_athletes), visits)
    mean = 150.0 + alpha[index] + splines.spline_value(age - beta[index], linear, coefs, KNOTS)
    data = GrowthData(
        athlete_ids=tuple(f"a{k}" for k in range(n_athletes)),
        athlete_index=index.astype(np.int64),
        age=age,
        stature=mean + rng.normal(0.0, sigma, age.size),
        prior_aphv=prior,
        sex="male",
        synthetic=True,
    )
    return data, {"alpha": alpha, "beta": beta}


def _shared(
    n_draws: int, sigma: float = 0.2, sigma_alpha: float = 5.0, sigma_delta: float = 0.0
) -> SharedDraws:
    linear, coefs = _true_curve()
    ones = np.ones(n_draws)
    return SharedDraws(
        intercept=150.0 * ones,
        linear=linear * ones,
        coefs=np.tile(coefs, (n_draws, 1)),
        sigma=sigma * ones,
        sigma_alpha=sigma_alpha * ones,
        sigma_gamma=0.0 * ones,
        sigma_delta=sigma_delta * ones,
        timing_offset=0.0 * ones,
        sigma_beta=1.0 * ones,
        cohort_mean=np.nan * ones,
        cohort_sd=np.nan * ones,
    )


# ------------------------------------------------------------------------- data preparation


def test_prepare_growth_data_from_cohort(cohort: SyntheticCohort) -> None:
    athletes, visits = cohort.tables["athletes"], cohort.tables["anthropometrics"]
    data = sitar_bayes.prepare_growth_data(athletes, visits, "female")
    girls = set(athletes.loc[athletes["sex"] == "female", "id"])
    assert set(data.athlete_ids) == girls and data.sex == "female" and data.synthetic
    assert data.n_observations == int(visits["athlete_id"].isin(girls).sum())
    assert data.age.min() > 7.8 and data.age.max() < 18.2
    # Stature is the mean of the two duplicates.
    first = visits[visits["athlete_id"] == data.athlete_ids[0]].sort_values("date").iloc[0]
    assert data.stature[data.athlete_index == 0][0] == pytest.approx(
        (first["stature_1"] + first["stature_2"]) / 2
    )
    assert np.all((data.prior_aphv > 8.0) & (data.prior_aphv < 18.0))
    assert data.n_per_athlete.sum() == data.n_observations


def test_prepare_refuses_mixed_or_empty_data(cohort: SyntheticCohort) -> None:
    athletes, visits = cohort.tables["athletes"], cohort.tables["anthropometrics"]
    mixed = athletes.copy()
    mixed.loc[mixed.index[0], "synthetic"] = False
    with pytest.raises(ValueError, match="mix of synthetic and real"):
        sitar_bayes.prepare_growth_data(mixed, visits, "male")
    with pytest.raises(ValueError, match="No anthropometric data"):
        sitar_bayes.prepare_growth_data(athletes[athletes["sex"] == "male"], visits, "female")


def test_padded_layout_round_trips() -> None:
    data = GrowthData(
        athlete_ids=("a", "b", "c"),
        athlete_index=np.array([1, 0, 1, 2, 1], dtype=np.int64),
        age=np.array([11.0, 10.0, 11.5, 12.0, 12.5]),
        stature=np.array([141.0, 130.0, 143.0, 150.0, 146.0]),
        prior_aphv=np.array([13.0, 13.5, 14.0]),
        sex="male",
        synthetic=True,
    )
    age, stature, mask = data.padded()
    assert age.shape == (3, 3)
    assert mask.tolist() == [[1, 0, 0], [1, 1, 1], [1, 0, 0]]
    assert age[1].tolist() == [11.0, 11.5, 12.5] and stature[1].tolist() == [141.0, 143.0, 146.0]
    assert stature[2, 0] == 150.0 and data.n_per_athlete.tolist() == [1, 3, 1]


# ------------------------------------------------------------------------- marginal likelihood


def test_grids_are_normalised_and_respect_the_intensity_switch(config: dict[str, Any]) -> None:
    grids = sitar_bayes.effect_grids(config)
    assert np.exp(grids.timing_log_weight).sum() == pytest.approx(1.0)
    assert np.exp(grids.intensity_log_weight).sum() == pytest.approx(1.0)
    assert grids.timing.size == 81 and grids.timing_step == pytest.approx(0.1)
    disabled = copy.deepcopy(config)
    disabled["intensity"]["enabled"] = False
    off = sitar_bayes.effect_grids(disabled)
    assert off.intensity.tolist() == [0.0] and off.intensity_log_weight.tolist() == [0.0]


def _weights(data: GrowthData, config: dict[str, Any], xp: Any = np, **overrides: Any) -> Any:
    linear, coefs = _true_curve()
    age, stature, mask = data.padded()
    arguments: dict[str, Any] = {
        "age": age,
        "stature": stature,
        "mask": mask,
        "n_visits": data.n_per_athlete.astype(np.float64),
        "prior_aphv": data.prior_aphv,
        "grids": sitar_bayes.effect_grids(config),
        "knots": KNOTS,
        "intercept": 150.0,
        "linear": linear,
        "coefs": coefs,
        "sigma": 0.2,
        "sigma_alpha": 5.0,
        "sigma_beta": 1.0,
        "sigma_gamma": 0.1,
        "sigma_delta": 0.4,
        "timing_offset": 0.3,
    }
    arguments.update(overrides)
    return sitar_bayes._grid_log_weights(xp, **arguments)


def test_size_effect_is_integrated_out_correctly(config: dict[str, Any]) -> None:
    """Each grid cell equals a multivariate normal density with the full marginal covariance.

    Covariance = noise + size effect (all ones) + velocity deviation (centred ages), and the
    timing prior centre includes the cohort-level offset.
    """
    data, _ = _simulate(3, seed=1, visits=5)
    grids = sitar_bayes.effect_grids(config)
    log_weight, _, _ = _weights(data, config)
    linear, coefs = _true_curve()
    for athlete, b, g in ((0, 40, 2), (1, 10, 0), (2, 70, 4)):
        mine = data.athlete_index == athlete
        beta = data.prior_aphv[athlete] + 0.3 + 1.0 * grids.timing[b]
        z = (data.age[mine] - beta) * np.exp(0.1 * grids.intensity[g])
        mean = 150.0 + splines.spline_value(z, linear, coefs, KNOTS)
        n = int(mine.sum())
        centred = data.age[mine] - data.age[mine].mean()
        covariance = (
            0.2**2 * np.eye(n) + 5.0**2 * np.ones((n, n)) + 0.4**2 * np.outer(centred, centred)
        )
        expected = (
            stats.multivariate_normal(mean, covariance).logpdf(data.stature[mine])
            + grids.timing_log_weight[b]
            + grids.intensity_log_weight[g]
        )
        assert log_weight[athlete, b, g] == pytest.approx(expected, rel=1e-9)


def test_padding_does_not_change_the_likelihood(config: dict[str, Any]) -> None:
    """An athlete with fewer visits gets the same weights alone as when padded next to others."""
    data, _ = _simulate(2, seed=2, visits=6)
    keep = ~((data.athlete_index == 0) & (np.arange(data.n_observations) % 6 >= 3))
    ragged = GrowthData(
        data.athlete_ids,
        data.athlete_index[keep],
        data.age[keep],
        data.stature[keep],
        data.prior_aphv,
        "male",
        True,
    )
    alone = GrowthData(
        ("a0",),
        np.zeros(3, dtype=np.int64),
        ragged.age[:3],
        ragged.stature[:3],
        data.prior_aphv[:1],
        "male",
        True,
    )
    np.testing.assert_allclose(_weights(ragged, config)[0][0], _weights(alone, config)[0][0])


def test_numpy_and_pytensor_likelihoods_agree(config: dict[str, Any]) -> None:
    import pytensor.tensor as pt

    data, _ = _simulate(4, seed=3)
    numeric = _weights(data, config)
    symbolic = _weights(data, config, xp=pt)
    for ours, theirs in zip(symbolic, numeric, strict=True):
        np.testing.assert_allclose(ours.eval(), theirs, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------------- athlete effects


def test_athlete_effects_recover_known_timing_and_size(config: dict[str, Any]) -> None:
    """Given the true shared curve, 90% intervals for beta and alpha cover the truth as stated."""
    data, truth = _simulate(300, seed=4)
    posterior = sitar_bayes.draw_athlete_effects(_shared(200), data, config, seed=5)
    assert posterior.beta.shape == (200, 300) and posterior.synthetic
    for name in ("beta", "alpha"):
        draws = getattr(posterior, name)
        low, high = np.quantile(draws, [0.05, 0.95], axis=0)
        coverage = np.mean((truth[name] >= low) & (truth[name] <= high))
        assert 0.84 <= coverage <= 0.96, (name, coverage)
    error = np.abs(posterior.beta.mean(axis=0) - truth["beta"])
    prior_error = np.abs(data.prior_aphv - truth["beta"])
    assert error.mean() < 0.6 * prior_error.mean()  # the data sharpen the Moore-style prior


def test_velocity_deviation_is_recovered_and_widens_timing(config: dict[str, Any]) -> None:
    """Athletes who grow 1 cm/year faster than the shared shape: delta is found, not timing."""
    data, truth = _simulate(200, seed=21)
    faster = GrowthData(
        **{
            **data.__dict__,
            "stature": data.stature + 1.0 * (data.age - data.mean_age[data.athlete_index]),
        }
    )
    without = sitar_bayes.draw_athlete_effects(_shared(150), faster, config, seed=1)
    allowed = sitar_bayes.draw_athlete_effects(
        _shared(150, sigma_delta=1.0), faster, config, seed=1
    )
    assert np.all(without.delta == 0.0)
    # Partly shrunk by its prior and partly shared with timing, but clearly detected.
    assert 0.3 < allowed.delta.mean() < 1.0
    assert allowed.beta.std(axis=0).mean() > without.beta.std(axis=0).mean()

    def coverage(posterior: Any) -> float:
        low, high = np.quantile(posterior.beta, [0.05, 0.95], axis=0)
        return float(np.mean((truth["beta"] >= low) & (truth["beta"] <= high)))

    # Ignoring the deviation gives confidently wrong timing; allowing it restores coverage.
    assert coverage(without) < 0.6
    assert coverage(allowed) > 0.8


def test_two_peaked_timing_is_represented(config: dict[str, Any]) -> None:
    """Two visits with the same growth speed fit either side of the spurt: both must appear.

    Size is made uninformative (large sigma_alpha) so that only growth speed carries
    information; otherwise height-for-age would settle which side the athlete is on.
    """
    linear, coefs = _true_curve()
    beta_true = 14.0
    age = np.array([12.9, 13.1])  # about one year before the peak
    stature = 150.0 + splines.spline_value(age - beta_true, linear, coefs, KNOTS)
    data = GrowthData(
        ("a",), np.zeros(2, dtype=np.int64), age, stature, np.array([13.0]), "male", True
    )
    shared = _shared(2000, sigma_alpha=200.0)
    posterior = sitar_bayes.draw_athlete_effects(shared, data, config, seed=6)
    beta = posterior.beta[:, 0]
    before_peak = np.mean(beta > 13.0)  # athlete is before their peak (true)
    assert 0.1 < before_peak < 0.9  # neither explanation is ruled out
    assert np.std(beta) > 0.5


def test_more_visits_narrow_the_posterior_and_update_matches_refit(
    config: dict[str, Any],
) -> None:
    data, truth = _simulate(40, seed=7, visits=8)
    early = data.athlete_index >= 0
    early &= np.tile(np.arange(8), 40) < 3  # first three visits only
    first = GrowthData(
        data.athlete_ids,
        data.athlete_index[early],
        data.age[early],
        data.stature[early],
        data.prior_aphv,
        "male",
        True,
    )
    before = sitar_bayes.draw_athlete_effects(_shared(300), first, config, seed=8)
    after = sitar_bayes.update(before, data, config, seed=9)
    assert after.beta.std(axis=0).mean() < 0.7 * before.beta.std(axis=0).mean()
    assert (
        np.abs(after.beta.mean(axis=0) - truth["beta"]).mean()
        < np.abs(before.beta.mean(axis=0) - truth["beta"]).mean()
    )
    # Shared draws are carried over unchanged, and the result equals drawing from scratch.
    np.testing.assert_array_equal(after.coefs, before.coefs)
    direct = sitar_bayes.draw_athlete_effects(_shared(300), data, config, seed=9)
    np.testing.assert_allclose(after.beta, direct.beta)


def test_update_can_add_a_new_athlete_and_checks_compatibility(config: dict[str, Any]) -> None:
    data, _ = _simulate(5, seed=10)
    posterior = sitar_bayes.draw_athlete_effects(_shared(50), data, config, seed=1)
    bigger, _ = _simulate(6, seed=10)
    assert sitar_bayes.update(posterior, bigger, config, seed=2).alpha.shape == (50, 6)
    other_sex = GrowthData(**{**data.__dict__, "sex": "female"})
    with pytest.raises(ValueError, match="other sex"):
        sitar_bayes.update(posterior, other_sex, config, seed=2)
    real = GrowthData(**{**data.__dict__, "synthetic": False})
    with pytest.raises(ValueError, match="synthetic and real"):
        sitar_bayes.update(posterior, real, config, seed=2)


# ------------------------------------------------------------------------- model and sampler


def test_initial_values_reproduce_the_curve_at_the_prior(config: dict[str, Any]) -> None:
    data, _ = _simulate(60, seed=11)
    start = sitar_bayes.initial_values(data, config)
    transform = splines.orthogonalising_transform(KNOTS)
    curve = transform @ start["theta"]
    z = data.age - data.prior_aphv[data.athlete_index]
    fitted = data.stature.mean() + curve[0] + splines.spline_value(z, curve[1], curve[2:], KNOTS)
    # Size and timing effects are ignored at the start, so the residual is a few cm, not tens.
    assert np.sqrt(np.mean((fitted - data.stature) ** 2)) < 8.0


def test_model_builds_and_likelihood_prefers_the_truth(config: dict[str, Any]) -> None:
    data, _ = _simulate(30, seed=12)
    model = sitar_bayes.build_model(data, config)
    assert {v.name for v in model.free_RVs} == {
        "theta",
        "sigma",
        "sigma_alpha",
        "sigma_gamma",
        "sigma_delta",
        "timing_offset",
        "sigma_beta",
    }
    logp = model.compile_logp()
    linear, coefs = _true_curve()
    transform = splines.orthogonalising_transform(KNOTS)
    target = np.concatenate([[150.0 - data.stature.mean()], [linear], coefs])
    point = model.initial_point()
    point["theta"] = np.linalg.solve(transform, target)
    at_truth = float(logp(point))
    point["theta"] = point["theta"] * 0.0
    assert at_truth > float(logp(point)) + 100.0

    disabled = copy.deepcopy(config)
    disabled["intensity"]["enabled"] = False
    disabled["velocity_deviation"]["enabled"] = False
    disabled["timing_prior"].update(learn_sd=False, learn_offset=False)
    free = {v.name for v in sitar_bayes.build_model(data, disabled).free_RVs}
    assert free == {"theta", "sigma", "sigma_alpha"}


def test_unknown_inference_method_is_rejected(config: dict[str, Any]) -> None:
    data, _ = _simulate(5, seed=13)
    bad = copy.deepcopy(config)
    bad["inference"]["method"] = "magic"
    with pytest.raises(ValueError, match="Unknown inference method"):
        sitar_bayes.fit(data, bad, seed=1)


@pytest.mark.slow
def test_small_fit_end_to_end(config: dict[str, Any]) -> None:
    """A short NUTS run on model-generated data recovers the residual SD and athlete timing."""
    data, truth = _simulate(40, seed=14)
    quick = copy.deepcopy(config)
    quick["marginalisation"].update(timing_points=41, intensity_points=3)
    quick["inference"].update(draws=150, tune=250, chains=2, cores=2)
    quick["growth_state"]["n_posterior_draws"] = 200
    idata = sitar_bayes.fit(data, quick, seed=15)
    summary = sitar_bayes.diagnostics(idata)
    assert set(summary) == {"max_rhat", "min_ess_bulk", "n_divergences"}
    posterior = sitar_bayes.extract_posterior(idata, data, quick, seed=16)
    assert posterior.n_draws == 200 and posterior.has_interior_peak().mean() > 0.9
    assert posterior.sigma.mean() == pytest.approx(0.2, abs=0.08)
    error = np.abs(posterior.beta.mean(axis=0) - truth["beta"]).mean()
    assert error < np.abs(data.prior_aphv - truth["beta"]).mean()


# ------------------------------------------------------------------------- figures


def test_synthetic_figures_are_watermarked(tmp_path: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    assert not is_watermarked(figure)
    path = save_figure(figure, tmp_path / "f.png", {"synthetic": True, "run_id": "t"})
    assert is_watermarked(figure) and path.is_file()
    assert (tmp_path / "f.png.provenance.json").is_file()
    watermark_synthetic(figure)
    assert sum(text.get_text() == "SYNTHETIC" for text in figure.texts) == 1  # idempotent

    real, _ = plt.subplots()
    save_figure(real, tmp_path / "r.png", {"synthetic": False})
    assert not is_watermarked(real)
    with pytest.raises(ValueError, match="synthetic"):
        save_figure(real, tmp_path / "x.png", {"run_id": "t"})
    plt.close("all")


def test_shared_draws_from_inference_data() -> None:
    import arviz as az

    rng = np.random.default_rng(0)
    posterior = {
        "intercept": rng.normal(150, 1, (2, 30)),
        "linear": rng.normal(5, 0.1, (2, 30)),
        "coefs": rng.normal(0, 1, (2, 30, 5)),
        "sigma": np.abs(rng.normal(0.2, 0.01, (2, 30))),
        "sigma_alpha": np.abs(rng.normal(5, 0.1, (2, 30))),
    }
    idata = az.from_dict(posterior=posterior)
    settings = load_config("growth/sitar.yaml")
    shared = sitar_bayes.shared_draws(idata, 20, 1, settings)
    assert shared.n_draws == 20 and shared.coefs.shape == (20, 5)
    # Parameters absent from the fit take their fixed values.
    assert np.all(shared.sigma_gamma == 0.0) and np.all(shared.sigma_delta == 0.0)
    assert np.all(shared.timing_offset == 0.0)
    assert np.all(shared.sigma_beta == settings["timing_prior"]["sd_years"])
    everything = sitar_bayes.shared_draws(idata, 500, 1, settings)
    assert everything.n_draws == 60
    assert isinstance(pd.Series(everything.sigma).mean(), float)


# ------------------------------------------------------------------- distance-dependent prior


def test_unit_prior_scale_leaves_the_likelihood_unchanged(config: dict[str, Any]) -> None:
    data, _ = _simulate(6, seed=21)
    plain = _weights(data, config)[0]
    scaled = _weights(data, config, prior_scale=np.ones(data.n_athletes))[0]
    np.testing.assert_allclose(scaled, plain)
    # Doubling an athlete's prior scale is the same as doubling sigma_beta for that athlete.
    doubled = _weights(data, config, prior_scale=np.full(data.n_athletes, 2.0))[0]
    np.testing.assert_allclose(doubled, _weights(data, config, sigma_beta=2.0)[0])


def test_slope_widens_timing_only_for_athletes_far_from_phv(config: dict[str, Any]) -> None:
    """With little data, a positive slope widens the timing posterior of distant athletes."""
    data, _ = _simulate(2, seed=22, visits=2)
    data = GrowthData(**{**data.__dict__, "prior_distance": np.array([0.0, 3.0])})
    weak = _shared(300, sigma=20.0)  # nearly uninformative data: the prior dominates
    flat = sitar_bayes.draw_athlete_effects(weak, data, config, seed=1)
    shared = SharedDraws(**{**weak.__dict__, "sigma_beta_slope": np.full(300, 0.5)})
    wide = sitar_bayes.draw_athlete_effects(shared, data, config, seed=1)
    spread_flat, spread_wide = flat.beta.std(axis=0), wide.beta.std(axis=0)
    assert spread_wide[0] == pytest.approx(spread_flat[0], rel=0.15)  # distance 0: unchanged
    assert spread_wide[1] > 1.5 * spread_flat[1]  # distance 3 y: prior SD x 2.5
    np.testing.assert_allclose(wide.sigma_beta_slope, 0.5)
    # The update path carries the slope through.
    again = sitar_bayes.update(wide, data, config, seed=1)
    np.testing.assert_allclose(again.beta, wide.beta)


def test_prior_can_come_from_other_visits() -> None:
    """A reference child fitted on the full record can take its prior from one season."""
    ages = np.arange(9.0, 17.01, 0.5)
    frame = pd.DataFrame(
        {
            "athlete_id": "c1",
            "age": ages,
            "stature": 130.0 + 5.0 * (ages - 9.0),
            "sitting_height": 0.52 * (130.0 + 5.0 * (ages - 9.0)),
        }
    )
    season = frame[frame["age"].between(9.0, 10.0)]
    full = sitar_bayes.growth_data_from_measurements(frame, "male", synthetic=True)
    seasonal = sitar_bayes.growth_data_from_measurements(
        frame, "male", synthetic=True, prior_frame=season
    )
    assert seasonal.n_observations == full.n_observations == ages.size
    assert seasonal.prior_aphv[0] != pytest.approx(full.prior_aphv[0])
    assert seasonal.distance[0] > full.distance[0]  # one early season is far from PHV
    with pytest.raises(ValueError, match="No prior visits"):
        sitar_bayes.growth_data_from_measurements(
            frame, "male", synthetic=True, prior_frame=season.assign(athlete_id="other")
        )


def test_posteriors_saved_without_a_slope_still_load(tmp_path: Any, config: dict[str, Any]) -> None:
    data, _ = _simulate(3, seed=23)
    posterior = sitar_bayes.draw_athlete_effects(_shared(20), data, config, seed=0)
    path = posterior.save(tmp_path / "post")
    with np.load(path) as stored:
        arrays = {k: stored[k] for k in stored.files if k != "sigma_beta_slope"}
    np.savez_compressed(path, **arrays)
    loaded = type(posterior).load(tmp_path / "post")
    np.testing.assert_allclose(loaded.sigma_beta_slope, 0.0)


# ------------------------------------------------------------------- bivariate (leg) twin


def _leg_curve() -> tuple[float, np.ndarray]:
    """Leg growth peaks about 0.6 y before stature on the transformed axis."""
    z = np.linspace(-4.0, 4.0, 400)
    design = np.column_stack([z, *splines.basis(z, KNOTS)])
    solution, *_ = np.linalg.lstsq(design, 2.5 * z + 3.5 * np.tanh(1.3 * (z + 0.6)), rcond=None)
    return float(solution[0]), solution[1:]


def _bivariate(n: int, visits: int, seed: int) -> tuple[GrowthData, np.ndarray]:
    rng = np.random.default_rng(seed)
    linear, coefs = _true_curve()
    leg_linear, leg_coefs = _leg_curve()
    prior = rng.uniform(12.5, 15.5, n)
    beta = prior + rng.normal(0.0, 0.8, n)
    start = beta + rng.uniform(-2.0, 1.0, n)
    age = (start[:, None] + np.arange(visits)[None, :] * 0.5).ravel()
    index = np.repeat(np.arange(n), visits)
    z = age - beta[index]
    stature = 150.0 + rng.normal(0, 5.0, n)[index] + splines.spline_value(z, linear, coefs, KNOTS)
    leg = (
        75.0 + rng.normal(0, 3.0, n)[index] + splines.spline_value(z, leg_linear, leg_coefs, KNOTS)
    )
    data = GrowthData(
        athlete_ids=tuple(f"a{k}" for k in range(n)),
        athlete_index=index.astype(np.int64),
        age=age,
        stature=stature + rng.normal(0.0, 0.3, age.size),
        prior_aphv=prior,
        sex="male",
        synthetic=True,
        leg=leg + rng.normal(0.0, 0.3, age.size),
    )
    return data, beta


def _with_leg(shared: SharedDraws) -> SharedDraws:
    leg_linear, leg_coefs = _leg_curve()
    ones = np.ones(shared.n_draws)
    leg = sitar_bayes.LegCurve(
        intercept=75.0 * ones,
        linear=leg_linear * ones,
        coefs=np.tile(leg_coefs, (shared.n_draws, 1)),
        sigma=0.3 * ones,
        sigma_alpha=3.0 * ones,
        sigma_delta=0.0 * ones,
    )
    return SharedDraws(**{**shared.__dict__, "leg": leg})


def test_leg_length_sharpens_timing_from_few_visits(config: dict[str, Any]) -> None:
    data, beta = _bivariate(40, visits=3, seed=31)
    shared = _shared(300, sigma=0.3)
    height_only = sitar_bayes.draw_athlete_effects(shared, data, config, seed=2)
    both = sitar_bayes.draw_athlete_effects(_with_leg(shared), data, config, seed=2)
    sd_height, sd_both = height_only.beta.std(axis=0), both.beta.std(axis=0)
    assert np.median(sd_both) < 0.8 * np.median(sd_height)
    error_height = np.abs(height_only.beta.mean(axis=0) - beta).mean()
    error_both = np.abs(both.beta.mean(axis=0) - beta).mean()
    assert error_both < error_height
    # Without leg data, or with the leg curve absent, nothing changes.
    no_leg = GrowthData(**{**data.__dict__, "leg": None})
    again = sitar_bayes.draw_athlete_effects(_with_leg(shared), no_leg, config, seed=2)
    np.testing.assert_allclose(again.beta, height_only.beta)


def test_bivariate_model_builds_and_starts_at_finite_density(config: dict[str, Any]) -> None:
    data, _ = _bivariate(8, visits=4, seed=32)
    settings = copy.deepcopy(config)
    settings["bivariate"]["enabled"] = True
    model = sitar_bayes.build_model(data, settings)
    assert {"theta_leg", "sigma_leg", "sigma_alpha_leg"} <= set(model.named_vars)
    start = {**model.initial_point(), **sitar_bayes.initial_values(data, settings)}
    assert np.isfinite(model.compile_logp()(start))
    # Disabled: the stature-only model.
    plain = sitar_bayes.build_model(data, config)
    assert "theta_leg" not in plain.named_vars
