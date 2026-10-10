"""Neural hazard, uncertainty propagation, data assembly and the ablation runner."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.eval import metrics
from youthtwin.eval.ablation import RESULT_COLUMNS, run_ablation
from youthtwin.features.person_period import build_person_period
from youthtwin.growth import splines
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models import dataset
from youthtwin.models.encoder.sequences import SessionSequences, channel_names
from youthtwin.models.risk import uncertainty
from youthtwin.models.risk.hazard_neural import NeuralData, Standardiser, fit_neural_hazard
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config

SIDES = ("left", "right")
CHANNELS = channel_names(SIDES, frontal=True)


@pytest.fixture(scope="module")
def encoder_config() -> dict[str, Any]:
    return {**load_config("models/tcn_film.yaml"), "hidden": 16, "n_blocks": 2}


@pytest.fixture(scope="module")
def train_config() -> dict[str, Any]:
    config = load_config("models/neural.yaml")
    config.update(max_epochs=12, patience=4, ensemble_size=2, batch_size=128, learning_rate=0.01)
    return config


def _toy(
    n_athletes: int = 80, weeks: int = 12, seed: int = 0
) -> tuple[NeuralData, SessionSequences]:
    """Risk depends on movement amplitude and, through an interaction, on the growth state.

    One session per athlete with two trials whose first channel has amplitude ``a``; the event
    probability is sigmoid(-1 + 2 * a * g0 + x0).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 101)
    amplitude = rng.normal(0.0, 1.0, n_athletes)
    trials = rng.normal(0, 0.1, (n_athletes, 2, len(CHANNELS), 101))
    trials[:, :, 0, :] += amplitude[:, None, None] * np.sin(np.pi * t)
    sequences = SessionSequences(
        session_ids=tuple(f"s{k}" for k in range(n_athletes)),
        athlete_ids=tuple(f"a{k}" for k in range(n_athletes)),
        trials=trials,
        mask=np.ones((n_athletes, 2)),
        channels=tuple(CHANNELS),
    )
    athlete = np.repeat(np.arange(n_athletes), weeks)
    growth = rng.normal(0, 1, (athlete.size, 2))
    tabular = rng.normal(0, 1, (athlete.size, 2))
    logit = -1.0 + 2.0 * amplitude[athlete] * growth[:, 0] + tabular[:, 0]
    outcome = (rng.random(athlete.size) < 1 / (1 + np.exp(-logit))).astype(float)
    data = NeuralData(
        growth=growth,
        tabular=tabular,
        session_index=athlete.astype(np.int64),
        outcome=outcome,
        athlete=np.array([f"a{k}" for k in athlete]),
    )
    return data, sequences


def _train_mask(data: NeuralData, n_train: int = 60) -> np.ndarray:
    return np.isin(data.athlete, [f"a{k}" for k in range(n_train)])


# ----------------------------------------------------------------------------- preprocessing


def test_standardiser_uses_the_rows_it_is_given_and_fills_missing_with_the_mean() -> None:
    values = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    scaler = Standardiser.fit(values)
    np.testing.assert_allclose(scaler.mean, [3.0, 20.0])
    np.testing.assert_allclose(scaler.sd, [np.sqrt(8 / 3), 10.0])
    out = scaler.transform(np.array([[3.0, np.nan], [5.0, 40.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0], [2 / np.sqrt(8 / 3), 2.0]])
    constant = Standardiser.fit(np.array([[2.0], [2.0]]))
    assert constant.sd[0] == 1.0  # no division by zero
    empty = Standardiser.fit(np.zeros((4, 0)))
    assert empty.transform(np.zeros((4, 0))).shape == (4, 0)


# ----------------------------------------------------------------------------- training


def test_neural_hazard_learns_and_generalises_to_new_athletes(
    encoder_config: dict[str, Any], train_config: dict[str, Any]
) -> None:
    data, sequences = _toy()
    train = _train_mask(data)
    fitted = fit_neural_hazard(
        data,
        sequences,
        train,
        mode="film",
        sides=SIDES,
        encoder_config=encoder_config,
        config=train_config,
        seed=1,
    )
    test = np.flatnonzero(~train)
    risk = fitted.predict_proba(data, sequences, test)
    assert risk.shape == (test.size,) and risk.min() >= 0 and risk.max() <= 1
    assert metrics.auc(data.outcome[test], risk) > 0.7  # held-out athletes
    assert len(fitted.networks) == 2 and len(fitted.history) == 2
    assert all(len(h) <= train_config["max_epochs"] for h in fitted.history)
    members = fitted.predict_members(data, sequences, test)
    assert members.shape == (2, test.size) and not np.allclose(members[0], members[1])
    np.testing.assert_allclose(members.mean(axis=0), risk)


def test_conditioning_is_needed_for_an_interaction_with_growth(
    encoder_config: dict[str, Any], train_config: dict[str, Any]
) -> None:
    """In the toy data movement only matters through growth: an unconditioned encoder cannot
    use it, a FiLM-conditioned one can. (A property of this constructed example.)"""
    data, sequences = _toy(seed=3)
    train = _train_mask(data)
    test = np.flatnonzero(~train)
    scores = {}
    for mode in ("none", "film"):
        fitted = fit_neural_hazard(
            data,
            sequences,
            train,
            mode=mode,
            sides=SIDES,
            encoder_config=encoder_config,
            config=train_config,
            seed=2,
        )
        scores[mode] = metrics.auc(data.outcome[test], fitted.predict_proba(data, sequences, test))
    assert scores["film"] > scores["none"] + 0.05


def test_training_is_reproducible_and_blind_to_held_out_rows(
    encoder_config: dict[str, Any], train_config: dict[str, Any]
) -> None:
    data, sequences = _toy(seed=4)
    train = _train_mask(data)
    quick = {**train_config, "ensemble_size": 1, "max_epochs": 3}
    kwargs = {"mode": "film", "sides": SIDES, "encoder_config": encoder_config, "config": quick}
    first = fit_neural_hazard(data, sequences, train, seed=5, **kwargs)
    second = fit_neural_hazard(data, sequences, train, seed=5, **kwargs)
    rows = np.flatnonzero(train)[:50]
    np.testing.assert_array_equal(
        first.predict_proba(data, sequences, rows), second.predict_proba(data, sequences, rows)
    )
    # Corrupt everything belonging to held-out athletes: the fitted model must not change.
    held_out = ~train
    corrupted = NeuralData(
        growth=np.where(held_out[:, None], 1e3, data.growth),
        tabular=np.where(held_out[:, None], -1e3, data.tabular),
        session_index=data.session_index,
        outcome=np.where(held_out, 1.0, data.outcome),
        athlete=data.athlete,
    )
    trials = sequences.trials.copy()
    trials[60:] = 1e4
    corrupted_sequences = SessionSequences(
        sequences.session_ids, sequences.athlete_ids, trials, sequences.mask, sequences.channels
    )
    third = fit_neural_hazard(corrupted, corrupted_sequences, train, seed=5, **kwargs)
    np.testing.assert_array_equal(first.growth_scaler.mean, third.growth_scaler.mean)
    np.testing.assert_array_equal(first.channel_scaler.mean, third.channel_scaler.mean)
    np.testing.assert_array_equal(
        first.predict_proba(data, sequences, rows), third.predict_proba(data, sequences, rows)
    )
    different = fit_neural_hazard(data, sequences, train, seed=6, **kwargs)
    assert not np.array_equal(
        first.predict_proba(data, sequences, rows), different.predict_proba(data, sequences, rows)
    )


def test_unscreened_rows_and_single_class_training(
    encoder_config: dict[str, Any], train_config: dict[str, Any]
) -> None:
    data, sequences = _toy(seed=6)
    never = NeuralData(
        data.growth, data.tabular, np.full_like(data.session_index, -1), data.outcome, data.athlete
    )
    quick = {**train_config, "ensemble_size": 1, "max_epochs": 2}
    fitted = fit_neural_hazard(
        never,
        sequences,
        _train_mask(never),
        mode="none",
        sides=SIDES,
        encoder_config=encoder_config,
        config=quick,
        seed=1,
    )
    assert np.isfinite(fitted.predict_proba(never, sequences)).all()
    no_events = NeuralData(
        data.growth, data.tabular, data.session_index, np.zeros_like(data.outcome), data.athlete
    )
    with pytest.raises(ValueError, match="single outcome class"):
        fit_neural_hazard(
            no_events,
            sequences,
            _train_mask(data),
            mode="none",
            sides=SIDES,
            encoder_config=encoder_config,
            config=quick,
            seed=1,
        )


# ----------------------------------------------------------------------------- uncertainty


def test_risk_distribution_summaries() -> None:
    samples = np.array([[0.1, 0.5], [0.3, 0.5], [0.2, 0.5]])
    distribution = uncertainty.RiskDistribution(samples)
    np.testing.assert_allclose(distribution.mean, [0.2, 0.5])
    np.testing.assert_allclose(distribution.sd, [0.1, 0.0])
    low, high = distribution.interval(0.90)
    np.testing.assert_allclose(low, [0.11, 0.5])
    np.testing.assert_allclose(high, [0.29, 0.5])
    assert uncertainty.RiskDistribution(samples[:1]).sd.tolist() == [0.0, 0.0]


def test_growth_uncertainty_becomes_risk_uncertainty() -> None:
    """N5: no growth uncertainty -> no spread; more growth uncertainty -> wider risk interval."""
    predict = lambda g: 1.0 / (1.0 + np.exp(-(2.0 * g[:, 0] - 1.0)))  # noqa: E731
    rng = np.random.default_rng(0)
    centre = np.array([[0.0], [1.0], [-1.0]])
    certain = uncertainty.propagate_growth_draws(predict, np.repeat(centre[None], 200, axis=0))
    np.testing.assert_allclose(certain.sd, 0.0, atol=1e-12)
    np.testing.assert_allclose(certain.mean, predict(centre))
    widths = []
    for spread in (0.2, 0.8):
        draws = centre[None] + spread * rng.standard_normal((4000, 3, 1))
        distribution = uncertainty.propagate_growth_draws(predict, draws)
        low, high = distribution.interval(0.90)
        widths.append(high - low)
        # The interval is the image of the growth interval under the (monotone) model.
        expected = predict(centre + 1.645 * spread) - predict(centre - 1.645 * spread)
        np.testing.assert_allclose(high - low, expected, rtol=0.08)
    assert np.all(widths[1] > widths[0])
    # Propagated mean differs from plugging in the mean growth state (non-linear model).
    assert abs(distribution.mean[1] - predict(centre)[1]) > 0.02


def test_variance_decomposition() -> None:
    # Two ensemble members that differ by a constant 0.1; growth draws shift both by +-0.2.
    members = lambda g: np.stack([0.3 + g[:, 0], 0.4 + g[:, 0]])  # noqa: E731
    draws = np.array([[[-0.2]], [[0.2]]])
    parts = uncertainty.decompose(members, draws)
    assert parts["growth"][0] == pytest.approx(0.04)  # variance of {0.15, 0.55}
    assert parts["model"][0] == pytest.approx(0.0025)  # variance of {0.3, 0.4} + shift
    assert parts["total"][0] == pytest.approx(0.0425)
    combined = uncertainty.propagate_growth_draws(members, draws)
    assert combined.samples.shape == (4, 1)  # draws x members


def test_neural_model_propagates_growth_draws(
    encoder_config: dict[str, Any], train_config: dict[str, Any]
) -> None:
    data, sequences = _toy(seed=7)
    train = _train_mask(data)
    rows = np.flatnonzero(~train)[:40]
    rng = np.random.default_rng(0)
    draws = data.growth[rows][None] + 0.7 * rng.standard_normal((25, rows.size, 2))
    spreads = {}
    for mode in ("none", "film"):
        fitted = fit_neural_hazard(
            data,
            sequences,
            train,
            mode=mode,
            sides=SIDES,
            encoder_config=encoder_config,
            config={**train_config, "ensemble_size": 1},
            seed=3,
        )
        distribution = uncertainty.propagate_growth_draws(
            lambda g, _f=fitted: _f.predict_members(data, sequences, rows, growth=g), draws
        )
        spreads[mode] = distribution.sd
    assert np.allclose(spreads["none"], 0.0)  # that model never sees the growth state
    assert spreads["film"].mean() > 0.02


# ----------------------------------------------------------------------------- data assembly


def _posterior(athletes: list[str], seed: int = 0) -> SitarPosterior:
    knots = (-4.0, -2.0, 0.0, 2.0, 4.0)
    z = np.linspace(-4, 4, 200)
    design = np.column_stack([z, *splines.basis(z, knots)])
    solution, *_ = np.linalg.lstsq(design, 5 * z + 6 * np.tanh(z), rcond=None)
    n, draws = len(athletes), 40
    ones = np.ones(draws)
    rng = np.random.default_rng(seed)
    return SitarPosterior(
        athlete_ids=tuple(athletes),
        knots=knots,
        intercept=150 * ones,
        linear=solution[0] * ones,
        coefs=np.tile(solution[1:], (draws, 1)),
        alpha=np.zeros((draws, n)),
        beta=13.0 + rng.normal(0, 0.4, (draws, n)),
        gamma=np.zeros((draws, n)),
        sigma=0.2 * ones,
        sigma_alpha=5 * ones,
        sigma_gamma=0 * ones,
        delta=np.zeros((draws, n)),
        sigma_delta=0 * ones,
        timing_offset=0 * ones,
        sigma_beta=ones,
        cohort_mean=np.nan * ones,
        cohort_sd=np.nan * ones,
        mean_age=np.full(n, 13.0),
        sex="male",
        synthetic=True,
    )


def test_growth_draw_arrays_replace_means_and_keep_sds() -> None:
    frame = pd.DataFrame(
        {
            "athlete_id": ["a", "a", "b", "c"],
            "sex": ["male", "male", "male", "female"],
            "age": [12.0, 12.5, 14.0, 13.0],
            "twin_maturity_offset_mean": [-1.0, -0.5, 1.0, 0.3],
            "twin_maturity_offset_sd": [0.4, 0.4, 0.4, 0.9],
            "hours_total": [5.0, 6.0, 7.0, 8.0],
        }
    )
    columns = ["twin_maturity_offset_mean", "twin_maturity_offset_sd"]
    posterior = _posterior(["a", "b"])
    draws = dataset.growth_draw_arrays(frame, {"male": posterior}, columns, n_draws=30, seed=1)
    assert draws.shape == (30, 4, 2)
    # SD columns are unchanged in every draw; the mean column varies for athletes with a posterior.
    np.testing.assert_array_equal(
        draws[:, :, 1], np.tile(frame["twin_maturity_offset_sd"], (30, 1))
    )
    assert draws[:, 0, 0].std() == pytest.approx(0.4, rel=0.4)
    # One posterior draw gives the athlete's whole trajectory: offsets 0.5 years apart stay so.
    np.testing.assert_allclose(draws[:, 1, 0] - draws[:, 0, 0], 0.5)
    # Athlete "c" has no posterior (other sex): table value in every draw.
    np.testing.assert_array_equal(draws[:, 3, 0], np.full(30, 0.3))
    untouched = dataset.growth_draw_arrays(
        frame, {"male": posterior}, ["hours_total"], n_draws=5, seed=1
    )
    np.testing.assert_array_equal(untouched[:, :, 0], np.tile(frame["hours_total"], (5, 1)))


@pytest.fixture(scope="module")
def small_study() -> dict[str, Any]:
    """A small synthetic cohort taken through the real pose pipeline."""
    cohort_config = load_config("sim/cohort.yaml")
    cohort_config.update(n_athletes=50, seed=11, follow_up_weeks=30)
    cohort = generate_cohort(cohort_config)
    sequences, excluded = dataset.cohort_sequences(
        cohort,
        cohort_config,
        filter_config=load_config("pose/filter.yaml"),
        event_config=load_config("pose/events.yaml"),
        feature_config=load_config("features/task_features.yaml"),
    )
    person_period_config = load_config("features/person_period.yaml")
    period = build_person_period(
        cohort.tables, config=person_period_config, hazard_config=load_config("models/hazard.yaml")
    )
    return {
        "cohort": cohort,
        "sequences": sequences,
        "excluded": excluded,
        "period": period,
        "person_period_config": person_period_config,
    }


def test_cohort_sequences_come_from_the_pose_pipeline(small_study: dict[str, Any]) -> None:
    sequences, cohort = small_study["sequences"], small_study["cohort"]
    sessions = cohort.tables["screening_sessions"]
    assert (
        len(sequences.session_ids) + small_study["excluded"]["session_id"].nunique()
        >= len(sessions) - 1
    )
    assert sequences.trials.shape[1:] == (3, 16, 101) and sequences.channels == tuple(CHANNELS)
    # The simulated link: a higher load proxy gives a stiffer landing (less peak knee flexion).
    features = cohort.tables["task_features"]
    load = features[features["feature"] == "knee_extensor_load_proxy"].set_index("session_id")[
        "value"
    ]
    peak = sequences.trials[:, :, CHANNELS.index("left_knee_flexion"), :].max(axis=2).mean(axis=1)
    correlation = np.corrcoef(load.loc[list(sequences.session_ids)].to_numpy(), peak)[0, 1]
    assert correlation < -0.8
    real = cohort.__class__(cohort.tables, cohort.truth, {**cohort.metadata, "synthetic": False})
    with pytest.raises(ValueError, match="synthetic cohorts only"):
        dataset.cohort_sequences(real, {}, filter_config={}, event_config={}, feature_config={})


def test_neural_data_assembly_and_leakage_guard(small_study: dict[str, Any]) -> None:
    frame, sequences = small_study["period"].frame, small_study["sequences"]
    config = load_config("models/neural.yaml")
    growth = dataset.growth_columns(frame, config)
    assert growth == config["growth_columns_observed"]  # no growth-twin columns in this table
    data = dataset.neural_data(
        frame,
        sequences,
        label="y_all_non_contact",
        growth=growth,
        tabular=config["tabular_columns"],
        leakage_config=small_study["person_period_config"],
    )
    assert data.growth.shape == (len(frame), 3) and data.tabular.shape == (len(frame), 4)
    screened = data.session_index >= 0
    assert screened.mean() > 0.9
    # The linked session belongs to the same athlete and is the one named in the table.
    row = int(np.flatnonzero(screened)[100])
    assert sequences.athlete_ids[data.session_index[row]] == frame["athlete_id"].iloc[row]
    assert sequences.session_ids[data.session_index[row]] == frame["screen_session_id"].iloc[row]
    with pytest.raises(ValueError, match="Leakage"):
        dataset.neural_data(
            frame,
            sequences,
            label="y_all_non_contact",
            growth=growth,
            tabular=["y_growth_related"],
            leakage_config=small_study["person_period_config"],
        )
    with_twin = frame.assign(**{c: 0.0 for c in config["growth_columns_twin"]})
    assert dataset.growth_columns(with_twin, config) == config["growth_columns_twin"]


# ----------------------------------------------------------------------------- ablation grid


def test_ablation_grid_runs_end_to_end(small_study: dict[str, Any]) -> None:
    """All nine pre-registered configurations run and report from out-of-fold predictions."""
    neural = load_config("models/neural.yaml")
    neural.update(max_epochs=2, ensemble_size=1, patience=1)
    neural["pretraining"] = {**neural["pretraining"], "epochs": 1}
    neural["uncertainty"] = {**neural["uncertainty"], "n_growth_draws": 4}
    configs = {
        "ablations": load_config("eval/ablations.yaml"),
        "baselines": load_config("models/baselines.yaml"),
        "person_period": small_study["person_period_config"],
        "neural": neural,
        "encoder": {**load_config("models/tcn_film.yaml"), "hidden": 8, "n_blocks": 1},
    }
    frame = small_study["period"].frame
    posterior = {
        sex: _posterior(sorted(frame.loc[frame["sex"] == sex, "athlete_id"].unique()), seed=k)
        for k, sex in enumerate(("female", "male"))
    }
    posterior["female"] = SitarPosterior(**{**posterior["female"].__dict__, "sex": "female"})
    twin = copy.deepcopy(frame)
    for sex, fitted in posterior.items():
        for athlete, rows in twin[twin["sex"] == sex].groupby("athlete_id").groups.items():
            for name, values in fitted.features_at(
                athlete, twin.loc[rows, "age"].to_numpy()
            ).items():
                twin.loc[rows, f"twin_{name}"] = values
    results = run_ablation(
        twin,
        small_study["sequences"],
        label="y_all_non_contact",
        configs=configs,
        posteriors=posterior,
        n_folds=2,
        seed=0,
    )
    assert list(results.columns) == RESULT_COLUMNS
    assert sorted(results["ablation_id"].unique()) == list(range(1, 10))
    assert results["run_id"].nunique() == 9 and results["run_id"].str.startswith("ablation-").all()
    assert results["synthetic"].all() and (results["n_folds"] == 2).all()
    wide = results.pivot(index="ablation_id", columns="metric", values="value")
    assert wide[["auc", "brier"]].notna().all().all()
    assert ((wide["auc"] >= 0) & (wide["auc"] <= 1)).all()
    assert (wide["observed_prevalence"] == frame["y_all_non_contact"].mean()).all()
    # Only the configuration that propagates uncertainty reports a risk SD, and it is positive.
    spread = wide["mean_risk_sd_from_uncertainty"]
    assert spread.notna().tolist() == [False] * 8 + [True] and spread.loc[9] > 0
    notes = results.drop_duplicates("ablation_id").set_index("ablation_id")["note"]
    assert "mechanistic proxies unavailable" in notes[8]
    assert (notes[[1, 2, 3, 4, 5, 6, 7]] == "").all()


# ----------------------------------------------------------------------------- known true risk


def test_true_window_risk_matches_the_generating_hazards(small_study: dict[str, Any]) -> None:
    from youthtwin.sim.truth import true_window_risk

    cohort, frame = small_study["cohort"], small_study["period"].frame
    risk = true_window_risk(cohort, frame, "y_growth_related", horizon_weeks=4)
    assert risk.shape == (len(frame),) and np.isfinite(risk).all()
    assert risk.min() > 0 and risk.max() < 1
    # One row by hand: 1 - product of (1 - weekly hazard) over the following four weeks.
    truth = cohort.truth["athlete_weeks_truth"]
    row = frame.iloc[200]
    mine = truth[truth["athlete_id"] == row["athlete_id"]].sort_values("week_index")
    week = int(
        mine.loc[pd.to_datetime(mine["week_start"]) == row["week_start"], "week_index"].iloc[0]
    )
    following = mine[mine["week_index"].between(week + 1, week + 4)]["hazard_growth_related"]
    assert len(following) == 4
    assert risk[200] == pytest.approx(1 - np.prod(1 - following.to_numpy()))
    # The union outcome is at least as likely as any of its parts.
    union = true_window_risk(cohort, frame, "y_all_non_contact", horizon_weeks=4)
    assert np.all(union >= risk - 1e-12)
    # On average the true risk matches how often events occur.
    assert union.mean() == pytest.approx(frame["y_all_non_contact"].mean(), abs=0.02)


def test_ablation_reports_agreement_with_true_risk(small_study: dict[str, Any]) -> None:
    from youthtwin.sim.truth import true_window_risk

    frame = small_study["period"].frame
    configs = {
        "ablations": {"ablations": [{"id": 1, "name": "age_only", "baseline": "age_only"}]},
        "baselines": load_config("models/baselines.yaml"),
        "person_period": small_study["person_period_config"],
        "neural": load_config("models/neural.yaml"),
        "encoder": load_config("models/tcn_film.yaml"),
    }
    truth = true_window_risk(small_study["cohort"], frame, "y_all_non_contact", 4)
    results = run_ablation(
        frame,
        small_study["sequences"],
        label="y_all_non_contact",
        configs=configs,
        n_folds=2,
        seed=0,
        true_risk=truth,
    )
    values = results.set_index("metric")["value"]
    assert {"mae_vs_true_risk", "correlation_with_true_risk"} <= set(values.index)
    assert 0 < values["mae_vs_true_risk"] < 0.1
    assert "interval_coverage_of_true_risk" not in values.index  # a baseline has no interval
