"""Sequential twin evaluation: folds, training/held-out separation, no look-ahead."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import berkeley, sequential
from youthtwin.growth import sitar_bayes, splines
from youthtwin.growth.sitar_bayes import SharedDraws
from youthtwin.sim.growth_sim import pb1_height, pb1_peak_offset
from youthtwin.utils.config import load_config

CONFIG = load_config("external/sequential.yaml")
WINDOW = load_config("external/berkeley.yaml")["window"]
AGES = np.arange(8.0, 18.01, 0.5)


@pytest.fixture(scope="module")
def model_config() -> dict[str, Any]:
    settings = load_config("growth/sitar.yaml")
    settings["timing_prior"]["distance_slope"]["enabled"] = True
    settings["marginalisation"].update(timing_points=41, intensity_points=3)
    return settings


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Berkeley-like records: 6-monthly Preece-Baines curves for 24 boys."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(24):
        aphv = rng.normal(13.7, 1.0)
        s0, s1 = 0.11 * np.exp(rng.normal(0, 0.05)), 1.15 * np.exp(rng.normal(0, 0.08))
        theta = aphv - float(pb1_peak_offset(np.array([s0]), np.array([s1]))[0])
        one = np.array([1.0])
        h1 = rng.normal(178, 6)
        height = pb1_height(AGES, h1 * one, (h1 - 12.5) * one, s0 * one, s1 * one, theta * one)
        height = height + rng.normal(0, 0.3, AGES.size)
        rows.append(
            pd.DataFrame(
                {
                    "athlete_id": f"b{k:02d}",
                    "sex": "male",
                    "age": AGES,
                    "stature": height,
                    "sitting_height": height * (0.53 - 0.004 * (AGES - 8.0)),
                    "mass": 25.0 + 3.5 * (AGES - 8.0),
                    "reference": aphv,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _shared(data: sitar_bayes.GrowthData, config: dict[str, Any], n_draws: int = 40) -> SharedDraws:
    """A cheap stand-in for a fitted curve: the least-squares curve at the priors."""
    knots = splines.check_knots(config["spline"]["knots_years"])
    curve = (
        splines.orthogonalising_transform(knots) @ sitar_bayes.initial_values(data, config)["theta"]
    )
    ones = np.ones(n_draws)
    return SharedDraws(
        intercept=(float(np.mean(data.stature)) + curve[0]) * ones,
        linear=curve[1] * ones,
        coefs=np.tile(curve[2:], (n_draws, 1)),
        sigma=0.6 * ones,
        sigma_alpha=6.0 * ones,
        sigma_gamma=0.1 * ones,
        sigma_delta=0.5 * ones,
        timing_offset=0.0 * ones,
        sigma_beta=0.9 * ones,
        cohort_mean=np.nan * ones,
        cohort_sd=np.nan * ones,
        sigma_beta_slope=0.3 * ones,
    )


def test_folds_are_balanced_and_disjoint() -> None:
    ids = [f"a{k}" for k in range(23)]
    folds = sequential.assign_folds(ids, 5, seed=1)
    assert set(folds) == set(ids)
    sizes = np.bincount(list(folds.values()))
    assert sizes.max() - sizes.min() <= 1
    assert folds == sequential.assign_folds(list(reversed(ids)), 5, seed=1)


def test_training_uses_full_records_with_seasonal_priors(
    frame: pd.DataFrame, model_config: dict[str, Any]
) -> None:
    train = [f"b{k:02d}" for k in range(16)]
    windows = berkeley.draw_windows(frame, train, WINDOW, seed=3)
    data = sequential.training_data(frame, train, "male", CONFIG, windows)
    assert set(data.athlete_ids) <= set(train)
    assert data.n_observations == len(frame[frame["athlete_id"].isin(data.athlete_ids)])
    # Seasonal priors are usually further from PHV than the record's best visit.
    full = sitar_bayes.growth_data_from_measurements(
        frame[frame["athlete_id"].isin(data.athlete_ids)], "male", synthetic=False
    )
    assert data.distance.mean() > full.distance.mean()


def test_sequences_start_at_entry_and_are_capped(frame: pd.DataFrame) -> None:
    test = [f"b{k:02d}" for k in range(16, 24)]
    sequences = sequential.entry_sequences(frame, test, CONFIG, seed=4)
    low, high = CONFIG["entry_age_range_years"]
    for visits in sequences.values():
        assert low <= visits["age"].iloc[0] <= high
        assert 2 <= len(visits) <= CONFIG["max_visits"] + 1
        assert visits["age"].is_monotonic_increasing


def test_twin_never_looks_ahead(frame: pd.DataFrame, model_config: dict[str, Any]) -> None:
    train = [f"b{k:02d}" for k in range(16)]
    test = [f"b{k:02d}" for k in range(16, 24)]
    windows = berkeley.draw_windows(frame, train, WINDOW, seed=3)
    shared = _shared(sequential.training_data(frame, train, "male", CONFIG, windows), model_config)
    sequences = sequential.entry_sequences(frame, test, CONFIG, seed=4)
    reference = frame.groupby("athlete_id")["reference"].first().to_dict()
    scores = sequential.score_sequences(
        shared, sequences, reference, "male", model_config, CONFIG, seed=5
    )
    assert set(scores["athlete_id"]) <= set(test)
    assert scores["visits"].max() <= CONFIG["max_visits"]
    # Changing the last visit of every sequence cannot change any earlier twin state.
    altered = {
        a: v.assign(stature=v["stature"].where(v.index < len(v) - 1, 999.0))
        for a, v in sequences.items()
    }
    again = sequential.score_sequences(
        shared, altered, reference, "male", model_config, CONFIG, seed=5
    )
    merged = scores.merge(again, on=["athlete_id", "visits"], suffixes=("", "_alt"))
    lengths = {a: len(v) for a, v in sequences.items()}
    earlier = merged[merged["visits"] < merged["athlete_id"].map(lengths) - 1]
    np.testing.assert_allclose(earlier["twin_aphv"], earlier["twin_aphv_alt"])
    # The twin's forecasts are sensible and its intervals contain most observations.
    forecasts = scores.dropna(subset=["twin_forecast"])
    assert (forecasts["twin_forecast"] - forecasts["observed_stature"]).abs().median() < 3.0
    summary = sequential.summarise(scores, ["visits"])
    assert list(summary["visits"]) == sorted(summary["visits"])
    assert {"twin_mae", "moore_mae", "forecast_coverage", "naive_mae"} <= set(summary.columns)
    # Decision output: a probability, and the truth label derived only from the reference.
    assert scores["twin_p_circa"].between(0, 1).all()
    expected = (scores["age"] - scores["reference_aphv"]).abs() <= CONFIG[
        "circa_phv_half_width_years"
    ]
    assert (scores["circa_now"].astype(bool) == expected).all()
    overall = sequential.summarise(scores.assign(all="all"), ["all"])
    assert 0 <= overall["circa_auc_twin"].iloc[0] <= 1


def test_conformal_intervals_reach_their_level_and_never_self_calibrate() -> None:
    """Posterior SDs that are too small by half are rescaled to the target coverage."""
    rng = np.random.default_rng(7)
    n = 4000
    sd = rng.uniform(0.3, 1.0, n)
    truth = rng.normal(13.0, 1.0, n)
    scores = pd.DataFrame(
        {
            "fold": rng.integers(0, 5, n),
            "visits": rng.integers(1, 4, n),
            "reference_aphv": truth,
            "twin_aphv": truth + rng.normal(0.0, 2.0 * sd),  # true error SD is twice the claimed
            "twin_sd": sd,
        }
    )
    out = sequential.conformalize(scores, 0.9)
    assert out["twin_conf_covered"].mean() == pytest.approx(0.9, abs=0.02)
    naive = (scores["reference_aphv"] - scores["twin_aphv"]).abs() <= 1.645 * scores["twin_sd"]
    assert naive.mean() < 0.7  # the uncalibrated interval undercovers badly
    # A row's own error cannot affect its factor: corrupting fold 0 leaves fold 0's factors alone.
    corrupted = scores.copy()
    corrupted.loc[corrupted["fold"] == 0, "twin_aphv"] += 50.0
    again = sequential.conformalize(corrupted, 0.9)
    fold0 = scores["fold"] == 0
    np.testing.assert_allclose(
        again.loc[fold0, "twin_conf_factor"], out.loc[fold0, "twin_conf_factor"]
    )


def test_robustness_scenarios_degrade_only_the_held_out_data(frame: pd.DataFrame) -> None:
    test = [f"b{k:02d}" for k in range(16, 24)]
    base = sequential.entry_sequences(frame, test, CONFIG, seed=4)
    yearly = sequential.entry_sequences(
        frame, test, {**CONFIG, "scenario": {"visit_step": 2}}, seed=4
    )
    for athlete, visits in yearly.items():
        assert np.allclose(np.diff(visits["age"]), 1.0)
        assert visits["age"].iloc[0] == base[athlete]["age"].iloc[0]  # same entry age
    noisy = sequential.entry_sequences(
        frame, test, {**CONFIG, "scenario": {"added_noise_cm": 1.0}}, seed=4
    )
    differences = np.concatenate(
        [(noisy[a]["stature"] - base[a]["stature"]).to_numpy() for a in base]
    )
    assert 0.6 < differences.std() < 1.4


def test_without_sitting_height_the_prior_uses_stature_only() -> None:
    ages = np.arange(11.0, 13.01, 0.5)
    frame = pd.DataFrame(
        {"athlete_id": "c", "age": ages, "stature": 140 + 6 * (ages - 11), "sitting_height": np.nan}
    )
    data = sitar_bayes.growth_data_from_measurements(
        frame, "male", synthetic=False, stature_only_prior=True
    )
    assert np.isfinite(data.prior_aphv).all() and data.leg is None
