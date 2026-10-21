"""Synthetic injury process: risk-set rules and the known effect directions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.io.schema import NON_CONTACT_GROUPS
from youthtwin.sim import injury_sim, movement_sim
from youthtwin.sim.cohort import DAYS_PER_WEEK, SyntheticCohort
from youthtwin.utils.seeding import make_rng


def _hazards(config: dict[str, Any], **overrides: Any) -> dict[str, np.ndarray]:
    base: dict[str, Any] = {
        "leg_length_velocity": np.array([3.0]),
        "load_proxy": np.array([0.0]),
        "maturity_offset": np.array([0.0]),
        "hours": np.array([8.0]),
        "spike": np.array([False]),
    }
    base.update(overrides)
    return injury_sim.weekly_hazards(config=config["injury"], **base)


def test_reference_hazard_is_sigmoid_of_intercept(cohort_config: dict[str, Any]) -> None:
    """At the standardisation centres with zero load, only the intercept remains."""
    intercept = cohort_config["injury"]["growth_related"]["intercept"]
    hazard = _hazards(cohort_config)["growth_related"][0]
    assert hazard == pytest.approx(1.0 / (1.0 + np.exp(-intercept)))


def test_known_effect_directions(cohort_config: dict[str, Any]) -> None:
    reference = _hazards(cohort_config)
    fast_growth = _hazards(cohort_config, leg_length_velocity=np.array([6.0]))
    high_load = _hazards(cohort_config, load_proxy=np.array([1.5]))
    both = _hazards(cohort_config, leg_length_velocity=np.array([6.0]), load_proxy=np.array([1.5]))
    post_phv = _hazards(cohort_config, maturity_offset=np.array([2.0]))
    pre_phv = _hazards(cohort_config, maturity_offset=np.array([-2.0]))
    spike = _hazards(cohort_config, spike=np.array([True]))
    growth, soft = "growth_related", "non_contact_soft_tissue"

    assert fast_growth[growth] > reference[growth]
    assert high_load[growth] > reference[growth]

    # Interaction: the combination exceeds the sum of the separate log-odds effects.
    def logit(p: np.ndarray) -> float:
        return float(np.log(p / (1 - p))[0])

    separate = (logit(fast_growth[growth]) - logit(reference[growth])) + (
        logit(high_load[growth]) - logit(reference[growth])
    )
    assert logit(both[growth]) - logit(reference[growth]) > separate
    # Soft-tissue risk is higher after PHV and with exposure spikes; growth-related is not
    # driven by maturity offset directly.
    assert post_phv[soft] > pre_phv[soft]
    assert spike[soft] > reference[soft]
    assert post_phv[growth] == pytest.approx(pre_phv[growth])
    assert fast_growth[soft] == pytest.approx(reference[soft])


def test_no_injury_while_out_of_the_risk_set(cohort_config: dict[str, Any]) -> None:
    """With certain injury every at-risk week, events and time-loss weeks must alternate."""
    n_athletes, n_weeks = 20, 40
    certain = {name: np.full((n_athletes, n_weeks), 0.999) for name in injury_sim.OUTCOME_GROUPS}
    events = injury_sim.simulate_injuries(make_rng(7, "i"), certain, cohort_config["injury"])
    assert events.week.size > 0
    assert events.at_risk[events.athlete_index, events.week].all()
    for athlete in range(n_athletes):
        mine = events.athlete_index == athlete
        weeks, losses = events.week[mine], events.time_loss_days[mine]
        for week, loss in zip(weeks, losses, strict=True):
            excluded = np.arange(week + 1, min(n_weeks, week + 1 + int(np.ceil(loss / 7))))
            assert not events.at_risk[athlete, excluded].any()
            assert not np.isin(excluded, weeks).any()
    assert set(events.outcome_group) <= set(injury_sim.OUTCOME_GROUPS)
    assert events.time_loss_days.min() >= 1
    assert events.onset_day_in_week.min() >= 0 and events.onset_day_in_week.max() < DAYS_PER_WEEK


def test_zero_hazard_gives_no_injuries(cohort_config: dict[str, Any]) -> None:
    none = {name: np.zeros((10, 20)) for name in injury_sim.OUTCOME_GROUPS}
    events = injury_sim.simulate_injuries(make_rng(7, "i"), none, cohort_config["injury"])
    assert events.week.size == 0
    assert events.at_risk.all()


def test_exposure_rises_with_age_and_spikes(cohort_config: dict[str, Any]) -> None:
    young = injury_sim.sample_exposure(
        make_rng(9, "e"), np.full((2000, 10), 9.0), cohort_config["exposure"]
    )
    old = injury_sim.sample_exposure(
        make_rng(9, "e"), np.full((2000, 10), 17.0), cohort_config["exposure"]
    )
    assert old.hours_training.mean() > young.hours_training.mean()
    assert young.hours_training[young.spike].mean() > young.hours_training[~young.spike].mean()
    assert young.spike.mean() == pytest.approx(
        cohort_config["exposure"]["spike_probability"], abs=0.01
    )
    assert np.all(young.hours_match >= 0)


def _lag_one_correlation(hours: np.ndarray) -> float:
    """Correlation between consecutive weeks after removing each athlete's own mean."""
    centred = hours - hours.mean(axis=1, keepdims=True)
    return float(np.corrcoef(centred[:, :-1].ravel(), centred[:, 1:].ravel())[0, 1])


def test_training_load_persists_within_athlete(cohort_config: dict[str, Any]) -> None:
    """A heavy week tends to follow a heavy week, and some athletes train more than others."""
    age = np.full((3000, 52), 12.0)
    config = copy.deepcopy(cohort_config["exposure"])
    spec = config["persistence"]
    persistent = injury_sim.sample_exposure(make_rng(3, "e"), age, config).hours_training
    config["persistence"] = {"athlete_sd_log": 0.0, "week_sd_log": 0.0, "week_autocorrelation": 0.0}
    independent = injury_sim.sample_exposure(make_rng(3, "e"), age, config).hours_training

    assert abs(_lag_one_correlation(independent)) < 0.02
    assert _lag_one_correlation(persistent) > 0.1
    # Athletes differ in their usual load by about the configured amount (log scale).
    usual = np.log(persistent.mean(axis=1))
    assert usual.std() == pytest.approx(spec["athlete_sd_log"], rel=0.25)
    assert np.log(independent.mean(axis=1)).std() < 0.1
    # Persistence redistributes load; it does not change the average.
    assert persistent.mean() == pytest.approx(independent.mean(), rel=0.03)
    # The weekly deviation is stationary: no drift in spread from first to last week.
    assert persistent[:, 0].std() == pytest.approx(persistent[:, -1].std(), rel=0.1)


def test_movement_latent_shifts_with_maturity(cohort_config: dict[str, Any]) -> None:
    config = cohort_config["movement"]
    trait = np.zeros(2)
    latent = movement_sim.latent_value(trait, np.array([-3.0, 3.0]), config)
    assert latent[1] > latent[0]
    observed = movement_sim.observe_feature(
        make_rng(2, "m"), "landing_peak_knee_flexion_deg", np.zeros(5000), np.zeros(5000), config
    )
    spec = config["features"]["landing_peak_knee_flexion_deg"]
    assert observed.mean() == pytest.approx(
        spec["mean"] + spec["scale"] * config["maturity_shift"] * 0.5, abs=0.5
    )
    traits = movement_sim.sample_traits(make_rng(2, "t"), 50, config)
    assert set(traits) == set(config["features"])
    assert movement_sim.LOAD_PROXY in traits


def test_cohort_injuries_respect_the_risk_set(cohort: SyntheticCohort) -> None:
    truth = cohort.truth["athlete_weeks_truth"]
    injuries = cohort.tables["injuries"]
    assert len(injuries) > 0
    assert set(injuries["outcome_group"]) <= set(NON_CONTACT_GROUPS)
    week_start = pd.to_datetime(injuries["onset_date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    keyed = truth.set_index(["athlete_id", "week_start"])["at_risk"]
    onset_at_risk = [
        keyed.loc[(a, w)] for a, w in zip(injuries["athlete_id"], week_start, strict=True)
    ]
    assert all(onset_at_risk)
    # No exposure is recorded during time-loss weeks.
    exposure = cohort.tables["exposure_weekly"]
    assert (exposure.loc[~truth["at_risk"].to_numpy(), "hours_training"] == 0).all()
    assert (exposure.loc[truth["at_risk"].to_numpy(), "hours_training"] > 0).mean() > 0.99


def test_cohort_recovers_the_generating_directions(cohort: SyntheticCohort) -> None:
    """Sanity check of the generator itself: observed rates follow the true hazard ordering."""
    truth = cohort.truth["athlete_weeks_truth"]
    at_risk = truth[truth["at_risk"]]
    for group in ("growth_related", "non_contact_soft_tissue"):
        hazard = at_risk[f"hazard_{group}"]
        expected = hazard.sum()
        observed = (cohort.tables["injuries"]["outcome_group"] == group).sum()
        # Poisson-ish tolerance around the expected count from the true hazards.
        assert abs(observed - expected) < 4 * np.sqrt(expected)


def test_realistic_config_matches_the_published_incidence() -> None:
    """Without test overrides, about 3.8% of athletes have a growth-related injury per season."""
    import importlib.util

    from youthtwin.sim.cohort import generate_cohort
    from youthtwin.utils.config import load_config, repo_root

    spec = importlib.util.spec_from_file_location(
        "calibrate", repo_root() / "scripts" / "calibrate_simulation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = load_config("sim/cohort.yaml")
    target = load_config("sim/calibration.yaml")["target_proportion"]
    config["n_athletes"] = 1500
    weekly = generate_cohort(config).truth["athlete_weeks_truth"]
    expected = module.expected_proportion(weekly, "hazard_growth_related", 0.0)
    assert expected == pytest.approx(target, abs=0.006)
