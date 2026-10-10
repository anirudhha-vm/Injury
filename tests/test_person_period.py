"""Athlete-week table, exposure features and mechanistic load proxies."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from youthtwin.features import exposure, mech_load
from youthtwin.features.person_period import PersonPeriod, build_person_period
from youthtwin.growth import splines
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.utils.config import load_config

START = date(2026, 1, 5)


def _tables(injuries: list[dict[str, Any]], n_weeks: int = 20) -> dict[str, pd.DataFrame]:
    """One athlete, weekly exposure, visits every 6 weeks, chosen injuries."""
    athlete = str(uuid4())
    weeks = [START + timedelta(weeks=k) for k in range(n_weeks)]
    visits = [START + timedelta(weeks=k, days=2) for k in range(0, n_weeks, 6)]
    session_ids = [str(uuid4()) for _ in visits]
    return {
        "athletes": pd.DataFrame(
            [
                {
                    "id": athlete,
                    "sex": "male",
                    "dob_month": "2013-01",
                    "sport": "football",
                    "academy_id": "academy_1",
                    "dominant_limb": "right",
                    "synthetic": True,
                }
            ]
        ),
        "anthropometrics": pd.DataFrame(
            [
                {
                    "athlete_id": athlete,
                    "date": day,
                    "stature_1": 150.0 + 0.9 * k,
                    "stature_2": 150.0 + 0.9 * k,
                    "sitting_height_1": 78.0 + 0.3 * k,
                    "sitting_height_2": 78.0 + 0.3 * k,
                    "mass_1": 40.0 + 0.5 * k,
                    "mass_2": 40.0 + 0.5 * k,
                    "measurer_id": "m",
                    "synthetic": True,
                }
                for k, day in enumerate(visits)
            ]
        ),
        "screening_sessions": pd.DataFrame(
            {
                "id": session_ids,
                "athlete_id": athlete,
                "date": visits,
                "camera_setup": "x",
                "synthetic": True,
            }
        ),
        "task_features": pd.DataFrame(
            [
                {
                    "athlete_id": athlete,
                    "session_id": session,
                    "task": "drop_vertical_jump",
                    "trial": trial,
                    "limb": "both",
                    "feature": "knee_flexion_max",
                    "value": 80.0 + k + (0 if trial is None else 50),
                    "quality": 1.0,
                    "synthetic": True,
                }
                for k, session in enumerate(session_ids)
                for trial in (None, 1)
            ]
        ),
        "exposure_weekly": pd.DataFrame(
            {
                "athlete_id": athlete,
                "week_start": weeks,
                "hours_training": 6.0,
                "hours_match": 2.0,
                "sessions": 4,
                "synthetic": True,
            }
        ),
        "injuries": pd.DataFrame(
            [
                {
                    "id": str(uuid4()),
                    "athlete_id": athlete,
                    "osiics_code": "X",
                    "body_area": "knee",
                    "tissue": "t",
                    "mechanism": "non_contact",
                    "diagnosed_by": "physio",
                    "synthetic": True,
                    **injury,
                }
                for injury in injuries
            ],
            columns=[
                "id",
                "athlete_id",
                "onset_date",
                "osiics_code",
                "body_area",
                "tissue",
                "mechanism",
                "time_loss_days",
                "outcome_group",
                "diagnosed_by",
                "synthetic",
            ],
        ),
    }


def _build(tables: dict[str, pd.DataFrame], **kwargs: Any) -> PersonPeriod:
    return build_person_period(
        tables,
        config=load_config("features/person_period.yaml"),
        hazard_config=load_config("models/hazard.yaml"),
        **kwargs,
    )


def _week(frame: pd.DataFrame, index: int) -> pd.Series:
    rows = frame[frame["week_start"] == pd.Timestamp(START + timedelta(weeks=index))]
    assert len(rows) == 1, f"week {index} not in the risk set"
    return rows.iloc[0]


# ----------------------------------------------------------------------------- outcome window


def test_outcome_window_is_the_four_weeks_after_the_prediction_time() -> None:
    # Injury on Wednesday of week 10. Prediction time of week w = end of week w.
    onset = START + timedelta(weeks=10, days=2)
    tables = _tables(
        [{"onset_date": onset, "time_loss_days": 3, "outcome_group": "growth_related"}]
    )
    frame = _build(tables).frame
    positive = frame.loc[frame["y_growth_related"] == 1, "week_start"]
    expected = [pd.Timestamp(START + timedelta(weeks=k)) for k in (6, 7, 8, 9)]
    assert list(positive) == expected  # weeks whose following four weeks contain the onset
    assert (frame["y_non_contact_soft_tissue"] == 0).all()
    assert frame["y_all_non_contact"].equals(frame["y_growth_related"])
    assert _week(frame, 9)["n_prior_injuries"] == 0  # not yet happened at that prediction time
    assert _week(frame, 11)["n_prior_injuries"] == 1
    assert _week(frame, 11)["prior_growth_related_injury"] == 1.0
    assert _week(frame, 12)["weeks_since_last_injury"] == pytest.approx(2 + 5 / 7)


def test_contact_injuries_are_history_but_not_a_non_contact_outcome() -> None:
    onset = START + timedelta(weeks=10, days=2)
    tables = _tables([{"onset_date": onset, "time_loss_days": 0, "outcome_group": "contact"}])
    frame = _build(tables).frame
    assert (
        frame[["y_growth_related", "y_non_contact_soft_tissue", "y_all_non_contact"]].sum().sum()
        == 0
    )
    assert _week(frame, 12)["n_prior_injuries"] == 1


def test_time_loss_weeks_leave_the_risk_set() -> None:
    onset = START + timedelta(weeks=8, days=1)  # Tuesday of week 8, out for 20 days
    tables = _tables(
        [{"onset_date": onset, "time_loss_days": 20, "outcome_group": "non_contact_soft_tissue"}]
    )
    result = _build(tables)
    present = set(result.frame["week_start"])
    week = lambda k: pd.Timestamp(START + timedelta(weeks=k))  # noqa: E731
    # Injured at the prediction times of weeks 8, 9 and 10 (recovery on day 21 after onset).
    assert not {week(8), week(9), week(10)} & present
    assert {week(7), week(11)} <= present
    assert result.summary["n_excluded_time_loss"] == 3


def test_weeks_with_unobservable_outcome_are_excluded() -> None:
    result = _build(_tables([], n_weeks=20))
    last = result.frame["week_start"].max()
    # The window after week 15 ends exactly at the end of week 19, the last observed week.
    assert last == pd.Timestamp(START + timedelta(weeks=15))
    assert result.summary["n_excluded_unobservable_outcome"] == 4
    assert result.summary["n_athlete_weeks_at_risk"] == 16
    assert result.horizon_weeks == 4 and result.summary["synthetic"] is True


# ----------------------------------------------------------------------------- predictors


def test_latest_visit_and_screen_are_carried_forward() -> None:
    frame = _build(_tables([])).frame
    # Visits are on Wednesday of weeks 0, 6, 12, 18. At the end of week 5 the latest is visit 0.
    assert _week(frame, 5)["stature"] == 150.0 and _week(frame, 6)["stature"] == 150.9
    assert _week(frame, 5)["weeks_since_screen"] == pytest.approx(5 + 5 / 7)
    assert _week(frame, 6)["weeks_since_screen"] == pytest.approx(5 / 7)
    column = "scr__drop_vertical_jump__both__knee_flexion_max"
    # Session-level value (trial empty) is used, not the trial-level row.
    assert _week(frame, 5)[column] == 80.0 and _week(frame, 7)[column] == 81.0
    assert _week(frame, 5)["leg_length"] == pytest.approx(72.0)


def test_observed_velocity_needs_enough_time_between_visits() -> None:
    frame = _build(_tables([])).frame
    assert np.isnan(_week(frame, 6)["obs_height_velocity"])  # two visits only 42 days apart
    # From the third visit (span 84 days): 0.9 cm per 6 weeks = 0.9 / 42 * 365.25 cm/year.
    assert _week(frame, 12)["obs_height_velocity"] == pytest.approx(0.9 / 42 * 365.25, rel=1e-6)
    assert _week(frame, 12)["obs_leg_length_velocity"] == pytest.approx(0.6 / 42 * 365.25, rel=1e-6)
    assert _week(frame, 12)["obs_mass_velocity"] == pytest.approx(0.5 / 42 * 365.25, rel=1e-6)


def test_maturity_offset_is_aged_forward_from_the_visit() -> None:
    frame = _build(_tables([])).frame
    # Between visits the offset advances one year per year: 7 days later = +7/365.25.
    step = _week(frame, 4)["moore_maturity_offset"] - _week(frame, 3)["moore_maturity_offset"]
    assert step == pytest.approx(7 / 365.25)
    # Born mid-January 2013; end of week 3 is 2 February 2026: 13 years and 18 days.
    assert _week(frame, 3)["age"] == pytest.approx(13 + 18 / 365.25, abs=0.005)


def test_growth_twin_features_are_added_when_posteriors_are_given() -> None:
    tables = _tables([])
    athlete = tables["athletes"]["id"].iloc[0]
    knots = (-4.0, -2.0, 0.0, 2.0, 4.0)
    z = np.linspace(-4, 4, 200)
    design = np.column_stack([z, *splines.basis(z, knots)])
    solution, *_ = np.linalg.lstsq(design, 5 * z + 6 * np.tanh(z), rcond=None)
    ones = np.ones(30)
    posterior = SitarPosterior(
        athlete_ids=(athlete,),
        knots=knots,
        intercept=150 * ones,
        linear=solution[0] * ones,
        coefs=np.tile(solution[1:], (30, 1)),
        alpha=np.zeros((30, 1)),
        beta=13.5 + np.random.default_rng(0).normal(0, 0.3, (30, 1)),
        gamma=np.zeros((30, 1)),
        sigma=0.2 * ones,
        sigma_alpha=5 * ones,
        sigma_gamma=0 * ones,
        delta=np.zeros((30, 1)),
        sigma_delta=0 * ones,
        timing_offset=0 * ones,
        sigma_beta=ones,
        mean_age=np.array([13.2]),
        sex="male",
        synthetic=True,
    )
    frame = _build(tables, posteriors={"male": posterior}).frame
    twin = [c for c in frame.columns if c.startswith("twin_")]
    assert {
        "twin_aphv_mean",
        "twin_aphv_sd",
        "twin_maturity_offset_mean",
        "twin_height_velocity_sd",
        "twin_pct_peak_velocity_mean",
    } <= set(twin)
    row = _week(frame, 3)
    state = posterior.growth_state(athlete, float(row["age"]))
    assert row["twin_aphv_mean"] == pytest.approx(state.aphv.mean)
    assert row["twin_maturity_offset_sd"] == pytest.approx(state.maturity_offset.sd)
    assert row["twin_height_velocity_mean"] == pytest.approx(state.height_velocity.mean)
    assert row["twin_pct_peak_velocity_mean"] == pytest.approx(state.pct_peak_velocity.mean)
    assert 0.2 < row["twin_aphv_sd"] < 0.45  # the uncertainty reaches the table (N5)


def test_summary_counts_add_up(person_period: PersonPeriod) -> None:
    summary = person_period.summary
    assert summary["n_athlete_weeks_total"] == 300 * 52
    assert (
        summary["n_athlete_weeks_at_risk"]
        + summary["n_excluded_time_loss"]
        + summary["n_excluded_unobservable_outcome"]
        == summary["n_athlete_weeks_total"]
    )
    frame = person_period.frame
    assert not frame.duplicated(["athlete_id", "week_start"]).any()
    assert (
        0 < summary["prevalence"]["y_growth_related"] < summary["prevalence"]["y_all_non_contact"]
    )
    assert (frame["y_all_non_contact"] >= frame["y_growth_related"]).all()


# ----------------------------------------------------------------------------- exposure


def test_ewma_reference_values() -> None:
    # half-life 1 week -> decay 0.5: 4, 0.5*8 + 0.5*4 = 6, 0.5*0 + 0.5*6 = 3
    np.testing.assert_allclose(exposure.ewma(np.array([4.0, 8.0, 0.0]), 1.0), [4.0, 6.0, 3.0])
    # A missing week carries the previous value forward.
    np.testing.assert_allclose(exposure.ewma(np.array([4.0, np.nan, 8.0]), 1.0), [4.0, 4.0, 6.0])
    constant = exposure.ewma(np.full(30, 7.0), 3.0)
    np.testing.assert_allclose(constant, 7.0)
    with pytest.raises(ValueError, match="positive"):
        exposure.ewma(np.ones(3), 0.0)


def test_exposure_features_use_only_past_and_present_weeks() -> None:
    weeks = [START + timedelta(weeks=k) for k in range(5)]
    table = pd.DataFrame(
        {
            "athlete_id": "a",
            "week_start": weeks,
            "hours_training": [4.0, 4.0, 10.0, 4.0, 4.0],
            "hours_match": [0.0, 0.0, 2.0, 0.0, 0.0],
            "sessions": [3, 3, 6, 3, 3],
        }
    )
    features = exposure.exposure_features(table, {"ewma_half_life_weeks": 1.0})
    assert features["hours_total"].tolist() == [4.0, 4.0, 12.0, 4.0, 4.0]
    assert features["ewma_hours"].tolist() == [4.0, 4.0, 8.0, 6.0, 5.0]
    assert np.isnan(features["hours_change"].iloc[0])
    assert features["hours_change"].tolist()[1:] == [0.0, 8.0, -4.0, -2.0]  # spike in week 2
    changed = table.copy()
    changed.loc[4, "hours_training"] = 40.0  # the future
    again = exposure.exposure_features(changed, {"ewma_half_life_weeks": 1.0})
    pd.testing.assert_frame_equal(features.iloc[:4], again.iloc[:4])


# ----------------------------------------------------------------------------- load proxies


def test_mech_proxy_reference_values() -> None:
    # 40 kg, thigh 0.40 m inclined 30 deg: 40 * 9.80665 * 0.40 * 0.5 = 78.4532 N m
    moment = mech_load.knee_extensor_moment_proxy(40.0, 0.40, 30.0)
    assert float(moment) == pytest.approx(78.4532)
    assert float(mech_load.patellar_tendon_load_proxy(moment, 0.35)) == pytest.approx(
        224.152, abs=1e-3
    )
    # 40 kg, forefoot lever 2.5 x heel lever: 40 * 9.80665 * 2.5 = 980.665 N
    assert float(mech_load.achilles_load_proxy(40.0, 2.5)) == pytest.approx(980.665)
    assert float(mech_load.knee_extensor_moment_proxy(40.0, 0.40, 0.0)) == pytest.approx(0.0)


def test_mech_proxies_scale_with_growth_as_stated() -> None:
    base = {
        "mass_kg": 40.0,
        "leg_length_cm": 75.0,
        "thigh_length_fraction": 0.52,
        "thigh_inclination_deg": 35.0,
        "forefoot_heel_lever_ratio": 2.4,
    }
    reference = mech_load.mech_proxies(**base)
    assert set(reference) == {
        "mech_knee_extensor_moment_proxy",
        "mech_patellar_tendon_load_proxy",
        "mech_achilles_load_proxy",
        "mech_patellar_tendon_load_proxy_bw",
        "mech_achilles_load_proxy_bw",
    }
    heavier = mech_load.mech_proxies(**{**base, "mass_kg": 50.0})
    assert float(heavier["mech_patellar_tendon_load_proxy"]) == pytest.approx(
        1.25 * float(reference["mech_patellar_tendon_load_proxy"])
    )
    # Body-weight-normalised proxies do not change with mass ...
    assert float(heavier["mech_patellar_tendon_load_proxy_bw"]) == pytest.approx(
        float(reference["mech_patellar_tendon_load_proxy_bw"])
    )
    # ... nor with overall leg length (only proportions and posture matter):
    longer = mech_load.mech_proxies(**{**base, "leg_length_cm": 90.0})
    assert float(longer["mech_patellar_tendon_load_proxy_bw"]) == pytest.approx(
        float(reference["mech_patellar_tendon_load_proxy_bw"])
    )
    # thigh/shank * sin(35 deg) = (0.52 / 0.48) * 0.573576 = 0.621374
    assert float(reference["mech_patellar_tendon_load_proxy_bw"]) == pytest.approx(
        0.621374, abs=1e-6
    )
    relatively_longer_thigh = mech_load.mech_proxies(**{**base, "thigh_length_fraction": 0.56})
    assert float(relatively_longer_thigh["mech_patellar_tendon_load_proxy_bw"]) > 0.621374


def test_mech_proxies_are_added_only_when_their_sources_exist(person_period: PersonPeriod) -> None:
    config = load_config("features/mech.yaml")
    assert config["validated_against_force_plates"] is False
    unchanged = mech_load.add_mech_proxies(person_period.frame, config)
    assert not [c for c in unchanged.columns if c.startswith("mech_")]  # no side-view features
    sources = config["sources"]
    frame = person_period.frame.head(50).assign(
        **{
            sources["thigh_inclination"]: 35.0,
            sources["thigh_length_fraction"]: 0.52,
            sources["forefoot_heel_lever_ratio"]: 2.4,
        }
    )
    with_proxies = mech_load.add_mech_proxies(frame, config)
    assert len([c for c in with_proxies.columns if c.startswith("mech_")]) == 5
    valid = with_proxies["mass"].notna()
    expected = with_proxies.loc[valid, "mass"] * 9.80665 * 2.4
    np.testing.assert_allclose(with_proxies.loc[valid, "mech_achilles_load_proxy"], expected)
