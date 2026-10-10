"""Athlete-week (person-period) table for discrete-time hazard models.

One row per athlete per calendar week in which the athlete is **at risk**. For week ``w`` the
prediction time ``t0`` is the end of that week.

- **Predictors** use only information dated before ``t0``: age, sex, sport; the latest
  anthropometric visit (size, Moore maturity offset aged forward to ``t0``, finite-difference
  growth velocities); the growth twin's posterior means and SDs at ``t0`` if posteriors are
  supplied; the latest screening session's features with ``weeks_since_screen``; exposure up to
  week ``w``; and injury history before ``t0``.
- **Outcomes** (one column per outcome group): does an injury of that type have its onset in the
  ``horizon`` weeks after ``t0``? ``all_non_contact`` is the union of the non-contact groups.
- **Risk set**: a week is excluded while the athlete is in time-loss from an earlier injury at
  ``t0``, and when the outcome window runs past the end of follow-up (outcome not observable).

Outcome labels come only from the ``injuries`` table, i.e. physiotherapist diagnoses
(non-negotiable N1). Injury history is a predictor only through injuries with onset before
``t0``. ``assert_no_leakage`` refuses label-defining or truth columns as model inputs (N2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.features.exposure import exposure_features
from youthtwin.growth.growth_state import DAYS_PER_YEAR, SitarPosterior
from youthtwin.growth.maturity_equations import Sex, moore2015
from youthtwin.io.schema import NON_CONTACT_GROUPS

DAYS_PER_WEEK = 7
ID_COLUMNS = ["athlete_id", "week_start", "academy_id", "screen_session_id"]
CATEGORICAL = ["sex", "sport"]
ALL_NON_CONTACT = "all_non_contact"
LABEL_PREFIX = "y_"
TWIN_PREFIX = "twin_"
SCREEN_PREFIX = "scr__"
_MID_MONTH = "-15"
_NS_PER_WEEK = 7 * 24 * 3600 * 1e9


@dataclass(frozen=True)
class PersonPeriod:
    frame: pd.DataFrame
    labels: tuple[str, ...]
    horizon_weeks: int
    summary: dict[str, Any]

    @property
    def predictors(self) -> list[str]:
        return predictor_columns(self.frame)


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Every column that may be offered to a model: not an ID, a label, or the synthetic flag."""
    skip = {*ID_COLUMNS, "synthetic"}
    return [c for c in frame.columns if c not in skip and not c.startswith(LABEL_PREFIX)]


def assert_no_leakage(columns: Sequence[str], config: Mapping[str, Any]) -> None:
    """Raise if any proposed model input is a label, a label-defining field, or simulation truth."""
    prefixes = tuple(config["forbidden_predictor_prefixes"])
    names = set(config["forbidden_predictor_names"])
    bad = sorted(c for c in columns if c.startswith(prefixes) or c in names)
    if bad:
        raise ValueError(f"Leakage: these columns may not be model inputs: {bad}")


def _timestamps(values: Any) -> Any:
    """Dates as nanosecond timestamps (one resolution everywhere, so joins on dates work)."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def _days(later: Any, earlier: Any) -> Any:
    return (_timestamps(later) - _timestamps(earlier)).dt.days.astype("float64")


def _visit_table(
    athletes: pd.DataFrame, anthropometrics: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Per visit: mean measures, Moore maturity offset, and finite-difference velocities."""
    visits = anthropometrics.merge(
        athletes[["id", "sex", "dob_month"]], left_on="athlete_id", right_on="id"
    ).sort_values(["athlete_id", "date"])
    visits["visit_date"] = _timestamps(visits["date"])
    birth = _timestamps(visits["dob_month"] + _MID_MONTH)
    visits["visit_age"] = (visits["visit_date"] - birth).dt.days / DAYS_PER_YEAR
    for measure in ("stature", "sitting_height", "mass"):
        visits[measure] = visits[[f"{measure}_1", f"{measure}_2"]].mean(axis=1)
    visits["leg_length"] = visits["stature"] - visits["sitting_height"]
    visits["moore_mo_at_visit"] = np.nan
    sexes: tuple[Sex, ...] = ("male", "female")
    for sex in sexes:
        rows = visits["sex"] == sex
        if rows.any():
            estimate = moore2015(
                sex,
                visits.loc[rows, "visit_age"].to_numpy(),
                visits.loc[rows, "stature"].to_numpy(),
                visits.loc[rows, "sitting_height"].to_numpy(),
            )
            visits.loc[rows, "moore_mo_at_visit"] = estimate.mo

    window = int(config["observed_velocity"]["n_visits"])
    min_span = float(config["observed_velocity"]["min_span_days"])
    measures = {
        "stature": "obs_height_velocity",
        "leg_length": "obs_leg_length_velocity",
        "mass": "obs_mass_velocity",
    }
    years = (visits["visit_date"] - visits["visit_date"].min()).dt.days.to_numpy() / DAYS_PER_YEAR
    velocity = {column: np.full(len(visits), np.nan) for column in measures.values()}
    for visit_rows in visits.groupby("athlete_id", sort=False).indices.values():
        for k in range(1, len(visit_rows)):
            used = visit_rows[max(0, k - window + 1) : k + 1]
            span = years[used]
            if (span[-1] - span[0]) * DAYS_PER_YEAR < min_span:
                continue
            for measure, column in measures.items():
                slope = np.polyfit(span, visits[measure].to_numpy()[used], 1)[0]
                velocity[column][visit_rows[k]] = slope  # per year
    visits = visits.assign(**velocity)
    keep = [
        "athlete_id",
        "visit_date",
        "stature",
        "mass",
        "leg_length",
        "moore_mo_at_visit",
        *measures.values(),
    ]
    return visits[keep]


def _screen_table(task_features: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Session-level screening features, one row per session, wide."""
    level = task_features[task_features["trial"].isna()]
    if level.empty:
        return pd.DataFrame(columns=["athlete_id", "screen_date"])
    names = SCREEN_PREFIX + level["task"] + "__" + level["limb"] + "__" + level["feature"]
    wide = level.assign(name=names).pivot_table(
        index=["athlete_id", "session_id"], columns="name", values="value"
    )
    wide = wide.reset_index().merge(sessions[["id", "date"]], left_on="session_id", right_on="id")
    wide["screen_date"] = _timestamps(wide["date"])
    wide = wide.rename(columns={"session_id": "screen_session_id"})
    return wide.drop(columns=["id", "date"])


def _latest_before(weeks: pd.DataFrame, events: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """For each athlete-week, the latest row of ``events`` dated strictly before ``t0``."""
    if events.empty:
        return weeks
    merged = pd.merge_asof(
        weeks.sort_values("t0"),
        events.sort_values(date_column),
        left_on="t0",
        right_on=date_column,
        by="athlete_id",
        allow_exact_matches=False,
    )
    return merged.sort_values(["athlete_id", "t0"]).reset_index(drop=True)


def build_person_period(
    tables: Mapping[str, pd.DataFrame],
    *,
    config: Mapping[str, Any],
    hazard_config: Mapping[str, Any],
    posteriors: Mapping[str, SitarPosterior] | None = None,
) -> PersonPeriod:
    """Build the athlete-week table from the observable data tables.

    ``tables`` needs ``athletes, anthropometrics, exposure_weekly, injuries`` and may contain
    ``screening_sessions`` and ``task_features``. ``posteriors`` maps sex to a fitted growth
    posterior; if given, its features are added with the ``twin_`` prefix.
    """
    horizon = int(hazard_config["prediction_horizon_weeks"])
    athletes = tables["athletes"]
    weeks = exposure_features(tables["exposure_weekly"], config["exposure"])
    weeks["week_start"] = _timestamps(weeks["week_start"])
    weeks["t0"] = weeks["week_start"] + pd.Timedelta(days=DAYS_PER_WEEK)
    study_end = weeks["t0"].max()
    weeks = weeks.merge(
        athletes[["id", "sex", "sport", "academy_id", "dob_month"]],
        left_on="athlete_id",
        right_on="id",
    ).drop(columns="id")
    weeks["age"] = (
        weeks["t0"] - _timestamps(weeks["dob_month"] + _MID_MONTH)
    ).dt.days / DAYS_PER_YEAR
    weeks = weeks.sort_values(["athlete_id", "t0"]).reset_index(drop=True)

    # --- anthropometry ---
    weeks = _latest_before(
        weeks, _visit_table(athletes, tables["anthropometrics"], config), "visit_date"
    )
    since_visit = _days(weeks["t0"], weeks["visit_date"])
    weeks["weeks_since_anthropometry"] = since_visit / DAYS_PER_WEEK
    weeks["moore_maturity_offset"] = weeks.pop("moore_mo_at_visit") + since_visit / DAYS_PER_YEAR

    # --- screening ---
    if "task_features" in tables and "screening_sessions" in tables:
        screens = _screen_table(tables["task_features"], tables["screening_sessions"])
        weeks = _latest_before(weeks, screens, "screen_date")
        if "screen_date" in weeks.columns:
            weeks["weeks_since_screen"] = _days(weeks["t0"], weeks["screen_date"]) / DAYS_PER_WEEK

    # --- injuries: history, risk set, outcomes ---
    groups = [g for g in hazard_config["outcome_groups"] if g != ALL_NON_CONTACT]
    labels = [f"{LABEL_PREFIX}{g}" for g in hazard_config["outcome_groups"]]
    injuries = tables["injuries"].copy()
    injuries["onset"] = _timestamps(injuries["onset_date"])
    injuries["recovered"] = injuries["onset"] + pd.to_timedelta(
        injuries["time_loss_days"], unit="D"
    )
    by_athlete = {str(a): g.sort_values("onset") for a, g in injuries.groupby("athlete_id")}
    n = len(weeks)
    prior = np.zeros(n)
    prior_growth = np.zeros(n, dtype=bool)
    since_last = np.full(n, np.nan)
    in_time_loss = np.zeros(n, dtype=bool)
    outcome = {g: np.zeros(n, dtype=bool) for g in (*groups, ALL_NON_CONTACT)}
    t0_all = weeks["t0"].to_numpy()
    window = np.timedelta64(horizon * DAYS_PER_WEEK, "D")
    for athlete, rows in weeks.groupby("athlete_id", sort=False).indices.items():
        history = by_athlete.get(str(athlete))
        if history is None:
            continue
        t0 = t0_all[rows][:, None]
        onset = history["onset"].to_numpy()[None, :]
        recovered = history["recovered"].to_numpy()[None, :]
        group = history["outcome_group"].to_numpy()
        before = onset < t0
        prior[rows] = before.sum(axis=1)
        prior_growth[rows] = (before & (group == "growth_related")[None, :]).any(axis=1)
        onset_ns = onset.astype("datetime64[ns]").astype("int64")
        t0_ns = t0[:, 0].astype("datetime64[ns]").astype("int64")
        latest = np.max(np.where(before, onset_ns, np.iinfo("int64").min), axis=1)
        gap_weeks = (t0_ns - latest) / _NS_PER_WEEK
        since_last[rows] = np.where(before.any(axis=1), gap_weeks, np.nan)
        in_time_loss[rows] = (before & (recovered >= t0)).any(axis=1)
        upcoming = (onset >= t0) & (onset < t0 + window)
        for g in groups:
            outcome[g][rows] = (upcoming & (group == g)[None, :]).any(axis=1)
        non_contact = np.isin(group, NON_CONTACT_GROUPS)[None, :]
        outcome[ALL_NON_CONTACT][rows] = (upcoming & non_contact).any(axis=1)
    weeks["n_prior_injuries"] = prior
    weeks["prior_growth_related_injury"] = prior_growth.astype(float)
    weeks["weeks_since_last_injury"] = since_last
    for label in labels:
        weeks[label] = outcome[label.removeprefix(LABEL_PREFIX)].astype(int)

    # --- growth twin ---
    if posteriors:
        for sex, posterior in posteriors.items():
            known = set(posterior.athlete_ids)
            part = weeks[(weeks["sex"] == sex) & weeks["athlete_id"].isin(known)]
            for athlete, rows in part.groupby("athlete_id", sort=False).groups.items():
                features = posterior.features_at(str(athlete), weeks.loc[rows, "age"].to_numpy())
                for name, values in features.items():
                    weeks.loc[rows, f"{TWIN_PREFIX}{name}"] = values

    # --- risk set and censoring ---
    censored = (weeks["t0"] + pd.Timedelta(weeks=horizon) > study_end).to_numpy()
    keep = ~in_time_loss & ~censored
    synthetic = bool(athletes["synthetic"].all())
    frame = (
        weeks.loc[keep]
        .drop(columns=["t0", "dob_month", "visit_date", "screen_date"], errors="ignore")
        .reset_index(drop=True)
    )
    frame["synthetic"] = synthetic
    summary = {
        "n_athlete_weeks_total": int(n),
        "n_excluded_time_loss": int(in_time_loss.sum()),
        "n_excluded_unobservable_outcome": int((censored & ~in_time_loss).sum()),
        "n_athlete_weeks_at_risk": len(frame),
        "n_athletes": int(frame["athlete_id"].nunique()),
        "prevalence": {label: float(frame[label].mean()) for label in labels},
        "synthetic": synthetic,
    }
    assert_no_leakage(predictor_columns(frame), config)
    return PersonPeriod(frame, tuple(labels), horizon, summary)
