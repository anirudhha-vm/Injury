"""Assemble a synthetic cohort: schema-shaped observable tables plus separate truth tables.

``tables`` look like what the real study will collect (see ``youthtwin.io.schema``) and are the
only thing the modelling pipeline may read. ``truth`` holds the generating parameters and true
hazards; it exists to score methods in the simulation study and must never be used as model
input. Every table carries ``synthetic=True``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.sim import growth_sim, injury_sim, movement_sim
from youthtwin.utils.provenance import run_metadata
from youthtwin.utils.seeding import make_rng

DAYS_PER_YEAR = 365.25
DAYS_PER_WEEK = 7
SCREENING_TASK = "drop_vertical_jump"
METADATA_FILE = "_metadata.json"
TRUTH_DIR = "truth"

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticCohort:
    tables: dict[str, pd.DataFrame]
    truth: dict[str, pd.DataFrame]
    metadata: dict[str, Any]


def _uuids(rng: np.random.Generator, n: int) -> list[str]:
    return [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(n)]


def _dates(start: date, days: NDArray[np.int64]) -> list[date]:
    return [start + timedelta(days=int(d)) for d in days]


def generate_cohort(config: Mapping[str, Any]) -> SyntheticCohort:
    """Generate the whole cohort deterministically from ``config['seed']``."""
    if config.get("synthetic") is not True:
        raise ValueError("Cohort config must set `synthetic: true`")
    seed = int(config["seed"])
    n = int(config["n_athletes"])
    n_weeks = int(config["follow_up_weeks"])
    start = date.fromisoformat(str(config["start_date"]))
    growth_cfg, movement_cfg = config["growth"], config["movement"]

    # --- athletes ---
    rng = make_rng(seed, "athletes")
    athlete_ids = _uuids(rng, n)
    is_female = rng.random(n) < config["proportion_female"]
    age_low, age_high = config["baseline_age_years"]
    baseline_age = rng.uniform(float(age_low), float(age_high), n)
    sports = rng.choice(np.asarray(config["sports"]), n)
    academies = rng.integers(1, int(config["n_academies"]) + 1, n)
    right_dominant = rng.random(n) < config["proportion_right_dominant"]

    params = growth_sim.sample_growth_params(
        make_rng(seed, "growth"), is_female, growth_cfg, entry_age=baseline_age
    )
    sex_shift = np.where(is_female, 1.0, -1.0) * growth_cfg["mid_parent_sex_adjustment_cm"]
    mid_parent = (
        params.h1 + sex_shift + rng.normal(0.0, growth_cfg["mid_parent_height_noise_sd_cm"], n)
    )
    births = [start - timedelta(days=round(a * DAYS_PER_YEAR)) for a in baseline_age]
    athletes = pd.DataFrame(
        {
            "id": athlete_ids,
            "sex": np.where(is_female, "female", "male"),
            "dob_month": [b.strftime("%Y-%m") for b in births],
            "sport": sports,
            "academy_id": [f"academy_{k}" for k in academies],
            "dominant_limb": np.where(right_dominant, "right", "left"),
            "mid_parent_height": np.round(mid_parent, 1),
        }
    )

    # --- visits: anthropometrics + screening sessions ---
    visits = growth_sim.sample_visits(
        make_rng(seed, "visits"), n, n_weeks * DAYS_PER_WEEK, config["visits"]
    )
    idx = visits.athlete_index
    visit_age = baseline_age[idx] + visits.day / DAYS_PER_YEAR
    visit_curves = growth_sim.GrowthCurves(params.at(idx), growth_cfg)
    rng = make_rng(seed, "measurement")
    err = config["measurement_error"]
    stature = growth_sim.measure_twice(rng, visit_curves.stature(visit_age), err["stature_cm"])
    sitting = growth_sim.measure_twice(
        rng, visit_curves.sitting_height(visit_age), err["sitting_height_cm"]
    )
    mass = growth_sim.measure_twice(rng, visit_curves.mass(visit_age), err["mass_kg"])
    visit_dates = _dates(start, visits.day)
    visit_athletes = [athlete_ids[i] for i in idx]
    anthropometrics = pd.DataFrame(
        {
            "athlete_id": visit_athletes,
            "date": visit_dates,
            "stature_1": stature[0],
            "stature_2": stature[1],
            "sitting_height_1": sitting[0],
            "sitting_height_2": sitting[1],
            "mass_1": mass[0],
            "mass_2": mass[1],
            "measurer_id": [
                f"measurer_{k}" for k in rng.integers(1, int(config["n_measurers"]) + 1, idx.size)
            ],
        }
    )

    rng = make_rng(seed, "movement")
    traits = movement_sim.sample_traits(rng, n, movement_cfg)
    session_ids = _uuids(rng, idx.size)
    screening_sessions = pd.DataFrame(
        {
            "id": session_ids,
            "athlete_id": visit_athletes,
            "date": visit_dates,
            "camera_setup": "synthetic",
        }
    )
    visit_offset = visit_curves.maturity_offset(visit_age)
    feature_frames = [
        pd.DataFrame(
            {
                "athlete_id": visit_athletes,
                "session_id": session_ids,
                "task": SCREENING_TASK,
                "trial": None,
                "limb": "both",
                "feature": name,
                "value": movement_sim.observe_feature(
                    rng, name, traits[name][idx], visit_offset, movement_cfg
                ),
                "quality": 1.0,
            }
        )
        for name in movement_cfg["features"]
    ]
    task_features = pd.concat(feature_frames, ignore_index=True)

    # --- weekly: exposure and injuries ---
    week = np.arange(n_weeks)
    week_age = baseline_age[:, None] + (week[None, :] * DAYS_PER_WEEK + 3.5) / DAYS_PER_YEAR
    flat_index = np.repeat(np.arange(n), n_weeks)
    week_curves = growth_sim.GrowthCurves(params.at(flat_index), growth_cfg)
    flat_age = week_age.ravel()
    week_offset = week_curves.maturity_offset(flat_age).reshape(n, n_weeks)
    leg_velocity = week_curves.leg_length_velocity(flat_age).reshape(n, n_weeks)
    load = movement_sim.latent_value(
        traits[movement_sim.LOAD_PROXY][:, None], week_offset, movement_cfg
    )
    exposure = injury_sim.sample_exposure(make_rng(seed, "exposure"), week_age, config["exposure"])
    total_hours = exposure.hours_training + exposure.hours_match
    hazards = injury_sim.weekly_hazards(
        leg_length_velocity=leg_velocity,
        load_proxy=load,
        maturity_offset=week_offset,
        hours=total_hours,
        spike=exposure.spike,
        config=config["injury"],
    )
    events = injury_sim.simulate_injuries(make_rng(seed, "injury"), hazards, config["injury"])

    week_starts = _dates(start, week * DAYS_PER_WEEK)
    flat_week_starts = week_starts * n
    flat_athletes = [athlete_ids[i] for i in flat_index]
    available = events.at_risk.ravel()
    training = np.where(available, exposure.hours_training.ravel(), 0.0)
    exposure_weekly = pd.DataFrame(
        {
            "athlete_id": flat_athletes,
            "week_start": flat_week_starts,
            "hours_training": np.round(training, 1),
            "hours_match": np.round(np.where(available, exposure.hours_match.ravel(), 0.0), 1),
            "sessions": np.round(training / config["exposure"]["hours_per_session"]).astype(int),
        }
    )
    rng = make_rng(seed, "injury_ids")
    injuries = pd.DataFrame(
        {
            "id": _uuids(rng, events.week.size),
            "athlete_id": [athlete_ids[i] for i in events.athlete_index],
            "onset_date": _dates(start, events.week * DAYS_PER_WEEK + events.onset_day_in_week),
            "osiics_code": "SYNTHETIC",
            "body_area": "synthetic",
            "tissue": "synthetic",
            "mechanism": "non_contact",
            "time_loss_days": events.time_loss_days.astype(int),
            "outcome_group": events.outcome_group,
            "diagnosed_by": "synthetic_generator",
        }
    )

    # --- truth (never model input) ---
    athlete_truth = pd.DataFrame(
        {
            "athlete_id": athlete_ids,
            "baseline_age": baseline_age,
            "true_aphv": params.aphv,
            "h1": params.h1,
            "h_theta": params.h_theta,
            "s0": params.s0,
            "s1": params.s1,
            "theta": params.theta,
            "sitting_ratio_adult": params.sitting_ratio_adult,
            **{f"trait_{name}": values for name, values in traits.items()},
        }
    )
    athlete_weeks_truth = pd.DataFrame(
        {
            "athlete_id": flat_athletes,
            "week_start": flat_week_starts,
            "week_index": np.tile(week, n),
            "age": flat_age,
            "true_maturity_offset": week_offset.ravel(),
            "true_height_velocity": week_curves.height_velocity(flat_age),
            "true_leg_length_velocity": leg_velocity.ravel(),
            "true_load_proxy": load.ravel(),
            "exposure_spike": exposure.spike.ravel(),
            "at_risk": available,
            **{f"hazard_{name}": hazards[name].ravel() for name in injury_sim.OUTCOME_GROUPS},
        }
    )

    tables = {
        "athletes": athletes,
        "anthropometrics": anthropometrics,
        "screening_sessions": screening_sessions,
        "task_features": task_features,
        "exposure_weekly": exposure_weekly,
        "injuries": injuries,
    }
    truth = {"athlete_truth": athlete_truth, "athlete_weeks_truth": athlete_weeks_truth}
    for frame in (*tables.values(), *truth.values()):
        frame["synthetic"] = True
    return SyntheticCohort(
        tables=tables,
        truth=truth,
        metadata=run_metadata(seed=seed, config=config, synthetic=True),
    )


def write_cohort(cohort: SyntheticCohort, out_dir: Path) -> Path:
    """Write tables as parquet, truth under ``truth/``, and the metadata file."""
    if cohort.metadata.get("synthetic") is not True:
        raise ValueError("Refusing to write a cohort that is not flagged synthetic")
    truth_dir = out_dir / TRUTH_DIR
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in cohort.tables.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    for name, frame in cohort.truth.items():
        frame.to_parquet(truth_dir / f"{name}.parquet", index=False)
    metadata = {
        **cohort.metadata,
        "n_rows": {name: len(frame) for name, frame in cohort.tables.items()},
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir
