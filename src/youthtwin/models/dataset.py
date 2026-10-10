"""Assemble model-ready arrays for the neural hazard from the cohort tables.

- ``cohort_sequences``: synthetic cohorts only. Simulates each screening session's trials and
  runs them through the real pose pipeline (keypoints -> angles -> events -> curves), so the
  encoder sees exactly what it would see from video-derived keypoints.
- ``neural_data``: athlete-week arrays (growth vector, tabular inputs, session index, outcome).
- ``growth_draw_arrays``: draws of the growth vector from the growth-twin posteriors, for
  uncertainty propagation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.features.person_period import TWIN_PREFIX, assert_no_leakage
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models.encoder.sequences import (
    SessionSequences,
    channel_names,
    stack_sessions,
    trial_channels,
)
from youthtwin.models.risk.hazard_neural import NeuralData
from youthtwin.pose import task_features as tf
from youthtwin.sim import movement_sim
from youthtwin.sim.cohort import SyntheticCohort
from youthtwin.utils.seeding import make_rng

FloatArray = NDArray[np.float64]
SIDES = ("left", "right")
_MEAN, _SD = "_mean", "_sd"


def cohort_sequences(
    cohort: SyntheticCohort,
    cohort_config: Mapping[str, Any],
    *,
    filter_config: Mapping[str, Any],
    event_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> tuple[SessionSequences, pd.DataFrame]:
    """Simulate and process every session's trials. Returns the sequences and the exclusion log."""
    if cohort.metadata.get("synthetic") is not True:
        raise ValueError("cohort_sequences simulates movement and is for synthetic cohorts only")
    features = cohort.tables["task_features"]
    load = features[features["feature"] == movement_sim.LOAD_PROXY].set_index("session_id")
    sessions = cohort.tables["screening_sessions"]
    rng = make_rng(int(cohort_config["seed"]), "trial_sequences")
    channels = channel_names(SIDES, frontal=True)
    per_session: dict[str, list[FloatArray]] = {}
    athlete_of: dict[str, str] = {}
    excluded: list[dict[str, Any]] = []
    for session_id, athlete_id in zip(sessions["id"], sessions["athlete_id"], strict=True):
        athlete_of[session_id] = athlete_id
        trials = movement_sim.session_trials(
            rng, float(load.loc[session_id, "value"]), cohort_config["movement"]
        )
        kept: list[FloatArray] = []
        for number, simulated in enumerate(trials, start=1):
            result = tf.trial_features(
                tf.TrialInput(
                    athlete_id,
                    session_id,
                    simulated.task,
                    number,
                    simulated.limb,
                    simulated.sagittal,
                    simulated.frontal,
                ),
                filter_config=filter_config,
                event_config=event_config,
                feature_config=feature_config,
            )
            if result.included:
                kept.append(trial_channels(result, SIDES, frontal=True))
            else:
                excluded.append(
                    {
                        "athlete_id": athlete_id,
                        "session_id": session_id,
                        "trial": number,
                        "reason": result.excluded_reason,
                    }
                )
        per_session[session_id] = kept
    log = pd.DataFrame(excluded, columns=["athlete_id", "session_id", "trial", "reason"])
    return stack_sessions(per_session, athlete_of, channels), log


def growth_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    """Growth-vector columns: growth-twin features if present, otherwise the observed fallback."""
    twin = list(config["growth_columns_twin"])
    if all(column in frame.columns for column in twin):
        return twin
    return list(config["growth_columns_observed"])


def neural_data(
    frame: pd.DataFrame,
    sequences: SessionSequences,
    *,
    label: str,
    growth: Sequence[str],
    tabular: Sequence[str],
    leakage_config: Mapping[str, Any],
) -> NeuralData:
    """Arrays for the neural hazard from an athlete-week table."""
    assert_no_leakage([*growth, *tabular], leakage_config)
    index = sequences.index_of()
    session_index = np.full(len(frame), -1, dtype=np.int64)
    if "screen_session_id" in frame.columns:
        mapped = frame["screen_session_id"].map(index).fillna(-1)
        session_index = np.asarray(mapped, dtype=np.int64)
    return NeuralData(
        growth=frame[list(growth)].to_numpy(dtype=np.float64),
        tabular=frame[list(tabular)].to_numpy(dtype=np.float64),
        session_index=session_index,
        outcome=frame[label].to_numpy(dtype=np.float64),
        athlete=frame["athlete_id"].to_numpy(dtype=np.str_),
    )


def growth_draw_arrays(
    frame: pd.DataFrame,
    posteriors: Mapping[str, SitarPosterior],
    growth: Sequence[str],
    *,
    n_draws: int,
    seed: int,
) -> FloatArray:
    """Draws of the growth vector for every athlete-week: (draws, rows, len(growth)).

    Columns ending in ``_mean`` are replaced by posterior draws of that quantity; ``_sd``
    columns describe the posterior itself and keep their value. Rows of athletes without a
    posterior keep their table values in every draw (no growth uncertainty available).
    """
    base = frame[list(growth)].to_numpy(dtype=np.float64)
    out = np.repeat(base[None, :, :], n_draws, axis=0)
    rng = np.random.default_rng(seed)
    drawn = [
        (k, column.removeprefix(TWIN_PREFIX).removesuffix(_MEAN))
        for k, column in enumerate(growth)
        if column.startswith(TWIN_PREFIX) and column.endswith(_MEAN)
    ]
    if not drawn:
        return out
    rows_of = frame.groupby("athlete_id", sort=False).indices
    sex_of = frame.groupby("athlete_id", sort=False)["sex"].first()
    ages = frame["age"].to_numpy(dtype=np.float64)
    for athlete, rows in rows_of.items():
        posterior = posteriors.get(str(sex_of.loc[str(athlete)]))
        if posterior is None or str(athlete) not in posterior.athlete_ids:
            continue
        keep = rng.choice(posterior.n_draws, n_draws, replace=posterior.n_draws < n_draws)
        draws = posterior.draws_at(str(athlete), ages[rows])
        for k, quantity in drawn:
            out[:, rows, k] = draws[quantity][keep]
    return out
