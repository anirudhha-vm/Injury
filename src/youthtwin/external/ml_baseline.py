"""A machine-learning baseline for age at PHV from the visits seen so far (M11).

Gradient boosting on features a practitioner has after ``k`` visits: age and measurements at
the latest visit, observed velocities, years observed, and the Moore estimate. Trained per sex
on sequences drawn from the training children only (several random entry ages each, all visit
counts), and evaluated on exactly the held-out sequences the twin sees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from youthtwin.external import sequential
from youthtwin.growth.maturity_equations import Sex

FEATURES = (
    "visits",
    "age_last",
    "stature_last",
    "sitting_last",
    "mass_last",
    "leg_last",
    "years_observed",
    "velocity_mean",
    "velocity_last",
    "sitting_velocity_mean",
    "moore_aphv",
    "moore_offset_last",
)


def features(visits: pd.DataFrame, sex: Sex) -> dict[str, float]:
    """Features from the visits so far (sorted by age)."""
    last = visits.iloc[-1]
    span = float(visits["age"].iloc[-1] - visits["age"].iloc[0])
    moore_aphv, _ = sequential._equation_aphv(visits, sex)
    out = {
        "visits": float(len(visits)),
        "age_last": float(last["age"]),
        "stature_last": float(last["stature"]),
        "sitting_last": float(last["sitting_height"]),
        "mass_last": float(last["mass"]),
        "leg_last": float(last["stature"] - last["sitting_height"]),
        "years_observed": span,
        "velocity_mean": np.nan,
        "velocity_last": np.nan,
        "sitting_velocity_mean": np.nan,
        "moore_aphv": moore_aphv,
        "moore_offset_last": sequential._moore_offset(visits, sex),
    }
    if len(visits) >= 2 and span > 0:
        previous = visits.iloc[-2]
        out["velocity_mean"] = float(visits["stature"].iloc[-1] - visits["stature"].iloc[0]) / span
        out["velocity_last"] = float(last["stature"] - previous["stature"]) / float(
            last["age"] - previous["age"]
        )
        out["sitting_velocity_mean"] = (
            float(visits["sitting_height"].iloc[-1] - visits["sitting_height"].iloc[0]) / span
        )
    return out


def training_set(
    frame: pd.DataFrame,
    athletes: Sequence[str],
    reference: Mapping[str, float],
    sex: Sex,
    config: Mapping[str, Any],
    *,
    entries: int,
    seed: int,
) -> pd.DataFrame:
    """Rows (features, target) from the training children's own sequences."""
    rows = []
    labelled = [a for a in athletes if a in reference]
    for draw in range(entries):
        sequences = sequential.entry_sequences(frame, labelled, config, seed + draw)
        for athlete, visits in sequences.items():
            for k in range(1, min(len(visits), int(config["max_visits"])) + 1):
                rows.append({**features(visits.head(k), sex), "target": reference[athlete]})
    return pd.DataFrame(rows)


def fit_predict(
    train: pd.DataFrame,
    sequences: Mapping[str, pd.DataFrame],
    sex: Sex,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    """Predictions for every held-out athlete after each visit (athlete_id, visits, ml_aphv)."""
    model = HistGradientBoostingRegressor(
        max_iter=int(settings["max_iter"]),
        learning_rate=float(settings["learning_rate"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        max_depth=int(settings["max_depth"]),
        random_state=seed,
    )
    model.fit(train[list(FEATURES)], train["target"])
    rows = []
    for athlete, visits in sequences.items():
        for k in range(1, min(len(visits), int(config["max_visits"])) + 1):
            row = features(visits.head(k), sex)
            rows.append({"athlete_id": athlete, "visits": k, "age": row["age_last"], **row})
    out = pd.DataFrame(rows)
    out["ml_aphv"] = model.predict(out[list(FEATURES)])
    return out[["athlete_id", "visits", "age", "ml_aphv"]]


def conformal_halfwidth(
    scores: pd.DataFrame, estimate: str, level: float, by: Sequence[str] = ("sex", "visits")
) -> pd.Series:
    """Cross-fold split-conformal half-width of absolute residuals (no scale model)."""
    error = (scores["reference_aphv"] - scores[estimate]).abs()
    keys = scores[list(by)].astype(str).agg("|".join, axis=1)
    half = pd.Series(np.nan, index=scores.index)
    for fold in scores["fold"].unique():
        mine = scores["fold"] == fold
        for key in keys[mine].unique():
            calibration = (~mine) & (keys == key)
            rows = mine & (keys == key)
            half[rows] = sequential._conformal_quantile(error[calibration].to_numpy(), level)
    return half
