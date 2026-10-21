"""Real injury outcomes for the risk pipeline: the competitive runners' dataset (M10, Part 2).

Lövdal, den Hartigh & Azzopardi (2021), DataverseNL doi:10.34894/UWU9PV, CC0. Each row is an
event day of one of 74 runners with three weeks of aggregated training load before it and a
flag for whether an injury occurred that day. Consecutive rows of a runner overlap in 20 of 21
days of inputs, so rows of one runner are nearly duplicates: splits must be grouped by runner
(N3). These are adult runners and running injuries; the study checks the evaluation machinery
on real outcomes and says nothing about growth-related injury.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from youthtwin.eval import splits
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.risk.hazard_xgb import make_hazard_xgb
from youthtwin.models.tabular import HazardModel

ID_PREFIX = "runner-"
LEAKAGE = {"forbidden_predictor_prefixes": ["y_"], "forbidden_predictor_names": ["injury", "day"]}
RATIO_COLUMNS = ("rel total kms week 0_1", "rel total kms week 0_2", "rel total kms week 1_2")
Factory = Callable[[pd.DataFrame], HazardModel]


def _snake(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def tidy_week_approach(raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Snake-case features with a week suffix (``_w0`` = the week before the event day).

    The file's own week-to-week ratios divide by zero in rest weeks (values up to 2e8). They are
    replaced by ``log((a + 1) / (b + 1))`` of the weekly kilometres: the same information,
    bounded, and computed row by row from that row alone, so it learns nothing from other rows.
    """
    out: dict[str, Any] = {}
    for column in raw.columns:
        if column in ("Athlete ID", "injury", "Date", *RATIO_COLUMNS):
            continue
        base, _, suffix = column.rpartition(".")
        week = int(suffix) if suffix.isdigit() else 0
        name = _snake(base if suffix.isdigit() else column)
        out[f"{name}_w{week}"] = raw[column].astype(float).to_numpy()
    frame = pd.DataFrame(out)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        frame[f"log_ratio_kms_w{a}_w{b}"] = np.log(
            (frame[f"total_kms_w{a}"] + 1.0) / (frame[f"total_kms_w{b}"] + 1.0)
        )
    frame.insert(0, "athlete_id", ID_PREFIX + raw["Athlete ID"].astype(int).astype(str))
    frame.insert(1, "day", raw["Date"].astype(int).to_numpy())
    frame.insert(
        2, "week_start", pd.Timestamp(config["origin_date"]) + pd.to_timedelta(frame["day"], "D")
    )
    frame[config["label"]] = raw["injury"].astype(int).to_numpy()
    frame["synthetic"] = False
    return frame.sort_values(["athlete_id", "day"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    reserved = {"athlete_id", "day", "week_start", "synthetic", config["label"]}
    return [c for c in frame.columns if c not in reserved]


def model_factories(
    frame: pd.DataFrame, config: Mapping[str, Any], seed: int
) -> dict[str, Factory]:
    """Unfitted model builders; every preprocessing statistic is learned in ``fit`` (N4)."""
    features = feature_columns(frame, config)
    available = {
        "load_only": partial(
            make_hazard_logit,
            features=list(config["load_only_predictors"]),
            config={**config["penalised_logistic"], "C": 1000.0},
            leakage_config=LEAKAGE,
            seed=seed,
            name="load_only",
        ),
        "penalised_logistic": partial(
            make_hazard_logit,
            features=features,
            config=config["penalised_logistic"],
            leakage_config=LEAKAGE,
            seed=seed,
            name="penalised_logistic",
        ),
        "gradient_boosting": partial(
            make_hazard_xgb,
            features=features,
            config={**config["gradient_boosting"], "monotone_constraints": {}},
            leakage_config=LEAKAGE,
            seed=seed,
            name="gradient_boosting",
        ),
    }
    unknown = sorted(set(config["models"]) - set(available))
    if unknown:
        raise KeyError(f"Unknown models: {unknown}")
    return {name: available[name] for name in config["models"]}


def evaluation_splits(
    frame: pd.DataFrame, config: Mapping[str, Any], seed: int
) -> dict[str, list[splits.Split]]:
    """The three schemes: runner-grouped folds (primary), temporal hold-out, row-level folds.

    Row-level folds put rows of the same runner on both sides; they are included only to show
    how much that inflates performance, never as a result.
    """
    label = frame[config["label"]]
    grouped = splits.grouped_stratified_folds(
        frame["athlete_id"], label, n_splits=int(config["n_folds"]), seed=seed
    )
    temporal = splits.temporal_holdout(
        frame["week_start"], fraction=float(config["temporal_holdout_fraction"]), horizon_weeks=0
    )
    row_level = []
    folds = StratifiedKFold(n_splits=int(config["n_folds"]), shuffle=True, random_state=seed)
    for k, (train, test) in enumerate(folds.split(np.zeros(len(frame)), label)):
        train_mask = np.zeros(len(frame), dtype=bool)
        test_mask = train_mask.copy()
        train_mask[train], test_mask[test] = True, True
        row_level.append(splits.Split(f"row_level_{k}", train_mask, test_mask))
    return {"runner_grouped": grouped, "temporal_holdout": [temporal], "row_level_leaky": row_level}


def predict(
    frame: pd.DataFrame,
    label: str,
    factories: Mapping[str, Factory],
    folds: Sequence[splits.Split],
    scheme: str,
) -> pd.DataFrame:
    """Held-out predictions of every model under one scheme (rows outside every test part
    are dropped). Columns as in ``eval.predictions``, so ``eval.report`` works on them."""
    tested = np.zeros(len(frame), dtype=bool)
    risks = {name: np.full(len(frame), np.nan) for name in factories}
    fold_of_row = np.full(len(frame), -1)
    for k, fold in enumerate(folds):
        if scheme == "runner_grouped":
            splits.assert_no_group_overlap(frame["athlete_id"], fold)
        tested |= fold.test
        fold_of_row[fold.test] = k
        for name, factory in factories.items():
            model = factory(frame).fit(frame[fold.train], label)
            risks[name][fold.test] = model.predict_proba(frame[fold.test])
    out = frame.loc[tested, ["athlete_id", "week_start"]].copy()
    out["outcome"] = frame.loc[tested, label].to_numpy()
    out["label"] = label
    out["scheme"] = scheme
    out["fold"] = fold_of_row[tested]
    out["synthetic"] = False
    for name, values in risks.items():
        out[f"risk_{name}"] = values[tested]
    return out.reset_index(drop=True)
