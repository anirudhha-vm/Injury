"""Runners' injury dataset: tidying, leakage guard, split schemes and held-out prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import runners
from youthtwin.utils.config import load_config

CONFIG = load_config("external/runners.yaml")
BASES = ["nr. sessions", "total kms", "avg exertion"]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """Shaped like week_approach_maskedID_timeseries.csv, with a known load effect."""
    rng = np.random.default_rng(0)
    n_runners, n_days = 12, 150
    rows = []
    for runner in range(n_runners):
        usual = rng.uniform(20, 80)
        for day in range(n_days):
            record: dict[str, Any] = {}
            for week in range(3):
                suffix = "" if week == 0 else f".{week}"
                kms = max(0.0, rng.normal(usual, 15))
                record[f"nr. sessions{suffix}"] = rng.integers(3, 9)
                record[f"total kms{suffix}"] = kms
                record[f"avg exertion{suffix}"] = rng.uniform(0, 1)
            record["Athlete ID"] = runner
            risk = 1 / (1 + np.exp(-(-4.0 + 0.04 * (record["total kms"] - 50))))
            record["injury"] = int(rng.random() < risk)
            record["rel total kms week 0_1"] = 2e8 if record["total kms.1"] == 0 else 1.0
            record["rel total kms week 0_2"] = 1.0
            record["rel total kms week 1_2"] = 1.0
            record["Date"] = day + 10 * runner
            rows.append(record)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame(raw: pd.DataFrame) -> pd.DataFrame:
    return runners.tidy_week_approach(raw, CONFIG)


def test_tidy_names_weeks_and_replaces_unbounded_ratios(
    raw: pd.DataFrame, frame: pd.DataFrame
) -> None:
    assert {"total_kms_w0", "total_kms_w1", "total_kms_w2", "nr_sessions_w2"} <= set(frame.columns)
    assert not any(c.startswith("rel_") for c in frame.columns)
    row = frame.iloc[5]
    expected = np.log((row["total_kms_w0"] + 1) / (row["total_kms_w1"] + 1))
    assert row["log_ratio_kms_w0_w1"] == pytest.approx(expected)
    assert frame.filter(like="log_ratio").abs().max().max() < 10
    assert frame["injury"].sum() == raw["injury"].sum() and not frame["synthetic"].any()
    assert frame["athlete_id"].str.startswith("runner-").all()


def test_label_and_time_are_not_inputs(frame: pd.DataFrame) -> None:
    features = runners.feature_columns(frame, CONFIG)
    assert "injury" not in features and "day" not in features and "week_start" not in features
    with pytest.raises(ValueError, match="Leakage"):
        runners.make_hazard_logit(
            frame, ["injury"], CONFIG["penalised_logistic"], runners.LEAKAGE, seed=0
        )


def test_split_schemes(frame: pd.DataFrame) -> None:
    schemes = runners.evaluation_splits(frame, CONFIG, seed=0)
    for fold in schemes["runner_grouped"]:
        assert not set(frame.loc[fold.train, "athlete_id"]) & set(
            frame.loc[fold.test, "athlete_id"]
        )
    (temporal,) = schemes["temporal_holdout"]
    assert frame.loc[temporal.train, "day"].max() < frame.loc[temporal.test, "day"].min()
    leaky = schemes["row_level_leaky"][0]
    assert set(frame.loc[leaky.train, "athlete_id"]) & set(frame.loc[leaky.test, "athlete_id"])


def test_predictions_are_held_out_and_find_the_load_effect(frame: pd.DataFrame) -> None:
    config = {**CONFIG, "models": ["load_only", "gradient_boosting"]}
    factories = runners.model_factories(frame, config, seed=0)
    folds = runners.evaluation_splits(frame, config, seed=0)["runner_grouped"]
    predictions = runners.predict(frame, "injury", factories, folds, "runner_grouped")
    assert (
        len(predictions) == len(frame)
        and predictions[["risk_load_only", "risk_gradient_boosting"]].notna().all().all()
    )
    assert set(predictions["fold"]) == set(range(len(folds)))
    from youthtwin.eval.metrics import auc

    assert auc(predictions["outcome"], predictions["risk_load_only"]) > 0.6
    with pytest.raises(KeyError, match="Unknown"):
        runners.model_factories(frame, {**CONFIG, "models": ["oracle"]}, seed=0)
