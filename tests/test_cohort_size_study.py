"""Cohort-size simulation study: structure, reproducibility and that it reads only held-out rows."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.eval import cohort_size
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config


@pytest.fixture(scope="module")
def configs(person_period_config: dict[str, Any], hazard_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "person_period": person_period_config,
        "hazard": hazard_config,
        "baselines": load_config("models/baselines.yaml"),
        "mech": load_config("features/mech.yaml"),
    }


@pytest.fixture(scope="module")
def study(cohort_config: dict[str, Any], configs: dict[str, Any]) -> pd.DataFrame:
    settings = load_config("eval/cohort_size_study.yaml")
    settings.update(cohort_sizes=[150, 300], n_replicates=2, n_folds=2, label="y_all_non_contact")
    return cohort_size.run_study(cohort_config, settings, configs)


def test_config_matches_the_planning_scenarios() -> None:
    settings = load_config("eval/cohort_size_study.yaml")
    assert settings["cohort_sizes"] == load_config("eval/sample_size.yaml")["cohort_size_scenarios"]
    assert not [k for k in settings if "expected" in k.lower() or "target" in k.lower()]


def test_study_structure(study: pd.DataFrame, cohort_config: dict[str, Any]) -> None:
    assert list(study.columns) == cohort_size.REPLICATE_COLUMNS
    assert sorted(study["n_athletes"].unique()) == [150, 300]
    # Replicates are different cohorts: the seed moves, and so does the event count.
    seeds = study.groupby("replicate")["cohort_seed"].unique()
    assert seeds[0].tolist() == [cohort_config["seed"]] and seeds[1].tolist() == [
        cohort_config["seed"] + 1
    ]
    events = study[study["metric"] == "athletes_with_event"]
    assert events["value"].nunique() > 1
    assert (events.groupby("n_athletes")["value"].mean().diff().dropna() > 0).all()
    auc = study[study["metric"] == "auc"]
    assert auc["value"].between(0, 1).all()
    assert set(auc["model"]) >= {"age_only", "hazard_logit", cohort_size.TRUE_RISK}
    # The known true risk discriminates better than chance.
    assert auc.loc[auc["model"] == cohort_size.TRUE_RISK, "value"].mean() > 0.55
    # The confirmatory model reports how many predictors the cohort supported, within its cap.
    counts = study[study["metric"] == "n_predictors"]["value"]
    cap = load_config("models/hazard.yaml")["confirmatory"]["max_parameters"]
    assert len(counts) == 4 and counts.between(0, cap).all()


def test_summary_and_reproducibility(
    study: pd.DataFrame, cohort_config: dict[str, Any], configs: dict[str, Any]
) -> None:
    summary = cohort_size.summarise(study, level=0.9)
    assert summary["synthetic"].all() and (summary["n_replicates"] == 2).all()
    row = summary[
        (summary["n_athletes"] == 300)
        & (summary["model"] == "age_only")
        & (summary["metric"] == "auc")
    ].iloc[0]
    values = study[
        (study["n_athletes"] == 300) & (study["model"] == "age_only") & (study["metric"] == "auc")
    ]["value"]
    assert row["mean"] == pytest.approx(values.mean())
    assert row["low"] <= row["mean"] <= row["high"]

    # One replicate recomputed on its own gives the same numbers.
    config = copy.deepcopy(cohort_config)
    config["n_athletes"] = 150
    again = cohort_size.evaluate_cohort(
        generate_cohort(config),
        label="y_all_non_contact",
        models=["age_only"],
        configs=configs,
        n_folds=2,
        seed=0,
    )
    first = study[(study["n_athletes"] == 150) & (study["replicate"] == 0)]
    for metric in ("auc", "calibration_slope"):
        assert again.loc[
            (again["model"] == "age_only") & (again["metric"] == metric), "value"
        ].item() == pytest.approx(
            first.loc[(first["model"] == "age_only") & (first["metric"] == metric), "value"].item()
        )


def test_unknown_model_is_refused(cohort_config: dict[str, Any], configs: dict[str, Any]) -> None:
    config = copy.deepcopy(cohort_config)
    config["n_athletes"] = 60
    with pytest.raises(KeyError, match="Unknown model"):
        cohort_size.evaluate_cohort(
            generate_cohort(config),
            label="y_all_non_contact",
            models=["crystal_ball"],
            configs=configs,
            n_folds=2,
            seed=0,
        )


def test_undefined_metric_is_nan() -> None:
    def undefined(outcome: Any, risk: Any) -> float:
        raise ValueError("no events")

    assert np.isnan(cohort_size._or_nan(undefined, [0], [0.1]))
