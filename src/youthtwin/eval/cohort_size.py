"""Simulation study: out-of-fold performance of the tabular models against cohort size.

For each cohort size, several independent synthetic cohorts are generated (different seeds) and
every model is fitted with athlete-grouped cross-validation, exactly as in the main evaluation.
The spread across replicate cohorts shows directly what a single cohort of that size can and
cannot establish, which is the sample-size argument made visually.

Because the cohorts are synthetic, the true risk of every athlete-week is known. Its own AUC is
reported as ``true_risk``: the ceiling that no model can exceed on average.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.eval import calibration, metrics, splits
from youthtwin.eval.predictions import out_of_fold_predictions, risk_columns
from youthtwin.features.mech_load import add_mech_proxies
from youthtwin.features.person_period import build_person_period
from youthtwin.models.baselines import make_baseline
from youthtwin.models.confirmatory import (
    athlete_level_counts,
    confirmatory_predictors,
    make_confirmatory,
)
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.tabular import HazardModel
from youthtwin.sim.cohort import SyntheticCohort, generate_cohort
from youthtwin.sim.truth import true_window_risk

COHORT = "(cohort)"  # rows describing the replicate cohort itself
TRUE_RISK = "true_risk"
REPLICATE_COLUMNS = ["n_athletes", "replicate", "cohort_seed", "model", "metric", "value"]


def _or_nan(function: Callable[[Any, Any], float], outcome: Any, risk: Any) -> float:
    """The metric, or NaN where it is undefined (no events, or a constant prediction)."""
    try:
        return float(function(outcome, risk))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _baseline(
    frame: pd.DataFrame, *, name: str, config: Mapping[str, Any], leakage: Mapping[str, Any]
) -> HazardModel:
    return make_baseline(name, frame, config, leakage)


def evaluate_cohort(
    cohort: SyntheticCohort,
    *,
    label: str,
    models: Sequence[str],
    configs: Mapping[str, Any],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Out-of-fold metrics of every model on one synthetic cohort (columns: model, metric, value).

    ``configs`` holds the loaded YAML blocks ``person_period, hazard, baselines, mech``.
    """
    leakage = configs["person_period"]
    period = build_person_period(cohort.tables, config=leakage, hazard_config=configs["hazard"])
    frame = add_mech_proxies(period.frame, configs["mech"])
    truth = true_window_risk(cohort, frame, label, period.horizon_weeks)
    n_athletes, proportion = athlete_level_counts(frame, label)
    rows: list[tuple[str, str, float]] = [
        (COHORT, "athletes_with_event", round(n_athletes * proportion)),
        (COHORT, "proportion_of_athletes_with_event", proportion),
        (COHORT, "athlete_weeks", len(frame)),
    ]

    factories: dict[str, Callable[[pd.DataFrame], HazardModel]] = {}
    for name in models:
        if name in configs["baselines"]["baselines"]:
            factories[name] = partial(
                _baseline, name=name, config=configs["baselines"], leakage=leakage
            )
        elif name == "hazard_logit":
            features = period.predictors + [c for c in frame.columns if c.startswith("mech_")]
            factories[name] = partial(
                make_hazard_logit,
                features=features,
                config=configs["hazard"]["hazard_logit"],
                leakage_config=leakage,
                seed=seed,
            )
        elif name == "confirmatory":
            # Chosen once from the whole cohort's size and event count, never from the outcome
            # model: the same number for every fold.
            chosen = confirmatory_predictors(frame, label, configs["hazard"]["confirmatory"])
            rows.append((name, "n_predictors", len(chosen)))
            if chosen:
                factories[name] = partial(
                    make_confirmatory,
                    predictors=chosen,
                    config=configs["hazard"]["confirmatory"],
                    leakage_config=leakage,
                )
        else:
            raise KeyError(f"Unknown model in the cohort-size study: {name}")

    folds = splits.grouped_stratified_folds(
        frame["athlete_id"], frame[label], n_splits=n_folds, seed=seed
    )
    predictions = out_of_fold_predictions(frame, label, factories, folds)
    outcome = predictions["outcome"].to_numpy()
    known = np.isfinite(truth)
    rows.append((TRUE_RISK, "auc", _or_nan(metrics.auc, outcome[known], truth[known])))
    for name, column in risk_columns(predictions).items():
        risk = predictions[column].to_numpy()
        rows += [
            (name, "auc", _or_nan(metrics.auc, outcome, risk)),
            (name, "calibration_slope", _or_nan(calibration.calibration_slope, outcome, risk)),
            (
                name,
                "correlation_with_true_risk",
                float(np.corrcoef(risk[known], truth[known])[0, 1]),
            ),
        ]
    return pd.DataFrame(rows, columns=["model", "metric", "value"])


def run_study(
    cohort_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    configs: Mapping[str, Any],
    *,
    on_replicate: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Every size x replicate; one tidy row per model and metric."""
    frames = []
    for size in study_config["cohort_sizes"]:
        for replicate in range(int(study_config["n_replicates"])):
            config = copy.deepcopy(dict(cohort_config))
            config["n_athletes"] = int(size)
            config["seed"] = int(cohort_config["seed"]) + replicate
            table = evaluate_cohort(
                generate_cohort(config),
                label=study_config["label"],
                models=study_config["models"],
                configs=configs,
                n_folds=int(study_config["n_folds"]),
                seed=replicate,
            )
            table.insert(0, "cohort_seed", config["seed"])
            table.insert(0, "replicate", replicate)
            table.insert(0, "n_athletes", int(size))
            frames.append(table)
            if on_replicate is not None:
                on_replicate(int(size), replicate)
    return pd.concat(frames, ignore_index=True)[REPLICATE_COLUMNS]


def summarise(replicates: pd.DataFrame, level: float) -> pd.DataFrame:
    """Mean and spread across replicate cohorts, per size, model and metric."""
    tail = (1.0 - level) / 2.0
    grouped = replicates.groupby(["n_athletes", "model", "metric"], sort=True)["value"]
    summary = grouped.agg(
        mean="mean",
        sd="std",
        low=lambda v: v.quantile(tail),
        high=lambda v: v.quantile(1.0 - tail),
        n_replicates="count",
    ).reset_index()
    summary["interval_level"] = level
    summary["synthetic"] = True
    return summary
