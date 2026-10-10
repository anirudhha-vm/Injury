"""The confirmatory risk model: as small as the number of events allows.

Decision recorded in ``docs/RESEARCH_STRATEGY.md``: realistic injury rates mean the planned
cohort supports only a few predictor parameters. The confirmatory model therefore takes its
predictors from a **pre-specified ordered list** and uses only as many as the Riley et al.
sample-size criteria support for the number of athletes and events actually observed. The
order is fixed in ``configs/models/hazard.yaml`` before outcome data are seen, so the data
cannot choose the predictors. Larger models (penalised hazard, boosted hazard, conditioned
encoder) are reported as exploratory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from youthtwin.eval.sample_size import supported_parameters
from youthtwin.models.tabular import HazardModel, split_columns


def athlete_level_counts(frame: pd.DataFrame, label: str) -> tuple[int, float]:
    """Number of athletes and the proportion with at least one event (the planning unit)."""
    per_athlete = frame.groupby("athlete_id")[label].max()
    return len(per_athlete), float(per_athlete.mean())


def confirmatory_predictors(
    frame: pd.DataFrame, label: str, config: Mapping[str, Any]
) -> list[str]:
    """The leading entries of the pre-specified list that the data can support.

    Uses the athlete-level sample size and event proportion of ``frame`` (pass the whole
    development cohort, so that the choice is one number fixed for all folds). Entries that are
    absent from the table are skipped without being replaced. Returns an empty list if not even
    one parameter is supported.
    """
    ordered = [c for c in config["candidate_predictors_in_order"] if c in frame.columns]
    n_athletes, proportion = athlete_level_counts(frame, label)
    if not 0.0 < proportion < 1.0:
        return []
    supported = supported_parameters(n=n_athletes, prevalence=proportion)
    return ordered[: min(supported, int(config["max_parameters"]))]


def make_confirmatory(
    frame: pd.DataFrame,
    predictors: list[str],
    config: Mapping[str, Any],
    leakage_config: Mapping[str, Any],
) -> HazardModel:
    """Plain logistic model on the chosen predictors (one parameter each, no splines)."""
    if not predictors:
        raise ValueError(
            "The cohort does not support a confirmatory model with even one predictor; "
            "report the outcome descriptively and treat all models as exploratory"
        )
    _, categorical = split_columns(frame, predictors)
    if categorical:
        raise ValueError(
            f"Confirmatory predictors must be numeric (one parameter each): {categorical}"
        )
    # Median imputation without missing-value indicators: an indicator per predictor would
    # double the parameter count that the sample-size calculation is meant to cap.
    prepare = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    classifier = LogisticRegression(C=float(config["C"]), max_iter=int(config["max_iter"]))
    pipeline = Pipeline([("prepare", prepare), ("model", classifier)])
    return HazardModel("confirmatory", list(predictors), pipeline, leakage_config)
