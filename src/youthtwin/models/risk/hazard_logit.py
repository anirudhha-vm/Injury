"""Penalised discrete-time logistic hazard (brief §6.6): the main tabular competitor.

Elastic-net logistic regression on athlete-weeks, with natural cubic spline terms for the
continuous growth variables so that risk can rise and fall with maturity instead of being
forced to be monotone. Nothing about the shape is set by hand (v2 used a hand-drawn Gaussian
risk curve around PHV); it is learned and can be reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from youthtwin.models.tabular import HazardModel, preprocessor, split_columns


def make_hazard_logit(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: Mapping[str, Any],
    leakage_config: Mapping[str, Any],
    *,
    seed: int = 0,
    name: str = "hazard_logit",
) -> HazardModel:
    """Build the (unfitted) model. ``config`` is the ``hazard_logit`` block of hazard.yaml."""
    numeric, categorical = split_columns(frame, features)
    prepare = preprocessor(
        numeric,
        categorical,
        spline=[c for c in config["spline_terms"] if c in features],
        n_knots=int(config["n_spline_knots"]),
    )
    classifier = LogisticRegression(
        solver="saga",  # with l1_ratio between 0 and 1 this is the elastic net
        C=float(config["C"]),
        l1_ratio=float(config["l1_ratio"]),
        max_iter=int(config["max_iter"]),
        class_weight=config["class_weight"],
        random_state=seed,  # the saga solver shuffles; fix it so fits are reproducible
    )
    pipeline = Pipeline([("prepare", prepare), ("model", classifier)])
    return HazardModel(name, list(features), pipeline, leakage_config)
