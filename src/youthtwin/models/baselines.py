"""Baseline risk models (brief §6.6): the comparisons the proposed model has to beat.

- ``age_only``: logistic regression on age.
- ``maturity_growth_velocity``: maturity offset and growth velocities (the clinical baseline).
- ``screening_only``: the latest screening features.

All are plain logistic regressions with median imputation and scaling fitted on training rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from youthtwin.models.tabular import HazardModel, preprocessor, split_columns


def baseline_features(name: str, frame: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    """Input columns of baseline ``name`` that exist in ``frame``."""
    try:
        spec = config["baselines"][name]
    except KeyError:
        raise KeyError(f"Unknown baseline '{name}'. Known: {sorted(config['baselines'])}") from None
    if "predictor_prefix" in spec:
        features = [c for c in frame.columns if c.startswith(spec["predictor_prefix"])]
    else:
        features = [c for c in spec["predictors"] if c in frame.columns]
    if not features:
        raise ValueError(f"Baseline '{name}' has no input columns in this table")
    return features


def make_baseline(
    name: str, frame: pd.DataFrame, config: Mapping[str, Any], leakage_config: Mapping[str, Any]
) -> HazardModel:
    features = baseline_features(name, frame, config)
    numeric, categorical = split_columns(frame, features)
    classifier = LogisticRegression(C=float(config["C"]), max_iter=int(config["max_iter"]))
    pipeline = Pipeline([("prepare", preprocessor(numeric, categorical)), ("model", classifier)])
    return HazardModel(name, features, pipeline, leakage_config)
