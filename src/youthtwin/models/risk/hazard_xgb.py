"""Gradient-boosted discrete-time hazard (brief §6.6).

XGBoost on athlete-weeks. Missing values are handled by the trees, so numeric inputs are passed
through unscaled and unimputed; categories are one-hot encoded. Optional monotone constraints
can encode a direction that is known in advance for a feature.

For explanation (M7) use ``HazardModel.estimator``, which is the fitted booster. v2 passed an
unfitted template estimator to SHAP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from youthtwin.models.tabular import HazardModel, preprocessor, split_columns


def make_hazard_xgb(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: Mapping[str, Any],
    leakage_config: Mapping[str, Any],
    *,
    seed: int,
    name: str = "hazard_xgb",
) -> HazardModel:
    """Build the (unfitted) model. ``config`` is the ``hazard_xgb`` block of hazard.yaml."""
    numeric, categorical = split_columns(frame, features)
    constraints = dict(config.get("monotone_constraints") or {})
    unknown = sorted(set(constraints) - set(numeric))
    if unknown:
        raise ValueError(f"Monotone constraints on columns that are not numeric inputs: {unknown}")
    n_categories = sum(frame[c].nunique() for c in categorical)
    monotone = tuple(int(constraints.get(c, 0)) for c in numeric) + (0,) * n_categories
    classifier = XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        min_child_weight=float(config["min_child_weight"]),
        monotone_constraints=monotone if constraints else None,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )
    prepare = preprocessor(numeric, categorical, scale=False, impute=False)
    pipeline = Pipeline([("prepare", prepare), ("model", classifier)])
    return HazardModel(name, list(features), pipeline, leakage_config)
