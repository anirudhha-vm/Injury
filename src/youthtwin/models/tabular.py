"""Shared machinery for tabular risk models on the athlete-week table.

``HazardModel`` bundles a scikit-learn pipeline with the list of input columns, so that

- all preprocessing (imputation, spline knots, scaling, category levels) is **fitted on the
  rows passed to ``fit`` and nowhere else** (non-negotiable N4; v2 fitted a scaler per clip);
- the fitted preprocessing travels with the model;
- label-defining or truth columns are refused as inputs (N2).

In a discrete-time hazard model each athlete-week at risk is one observation and the model is a
binary classifier for "event in the outcome window"; the predicted probability is the hazard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from youthtwin.features.person_period import assert_no_leakage
from youthtwin.growth import splines

FloatArray = NDArray[np.float64]
_KNOT_QUANTILES = (0.05, 0.95)


class NaturalSplineTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Natural cubic spline expansion of each column, knots at quantiles of the fitted data.

    A column with too few distinct values to place the knots is passed through unchanged.
    Output per column: the column itself followed by its non-linear basis functions.
    """

    def __init__(self, n_knots: int = 4) -> None:
        self.n_knots = n_knots

    def fit(self, x: Any, y: Any = None) -> Self:
        data = np.asarray(x, dtype=np.float64)
        quantiles = np.linspace(*_KNOT_QUANTILES, self.n_knots)
        self.knots_: list[tuple[float, ...] | None] = []
        for column in data.T:
            knots = np.unique(np.quantile(column[np.isfinite(column)], quantiles))
            self.knots_.append(tuple(float(k) for k in knots) if knots.size >= 3 else None)
        return self

    def transform(self, x: Any) -> FloatArray:
        data = np.asarray(x, dtype=np.float64)
        blocks: list[FloatArray] = []
        for column, knots in zip(data.T, self.knots_, strict=True):
            blocks.append(column[:, None])
            if knots is not None:
                blocks.append(np.column_stack(splines.basis(column, knots)))
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.str_]:
        names: list[str] = []
        for name, knots in zip(input_features, self.knots_, strict=True):
            names.append(str(name))
            if knots is not None:
                names += [f"{name}_spline{k + 1}" for k in range(len(knots) - 2)]
        return np.asarray(names, dtype=np.str_)


def split_columns(frame: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[str]]:
    """Numeric and categorical input columns, in the given order."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    spline: Sequence[str] = (),
    n_knots: int = 4,
    scale: bool = True,
    impute: bool = True,
) -> ColumnTransformer:
    """Column-wise preprocessing; every statistic is learned in ``fit``."""
    spline = [c for c in spline if c in numeric]
    plain = [c for c in numeric if c not in spline]

    def numeric_steps(with_spline: bool) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if impute:
            steps.append(
                ("impute", SimpleImputer(strategy="median", add_indicator=not with_spline))
            )
        if with_spline:
            steps.append(("spline", NaturalSplineTransformer(n_knots)))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps) if steps else Pipeline([("pass", "passthrough")])

    transformers: list[tuple[str, Any, list[str]]] = []
    if spline:
        transformers.append(("spline", numeric_steps(True), list(spline)))
    if plain:
        transformers.append(("numeric", numeric_steps(False), plain))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("categorical", encoder, list(categorical)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


@dataclass
class HazardModel:
    """A named, fitted-or-unfitted cause-specific risk model."""

    name: str
    features: list[str]
    pipeline: Pipeline
    leakage_config: Mapping[str, Any]
    label: str | None = None

    def __post_init__(self) -> None:
        assert_no_leakage(self.features, self.leakage_config)

    def fit(self, frame: pd.DataFrame, label: str) -> Self:
        """Fit on exactly the rows of ``frame`` (pass training rows only)."""
        outcome = frame[label].to_numpy()
        if len(np.unique(outcome)) < 2:
            raise ValueError(
                f"Cannot fit '{self.name}': '{label}' has a single class in these rows"
            )
        self.pipeline.fit(frame[self.features], outcome)
        self.label = label
        return self

    def predict_proba(self, frame: pd.DataFrame) -> FloatArray:
        """Predicted probability of the event in the outcome window, per row."""
        if self.label is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted")
        return np.asarray(self.pipeline.predict_proba(frame[self.features])[:, 1], dtype=np.float64)

    @property
    def estimator(self) -> Any:
        """The fitted final estimator (for explanation: the fitted object, never a template)."""
        if self.label is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted")
        return self.pipeline[-1]

    def transformed_feature_names(self) -> list[str]:
        return [str(n) for n in self.pipeline[:-1].get_feature_names_out()]


def partial_effect(
    model: HazardModel, frame: pd.DataFrame, feature: str, *, low: float = 0.1, high: float = 0.9
) -> float:
    """Mean change in predicted risk when ``feature`` moves from its low to its high quantile.

    A model-agnostic description of the fitted relationship's direction and size, holding the
    other inputs at their observed values. It describes the model, not a causal effect.
    """
    values = frame[feature]
    if pd.api.types.is_bool_dtype(values) or values.dropna().isin([0, 1]).all():
        low_value, high_value = 0.0, 1.0
    else:
        low_value, high_value = float(values.quantile(low)), float(values.quantile(high))
    at_low = model.predict_proba(frame.assign(**{feature: low_value}))
    at_high = model.predict_proba(frame.assign(**{feature: high_value}))
    return float(np.mean(at_high - at_low))
