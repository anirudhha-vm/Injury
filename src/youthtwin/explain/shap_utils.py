"""SHAP explanations for the tabular hazard models.

v2 defect: SHAP was given ``CalibratedClassifierCV(...).estimator``, an unfitted template. Here
the explainer always receives ``HazardModel.estimator``, the fitted final step of the pipeline,
together with inputs passed through the pipeline's own fitted preprocessing.

SHAP values are in **log-odds** units and are additive: base value + sum of a row's values =
the model's log-odds for that row. Values are computed per transformed column (spline pieces,
one-hot levels, missing-value indicators) and can be summed back to the original input names.
They describe how the model uses its inputs, not what causes injury.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray

from youthtwin.models.tabular import HazardModel

FloatArray = NDArray[np.float64]
_INDICATOR = "missingindicator_"


@dataclass(frozen=True)
class ShapResult:
    values: pd.DataFrame  # rows x transformed columns, log-odds
    base_value: float
    transformed: pd.DataFrame  # the model's actual inputs after preprocessing

    def log_odds(self) -> FloatArray:
        """Reconstructed log-odds per row (base value + row sum)."""
        return np.asarray(self.base_value + self.values.sum(axis=1).to_numpy(), dtype=np.float64)


def explain(model: HazardModel, frame: pd.DataFrame, background: pd.DataFrame) -> ShapResult:
    """SHAP values of ``model`` for the rows of ``frame``.

    ``background`` (training rows) defines the reference distribution for linear models; tree
    models use the trees' own node statistics.
    """
    prepare = model.pipeline[:-1]
    names = model.transformed_feature_names()
    inputs = np.asarray(prepare.transform(frame[model.features]), dtype=np.float64)
    estimator = model.estimator  # raises if the model has not been fitted
    if hasattr(estimator, "get_booster"):
        explainer = shap.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(inputs), dtype=np.float64)
    else:
        reference = np.asarray(prepare.transform(background[model.features]), dtype=np.float64)
        explainer = shap.LinearExplainer(estimator, reference)
        values = np.asarray(explainer.shap_values(inputs), dtype=np.float64)
    base = float(np.ravel(explainer.expected_value)[0])
    index = frame.index
    return ShapResult(
        values=pd.DataFrame(values, columns=names, index=index),
        base_value=base,
        transformed=pd.DataFrame(inputs, columns=names, index=index),
    )


def input_of(transformed_name: str, features: list[str]) -> str:
    """The original input a transformed column belongs to."""
    name = transformed_name.removeprefix(_INDICATOR)
    matches = [f for f in features if name == f or name.startswith(f + "_")]
    if not matches:
        raise KeyError(f"Cannot map transformed column '{transformed_name}' to an input")
    return max(matches, key=len)  # the longest match wins (e.g. "age" vs "age_band")


def aggregate_to_inputs(result: ShapResult, features: list[str]) -> pd.DataFrame:
    """Sum SHAP values of all columns derived from the same input (rows x inputs)."""
    owner = {column: input_of(column, features) for column in result.values.columns}
    owners = pd.Series(owner)
    return result.values.T.groupby(owners).sum().T


def importance(result: ShapResult, features: list[str]) -> pd.Series:
    """Mean absolute SHAP value per input, largest first."""
    return aggregate_to_inputs(result, features).abs().mean().sort_values(ascending=False)
