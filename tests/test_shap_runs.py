"""SHAP runs on fitted models and is additive (v2 regression).

v2 passed ``CalibratedClassifierCV(...).estimator`` to SHAP: an unfitted template.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.explain import shap_utils
from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.risk.hazard_xgb import make_hazard_xgb

LABEL = "y_all_non_contact"
FEATURES = [
    "age",
    "moore_maturity_offset",
    "obs_leg_length_velocity",
    "hours_total",
    "weeks_since_last_injury",
    "sex",
]


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


@pytest.fixture(scope="module")
def tables(person_period: PersonPeriod) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = person_period.frame
    return frame.iloc[:9000], frame.iloc[9000:9300]


def test_shap_on_the_fitted_booster_is_additive(
    tables: tuple[pd.DataFrame, pd.DataFrame],
    person_period_config: dict[str, Any],
    hazard_config: dict[str, Any],
) -> None:
    train, rows = tables
    model = make_hazard_xgb(
        train, FEATURES, hazard_config["hazard_xgb"], person_period_config, seed=0
    )
    model.fit(train, LABEL)
    result = shap_utils.explain(model, rows, train)
    assert result.values.shape == (len(rows), len(model.transformed_feature_names()))
    assert np.isfinite(result.values.to_numpy()).all()
    # Additivity: base value + a row's SHAP values = the model's log-odds for that row.
    np.testing.assert_allclose(result.log_odds(), _logit(model.predict_proba(rows)), atol=1e-4)
    assert result.values.abs().to_numpy().sum() > 0


def test_shap_on_the_fitted_logistic_model_is_additive(
    tables: tuple[pd.DataFrame, pd.DataFrame],
    person_period_config: dict[str, Any],
    hazard_config: dict[str, Any],
) -> None:
    train, rows = tables
    config = {**hazard_config["hazard_logit"], "max_iter": 500}
    model = make_hazard_logit(train, FEATURES, config, person_period_config).fit(train, LABEL)
    result = shap_utils.explain(model, rows, train)
    np.testing.assert_allclose(result.log_odds(), _logit(model.predict_proba(rows)), atol=1e-6)


def test_an_unfitted_model_cannot_be_explained(
    tables: tuple[pd.DataFrame, pd.DataFrame],
    person_period_config: dict[str, Any],
    hazard_config: dict[str, Any],
) -> None:
    train, rows = tables
    unfitted = make_hazard_xgb(
        train, FEATURES, hazard_config["hazard_xgb"], person_period_config, seed=0
    )
    with pytest.raises(Exception, match="fitted"):
        shap_utils.explain(unfitted, rows, train)


def test_values_aggregate_back_to_the_original_inputs(
    tables: tuple[pd.DataFrame, pd.DataFrame],
    person_period_config: dict[str, Any],
    hazard_config: dict[str, Any],
) -> None:
    train, rows = tables
    config = {**hazard_config["hazard_logit"], "max_iter": 500}
    model = make_hazard_logit(train, FEATURES, config, person_period_config).fit(train, LABEL)
    result = shap_utils.explain(model, rows, train)
    per_input = shap_utils.aggregate_to_inputs(result, model.features)
    assert set(per_input.columns) == set(FEATURES)  # spline pieces, one-hot levels, indicators
    np.testing.assert_allclose(per_input.sum(axis=1), result.values.sum(axis=1))
    ranking = shap_utils.importance(result, model.features)
    assert ranking.index.tolist() == ranking.sort_values(ascending=False).index.tolist()
    assert (ranking >= 0).all()


def test_transformed_columns_map_to_the_right_input() -> None:
    features = ["age", "age_band_code", "sex", "weeks_since_last_injury"]
    assert shap_utils.input_of("age", features) == "age"
    assert shap_utils.input_of("age_spline2", features) == "age"
    assert shap_utils.input_of("age_band_code", features) == "age_band_code"  # longest match
    assert shap_utils.input_of("sex_male", features) == "sex"
    assert shap_utils.input_of("missingindicator_weeks_since_last_injury", features) == (
        "weeks_since_last_injury"
    )
    with pytest.raises(KeyError, match="Cannot map"):
        shap_utils.input_of("height", features)
