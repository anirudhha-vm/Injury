"""One interface from table to prediction for every model kind (v2 regression).

v2 had call-signature mismatches between modules and used a deprecated XGBoost argument. Every
tabular model here is a ``HazardModel`` with the same ``fit(frame, label)`` /
``predict_proba(frame)`` contract, and explanation utilities accept any of them.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from youthtwin.eval.predictions import out_of_fold_predictions
from youthtwin.eval.splits import grouped_stratified_folds
from youthtwin.explain import counterfactual, integrated_gradients
from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.baselines import make_baseline
from youthtwin.models.risk import hazard_xgb
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.risk.hazard_xgb import make_hazard_xgb
from youthtwin.models.tabular import HazardModel
from youthtwin.utils.config import load_config

LABEL = "y_all_non_contact"
FEATURES = ["age", "moore_maturity_offset", "hours_total", "ewma_hours", "sex"]


@pytest.fixture(scope="module")
def models(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> dict[str, HazardModel]:
    frame = person_period.frame
    quick = {**hazard_config["hazard_logit"], "max_iter": 300}
    return {
        "age_only": make_baseline(
            "age_only", frame, load_config("models/baselines.yaml"), person_period_config
        ),
        "hazard_logit": make_hazard_logit(frame, FEATURES, quick, person_period_config),
        "hazard_xgb": make_hazard_xgb(
            frame, FEATURES, hazard_config["hazard_xgb"], person_period_config, seed=0
        ),
    }


def test_every_model_honours_the_same_contract(
    models: dict[str, HazardModel], person_period: PersonPeriod
) -> None:
    frame = person_period.frame
    train, test = frame.iloc[:8000], frame.iloc[8000:8500]
    for name, model in models.items():
        assert isinstance(model, HazardModel) and model.name == name
        assert list(inspect.signature(model.fit).parameters) == ["frame", "label"]
        assert list(inspect.signature(model.predict_proba).parameters) == ["frame"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)  # no deprecated arguments
            warnings.simplefilter("error", FutureWarning)
            fitted = model.fit(train, LABEL)
        assert fitted is model
        risk = model.predict_proba(test)
        assert isinstance(risk, np.ndarray) and risk.dtype == np.float64
        assert risk.shape == (len(test),) and ((risk >= 0) & (risk <= 1)).all()
        # Extra columns in the table are ignored; column order does not matter.
        shuffled = test[list(reversed(test.columns))].assign(unrelated=1.0)
        np.testing.assert_array_equal(model.predict_proba(shuffled), risk)
        with pytest.raises(KeyError):
            model.predict_proba(test.drop(columns=model.features[0]))


def test_deprecated_xgboost_argument_is_not_used() -> None:
    assert "use_label_encoder" not in inspect.getsource(hazard_xgb)


def test_models_plug_into_out_of_fold_prediction(
    person_period: PersonPeriod, person_period_config: dict[str, Any]
) -> None:
    frame = person_period.frame
    baselines = load_config("models/baselines.yaml")
    folds = grouped_stratified_folds(frame["athlete_id"], frame[LABEL], n_splits=2, seed=0)
    table = out_of_fold_predictions(
        frame,
        LABEL,
        {"age_only": lambda f: make_baseline("age_only", f, baselines, person_period_config)},
        folds,
    )
    assert isinstance(table, pd.DataFrame) and table["risk_age_only"].between(0, 1).all()


def test_what_if_is_labelled_as_sensitivity_not_causation(
    models: dict[str, HazardModel], person_period: PersonPeriod
) -> None:
    frame = person_period.frame.iloc[8000:8400]
    model = models["hazard_logit"].fit(person_period.frame.iloc[:8000], LABEL)
    settings = load_config("eval/metrics.yaml")["counterfactual"]
    changed = counterfactual.scale_columns(
        frame, settings["exposure_columns"], settings["exposure_scale"]
    )
    np.testing.assert_allclose(changed["hours_total"], 0.8 * frame["hours_total"])
    assert changed["age"].equals(frame["age"])  # nothing else is touched
    result = counterfactual.what_if(model, frame, changed)
    np.testing.assert_allclose(result["risk_before"], model.predict_proba(frame))
    np.testing.assert_allclose(result["risk_change"], result["risk_after"] - result["risk_before"])
    assert "not an estimate of a causal effect" in result.attrs["disclaimer"]
    summary = counterfactual.summarise_what_if(result)
    assert "causal" in str(summary["interpretation"])
    assert summary["mean_change"] == pytest.approx(result["risk_change"].mean())
    unchanged = counterfactual.what_if(model, frame, frame)
    assert (unchanged["risk_change"] == 0).all()
    with pytest.raises(KeyError, match="not in the table"):
        counterfactual.scale_columns(frame, ["training_load"], 0.8)
    with pytest.raises(ValueError, match="same rows"):
        counterfactual.what_if(model, frame, changed.iloc[:10])


def test_integrated_gradients_completeness() -> None:
    """Attributions sum to output(input) - output(baseline)."""
    torch.manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(12, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )
    function = lambda x: network(x).squeeze(-1)  # noqa: E731
    inputs = torch.randn(5, 3, 4)
    attributions = integrated_gradients.integrated_gradients(function, inputs, steps=200)
    assert attributions.shape == inputs.shape
    difference = function(inputs) - function(torch.zeros_like(inputs))
    torch.testing.assert_close(attributions.sum(dim=(1, 2)), difference, atol=1e-3, rtol=1e-3)
    # Linear function: attribution is exactly weight * (input - baseline), with any step count.
    weight = torch.arange(12.0).reshape(3, 4)
    linear = lambda x: (x * weight).sum(dim=(1, 2))  # noqa: E731
    baseline = torch.ones_like(inputs)
    exact = integrated_gradients.integrated_gradients(linear, inputs, baseline, steps=1)
    torch.testing.assert_close(exact, weight * (inputs - baseline))
    with pytest.raises(ValueError, match="at least 1"):
        integrated_gradients.integrated_gradients(linear, inputs, steps=0)
    per_trial = torch.ones(2, 3, 4, 5)
    pooled = integrated_gradients.channel_time_attribution(
        per_trial, torch.tensor([[1.0, 1, 0], [1, 0, 0]])
    )
    assert pooled.shape == (2, 4, 5) and pooled[0, 0, 0] == 2 and pooled[1, 0, 0] == 1
