"""Baseline and hazard models: they fit, predict probabilities, and on synthetic data recover the
directions of the effects that were simulated.

Recovering a direction on synthetic data checks the pipeline (table building, preprocessing,
fitting); it is not evidence about real injury risk.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from youthtwin.eval import splits
from youthtwin.features.person_period import PersonPeriod, build_person_period
from youthtwin.models.baselines import baseline_features, make_baseline
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.risk.hazard_xgb import make_hazard_xgb
from youthtwin.models.tabular import HazardModel, partial_effect
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config

LOAD = "scr__drop_vertical_jump__both__knee_extensor_load_proxy"
GROWTH, SOFT = "y_growth_related", "y_non_contact_soft_tissue"
# One representative per group of strongly correlated inputs. Height and leg-length velocity
# correlate at about 0.9, as do weekly hours and the change in hours; with both of a pair in the
# model, the sign of either one alone is not determined by the data. (The growth twin's
# collinearity check in M6 addresses the same issue.)
DIRECTION_FEATURES = [
    "age",
    "moore_maturity_offset",
    "obs_leg_length_velocity",
    LOAD,
    "hours_total",
    "n_prior_injuries",
    "sex",
]


@pytest.fixture(scope="module")
def baseline_config() -> dict[str, Any]:
    return load_config("models/baselines.yaml")


@pytest.fixture(scope="module")
def large(person_period_config: dict[str, Any], hazard_config: dict[str, Any]) -> PersonPeriod:
    """A larger synthetic cohort so that effect directions are not decided by chance."""
    config = load_config("sim/cohort.yaml")
    config.update(n_athletes=1200, seed=7)
    cohort = generate_cohort(config)
    return build_person_period(
        cohort.tables, config=person_period_config, hazard_config=hazard_config
    )


def _auc(outcome: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random event row scores above a random non-event row."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    positives = outcome == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ----------------------------------------------------------------------------- basics


def test_models_fit_and_predict_probabilities(
    person_period: PersonPeriod,
    person_period_config: dict[str, Any],
    hazard_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    frame = person_period.frame
    fold = splits.grouped_stratified_folds(
        frame["athlete_id"], frame["y_all_non_contact"], n_splits=4, seed=1
    )[0]
    train, test = frame[fold.train], frame[fold.test]
    features = person_period.predictors
    models: list[HazardModel] = [
        make_baseline(name, frame, baseline_config, person_period_config)
        for name in baseline_config["baselines"]
    ]
    quick = {**hazard_config["hazard_logit"], "max_iter": 300}
    models.append(make_hazard_logit(frame, features, quick, person_period_config))
    models.append(
        make_hazard_xgb(frame, features, hazard_config["hazard_xgb"], person_period_config, seed=1)
    )
    for model in models:
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(test)
        model.fit(train, "y_all_non_contact")
        risk = model.predict_proba(test)
        assert risk.shape == (len(test),) and np.isfinite(risk).all()
        assert risk.min() >= 0.0 and risk.max() <= 1.0
        # Unweighted likelihood: mean predicted risk is near the training prevalence.
        assert risk.mean() == pytest.approx(train["y_all_non_contact"].mean(), abs=0.02)
        assert model.label == "y_all_non_contact"
        assert len(model.transformed_feature_names()) >= len(model.features)


def test_baseline_definitions(
    person_period: PersonPeriod,
    baseline_config: dict[str, Any],
    person_period_config: dict[str, Any],
) -> None:
    frame = person_period.frame
    assert baseline_features("age_only", frame, baseline_config) == ["age"]
    clinical = baseline_features("maturity_growth_velocity", frame, baseline_config)
    assert clinical == [
        "moore_maturity_offset",
        "obs_height_velocity",
        "obs_leg_length_velocity",
        "obs_mass_velocity",
    ]
    screening = baseline_features("screening_only", frame, baseline_config)
    assert len(screening) == 3 and all(c.startswith("scr__") for c in screening)
    with pytest.raises(KeyError, match="Unknown baseline"):
        make_baseline("vibes", frame, baseline_config, person_period_config)
    with pytest.raises(ValueError, match="no input columns"):
        baseline_features("screening_only", frame[["age", "sex"]], baseline_config)


def test_single_class_training_data_is_an_error(
    person_period: PersonPeriod,
    baseline_config: dict[str, Any],
    person_period_config: dict[str, Any],
) -> None:
    frame = person_period.frame
    never = frame[frame["y_growth_related"] == 0]
    model = make_baseline("age_only", frame, baseline_config, person_period_config)
    with pytest.raises(ValueError, match="single class"):
        model.fit(never, "y_growth_related")
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.estimator


def test_xgb_uses_the_fitted_booster_and_handles_missing_values(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    """v2 gave SHAP an unfitted template estimator; the exposed estimator here is the fitted one."""
    frame = person_period.frame
    features = ["age", "obs_leg_length_velocity", "weeks_since_last_injury", "sex"]
    assert frame["weeks_since_last_injury"].isna().mean() > 0.5  # mostly missing: no imputation
    model = make_hazard_xgb(
        frame, features, hazard_config["hazard_xgb"], person_period_config, seed=3
    )
    model.fit(frame, "y_all_non_contact")
    booster = model.estimator.get_booster()
    assert booster.num_boosted_rounds() == hazard_config["hazard_xgb"]["n_estimators"]
    again = make_hazard_xgb(
        frame, features, hazard_config["hazard_xgb"], person_period_config, seed=3
    )
    np.testing.assert_allclose(
        model.predict_proba(frame), again.fit(frame, "y_all_non_contact").predict_proba(frame)
    )


def test_monotone_constraints(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    frame = person_period.frame
    config = copy.deepcopy(hazard_config["hazard_xgb"])
    config["monotone_constraints"] = {"hours_total": 1}
    model = make_hazard_xgb(
        frame, ["age", "hours_total", "sex"], config, person_period_config, seed=1
    )
    model.fit(frame, "y_all_non_contact")
    grid = np.linspace(frame["hours_total"].min(), frame["hours_total"].max(), 40)
    template = frame.head(1)
    risk = np.array([model.predict_proba(template.assign(hours_total=h))[0] for h in grid])
    assert np.all(np.diff(risk) >= -1e-9)
    config["monotone_constraints"] = {"sex": 1}
    with pytest.raises(ValueError, match="not numeric inputs"):
        make_hazard_xgb(frame, ["age", "sex"], config, person_period_config, seed=1)


# ----------------------------------------------------------------------------- effect directions


@pytest.fixture(scope="module")
def fitted(
    large: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> dict[tuple[str, str], tuple[HazardModel, Any]]:
    frame = large.frame
    out: dict[tuple[str, str], tuple[HazardModel, Any]] = {}
    for label in (GROWTH, SOFT):
        fold = splits.grouped_stratified_folds(
            frame["athlete_id"], frame[label], n_splits=4, seed=2
        )[0]
        train, test = frame[fold.train], frame[fold.test]
        logit = make_hazard_logit(
            frame, DIRECTION_FEATURES, hazard_config["hazard_logit"], person_period_config
        )
        boosted = make_hazard_xgb(
            frame, DIRECTION_FEATURES, hazard_config["hazard_xgb"], person_period_config, seed=2
        )
        out[("logit", label)] = (logit.fit(train, label), test)
        out[("xgb", label)] = (boosted.fit(train, label), test)
    return out


@pytest.mark.parametrize("kind", ["logit", "xgb"])
def test_growth_related_hazard_recovers_simulated_directions(
    fitted: dict[tuple[str, str], tuple[HazardModel, Any]], kind: str
) -> None:
    """Simulated: risk rises with leg-length velocity and with the landing load proxy."""
    model, test = fitted[(kind, GROWTH)]
    assert partial_effect(model, test, "obs_leg_length_velocity") > 0
    assert partial_effect(model, test, LOAD) > 0
    assert _auc(test[GROWTH].to_numpy(), model.predict_proba(test)) > 0.55  # held-out athletes


@pytest.mark.parametrize("kind", ["logit", "xgb"])
def test_soft_tissue_hazard_recovers_simulated_directions(
    fitted: dict[tuple[str, str], tuple[HazardModel, Any]], kind: str
) -> None:
    """Simulated: soft-tissue risk is higher after PHV and does not depend on the load proxy.

    The simulated exposure effect acts in the same week as the injury and weekly hours are
    independent from week to week, so this week's hours cannot predict next month's injuries;
    no exposure direction is asserted here (a limitation of the simulation, see M5_done.md).
    """
    model, test = fitted[(kind, SOFT)]
    assert partial_effect(model, test, "moore_maturity_offset") > 0
    growth_model, _ = fitted[(kind, GROWTH)]
    # The load proxy raises growth-related risk and does so more than soft-tissue risk (N6).
    assert partial_effect(growth_model, test, LOAD) > partial_effect(model, test, LOAD)
    assert _auc(test[SOFT].to_numpy(), model.predict_proba(test)) > 0.55


def test_outcomes_need_separate_models(
    fitted: dict[tuple[str, str], tuple[HazardModel, Any]],
) -> None:
    """N6: growth velocity and maturity act differently on the two injury types."""
    growth_model, test = fitted[("logit", GROWTH)]
    soft_model, _ = fitted[("logit", SOFT)]
    assert partial_effect(growth_model, test, "obs_leg_length_velocity") > partial_effect(
        soft_model, test, "obs_leg_length_velocity"
    )
    assert partial_effect(soft_model, test, "moore_maturity_offset") > partial_effect(
        growth_model, test, "moore_maturity_offset"
    )


def test_partial_effect_of_a_binary_feature(
    person_period: PersonPeriod,
    person_period_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    frame = person_period.frame.assign(flag=lambda f: (f["n_prior_injuries"] > 0).astype(float))
    config = {**baseline_config, "baselines": {"flag_only": {"predictors": ["flag", "age"]}}}
    model = make_baseline("flag_only", frame, config, person_period_config)
    model.fit(frame, "y_all_non_contact")
    effect = partial_effect(model, frame, "flag")
    by_hand = model.predict_proba(frame.assign(flag=1.0)) - model.predict_proba(
        frame.assign(flag=0.0)
    )
    assert effect == pytest.approx(float(by_hand.mean()))
