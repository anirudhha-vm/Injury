"""Preprocessing is fitted on training rows only (non-negotiable N4).

v2 fitted a RobustScaler separately on every clip, which erased differences between athletes
and let test data shape the features. Here every statistic lives inside the model pipeline and
is learned in ``fit``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from youthtwin.eval import splits
from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.baselines import make_baseline
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.tabular import NaturalSplineTransformer, preprocessor
from youthtwin.utils.config import load_config

LABEL = "y_all_non_contact"
FEATURES = ["age", "moore_maturity_offset", "obs_leg_length_velocity", "hours_total", "sex"]


def _fold(person_period: PersonPeriod) -> splits.Split:
    frame = person_period.frame
    return splits.grouped_stratified_folds(frame["athlete_id"], frame[LABEL], n_splits=4, seed=1)[0]


def _fitted_statistics(model: Any) -> dict[str, np.ndarray]:
    prepare = model.pipeline.named_steps["prepare"]
    spline = prepare.named_transformers_["spline"]
    numeric = prepare.named_transformers_["numeric"]
    return {
        "spline_median": spline.named_steps["impute"].statistics_,
        "knots": np.array([k for knots in spline.named_steps["spline"].knots_ for k in knots]),
        "spline_mean": spline.named_steps["scale"].mean_,
        "numeric_median": numeric.named_steps["impute"].statistics_,
        "numeric_scale": numeric.named_steps["scale"].scale_,
        "coefficients": model.estimator.coef_.ravel(),
    }


def test_statistics_equal_training_fold_statistics(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    frame, fold = person_period.frame, _fold(person_period)
    train = frame[fold.train]
    model = make_hazard_logit(frame, FEATURES, hazard_config["hazard_logit"], person_period_config)
    model.fit(train, LABEL)
    stats = _fitted_statistics(model)
    # hours_total is the only plain numeric input: its imputation value is the TRAINING median.
    assert stats["numeric_median"][0] == pytest.approx(train["hours_total"].median())
    assert stats["numeric_median"][0] != pytest.approx(frame["hours_total"].median(), abs=1e-12)
    spline_columns = ["age", "moore_maturity_offset", "obs_leg_length_velocity"]
    np.testing.assert_allclose(stats["spline_median"], train[spline_columns].median().to_numpy())
    # Spline knots are training quantiles of the (imputed) training column.
    age = train["age"].to_numpy()
    np.testing.assert_allclose(stats["knots"][:4], np.quantile(age, np.linspace(0.05, 0.95, 4)))


def test_changing_held_out_rows_cannot_change_the_fitted_model(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    frame, fold = person_period.frame, _fold(person_period)
    corrupted = frame.copy()
    rng = np.random.default_rng(0)
    for column in ("age", "moore_maturity_offset", "obs_leg_length_velocity", "hours_total"):
        corrupted.loc[fold.test, column] = rng.normal(500.0, 300.0, int(fold.test.sum()))
    corrupted.loc[fold.test, LABEL] = 1

    def fit(table: Any) -> dict[str, np.ndarray]:
        model = make_hazard_logit(
            table, FEATURES, hazard_config["hazard_logit"], person_period_config
        )
        return _fitted_statistics(model.fit(table[fold.train], LABEL))

    clean, dirty = fit(frame), fit(corrupted)
    for name in clean:
        np.testing.assert_array_equal(clean[name], dirty[name], err_msg=name)


def test_fitting_on_everything_gives_different_statistics(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    """The check above is not vacuous: including the held-out rows does move the statistics."""
    frame, fold = person_period.frame, _fold(person_period)
    config = hazard_config["hazard_logit"]
    on_train = _fitted_statistics(
        make_hazard_logit(frame, FEATURES, config, person_period_config).fit(
            frame[fold.train], LABEL
        )
    )
    on_all = _fitted_statistics(
        make_hazard_logit(frame, FEATURES, config, person_period_config).fit(frame, LABEL)
    )
    assert not np.allclose(on_train["spline_mean"], on_all["spline_mean"])
    assert not np.allclose(on_train["knots"], on_all["knots"])


def test_between_athlete_differences_survive_scaling(
    person_period: PersonPeriod, person_period_config: dict[str, Any]
) -> None:
    """One scaler for the whole training fold keeps tall and short athletes apart (v2 did not)."""
    frame, fold = person_period.frame, _fold(person_period)
    train = frame[fold.train].dropna(subset=["stature"])
    scaled = preprocessor(["stature"], []).fit_transform(train[["stature"]])[:, 0]
    per_athlete = train.assign(z=scaled).groupby("athlete_id")["z"].mean()
    assert per_athlete.std() > 0.8  # athletes differ after scaling
    raw_means = train.groupby("athlete_id")["stature"].mean()
    assert np.corrcoef(per_athlete, raw_means)[0, 1] > 0.999


def test_baseline_scaler_is_also_fold_scoped(
    person_period: PersonPeriod, person_period_config: dict[str, Any]
) -> None:
    frame, fold = person_period.frame, _fold(person_period)
    model = make_baseline(
        "age_only", frame, load_config("models/baselines.yaml"), person_period_config
    )
    model.fit(frame[fold.train], LABEL)
    scaler = (
        model.pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"]
    )
    assert scaler.mean_[0] == pytest.approx(frame.loc[fold.train, "age"].mean())


def test_spline_transformer() -> None:
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.normal(size=500), rng.integers(0, 2, 500).astype(float)])
    transformer = NaturalSplineTransformer(n_knots=5).fit(data)
    assert len(transformer.knots_[0]) == 5 and transformer.knots_[1] is None  # binary: linear only
    out = transformer.transform(data)
    assert out.shape == (500, 1 + 3 + 1)
    np.testing.assert_array_equal(out[:, 0], data[:, 0])
    names = transformer.get_feature_names_out(["x", "flag"])
    assert list(names) == ["x", "x_spline1", "x_spline2", "x_spline3", "flag"]
    # New data are expanded with the fitted knots, not re-fitted ones.
    shifted = transformer.transform(data + 100.0)
    assert shifted.shape == out.shape and transformer.knots_[0][0] < 0
