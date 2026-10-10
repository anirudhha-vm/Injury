"""The confirmatory model takes only as many pre-specified predictors as the events support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.confirmatory import (
    athlete_level_counts,
    confirmatory_predictors,
    make_confirmatory,
)


def _table(n_athletes: int, proportion: float, columns: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    athlete = np.repeat(np.arange(n_athletes), 10)
    injured = np.arange(n_athletes) < round(proportion * n_athletes)
    outcome = np.zeros(athlete.size, dtype=int)
    outcome[np.flatnonzero(injured[athlete])[::10]] = 1  # one event week per injured athlete
    data = {c: rng.normal(size=athlete.size) for c in columns}
    return pd.DataFrame({"athlete_id": athlete.astype(str), "y_growth_related": outcome, **data})


def test_counts_are_per_athlete_not_per_week() -> None:
    frame = _table(200, 0.10, ["obs_leg_length_velocity"])
    assert athlete_level_counts(frame, "y_growth_related") == (200, pytest.approx(0.10))
    assert frame["y_growth_related"].mean() == pytest.approx(0.01)  # the per-week rate is not used


@pytest.mark.parametrize(
    ("n_athletes", "proportion", "expected"),
    [(300, 0.04, 1), (300, 0.15, 2), (1000, 0.04, 4), (2000, 0.15, 5), (40, 0.05, 0)],
)
def test_number_of_predictors_follows_the_events(
    hazard_config: dict[str, Any], n_athletes: int, proportion: float, expected: int
) -> None:
    config = hazard_config["confirmatory"]
    columns = list(config["candidate_predictors_in_order"])
    frame = _table(n_athletes, proportion, columns)
    chosen = confirmatory_predictors(frame, "y_growth_related", config)
    assert len(chosen) == expected  # capped at max_parameters for the largest cohort
    assert chosen == columns[:expected]  # always the leading entries, in the fixed order


def test_absent_predictors_are_skipped_not_replaced(hazard_config: dict[str, Any]) -> None:
    config = hazard_config["confirmatory"]
    available = ["obs_leg_length_velocity", "moore_maturity_offset", "ewma_hours"]
    frame = _table(1000, 0.08, [*available, "an_attractive_extra_feature"])
    chosen = confirmatory_predictors(frame, "y_growth_related", config)
    assert chosen == available  # in list order; the extra column can never be selected
    no_events = frame.assign(y_growth_related=0)
    assert confirmatory_predictors(no_events, "y_growth_related", config) == []


def test_confirmatory_model_on_the_synthetic_cohort(
    person_period: PersonPeriod, hazard_config: dict[str, Any], person_period_config: dict[str, Any]
) -> None:
    frame, config = person_period.frame, hazard_config["confirmatory"]
    chosen = confirmatory_predictors(frame, "y_growth_related", config)
    assert 1 <= len(chosen) <= 3  # 300 athletes: a very small model
    assert chosen[0] == "obs_leg_length_velocity"
    model = make_confirmatory(frame, chosen, config, person_period_config).fit(
        frame, "y_growth_related"
    )
    assert model.estimator.coef_.size == len(chosen)  # exactly one parameter per predictor
    risk = model.predict_proba(frame)
    assert risk.mean() == pytest.approx(frame["y_growth_related"].mean(), abs=0.002)
    with pytest.raises(ValueError, match="does not support"):
        make_confirmatory(frame, [], config, person_period_config)
    with pytest.raises(ValueError, match="must be numeric"):
        make_confirmatory(frame, ["sex"], config, person_period_config)
