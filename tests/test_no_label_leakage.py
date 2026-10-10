"""No label leakage (non-negotiables N1-N3) and athlete-grouped splits.

v2 generated labels from its own rules and then trained on the inputs to those rules, and split
overlapping windows from one athlete across train and test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.eval import splits
from youthtwin.features.person_period import (
    ID_COLUMNS,
    PersonPeriod,
    assert_no_leakage,
    build_person_period,
    predictor_columns,
)
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.sim.cohort import SyntheticCohort

# ----------------------------------------------------------------------------- predictors


def test_predictors_exclude_labels_ids_and_truth(
    person_period: PersonPeriod, cohort: SyntheticCohort
) -> None:
    predictors = set(person_period.predictors)
    assert not [c for c in predictors if c.startswith("y_")]
    assert not predictors & set(ID_COLUMNS) and "synthetic" not in predictors
    truth_columns = {c for table in cohort.truth.values() for c in table.columns}
    assert not predictors & (truth_columns - {"athlete_id", "week_start", "age", "synthetic"})
    assert not [c for c in predictors if c.startswith(("true_", "hazard_", "trait_"))]
    assert set(person_period.labels) == {
        "y_growth_related",
        "y_non_contact_soft_tissue",
        "y_all_non_contact",
    }


@pytest.mark.parametrize(
    "column",
    [
        "y_growth_related",
        "true_aphv",
        "hazard_growth_related",
        "trait_x",
        "outcome_group",
        "time_loss_days",
        "onset_date",
        "at_risk",
    ],
)
def test_leaky_columns_are_refused(column: str, person_period_config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(["age", column], person_period_config)
    assert_no_leakage(["age", "n_prior_injuries", "sex"], person_period_config)


def test_a_model_cannot_be_built_on_a_label(
    person_period: PersonPeriod, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    """N2: the other outcome's label is defined by the same injuries, so it is not an input."""
    frame = person_period.frame
    with pytest.raises(ValueError, match="Leakage"):
        make_hazard_logit(
            frame, ["age", "y_all_non_contact"], hazard_config["hazard_logit"], person_period_config
        )


def test_predictors_do_not_depend_on_the_future(
    cohort: SyntheticCohort, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    """Rewriting everything dated after a cut-off leaves earlier athlete-weeks' inputs unchanged."""
    cutoff = pd.Timestamp("2026-07-06")
    tables = {name: table.copy() for name, table in cohort.tables.items()}
    rng = np.random.default_rng(0)

    late = pd.to_datetime(tables["injuries"]["onset_date"]) >= cutoff
    tables["injuries"] = tables["injuries"][~late]  # all later injuries vanish
    late = pd.to_datetime(tables["anthropometrics"]["date"]) >= cutoff
    tables["anthropometrics"].loc[late, ["stature_1", "stature_2"]] += 9.0
    late = pd.to_datetime(tables["exposure_weekly"]["week_start"]) >= cutoff
    tables["exposure_weekly"].loc[late, "hours_training"] = rng.uniform(0, 30, int(late.sum()))
    sessions = tables["screening_sessions"]
    late_sessions = set(sessions.loc[pd.to_datetime(sessions["date"]) >= cutoff, "id"])
    late = tables["task_features"]["session_id"].isin(late_sessions)
    tables["task_features"].loc[late, "value"] = 999.0

    kwargs = {"config": person_period_config, "hazard_config": hazard_config}
    original = build_person_period(cohort.tables, **kwargs).frame
    altered = build_person_period(tables, **kwargs).frame
    keys = ["athlete_id", "week_start"]
    # Prediction time is the end of the week, so weeks starting >= 7 days before the cut-off
    # are entirely in the unchanged past.
    early = lambda f: f[f["week_start"] <= cutoff - pd.Timedelta(days=7)].set_index(keys)  # noqa: E731
    before, after = early(original), early(altered)
    shared = before.index.intersection(after.index)
    assert len(shared) > 5000
    columns = predictor_columns(original)
    pd.testing.assert_frame_equal(before.loc[shared, columns], after.loc[shared, columns])
    # ...whereas the labels of those weeks do change (their windows reach past the cut-off).
    assert not before.loc[shared, "y_all_non_contact"].equals(
        after.loc[shared, "y_all_non_contact"]
    )


def test_labels_come_only_from_the_injury_table(
    cohort: SyntheticCohort, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> None:
    """N1: with no diagnosed injuries there are no positive labels, whatever the predictors say."""
    tables = dict(cohort.tables)
    tables["injuries"] = tables["injuries"].iloc[0:0]
    frame = build_person_period(
        tables, config=person_period_config, hazard_config=hazard_config
    ).frame
    assert frame[[c for c in frame.columns if c.startswith("y_")]].to_numpy().sum() == 0
    assert (frame["n_prior_injuries"] == 0).all()


# ----------------------------------------------------------------------------- splits


def test_grouped_folds_have_zero_athlete_overlap(person_period: PersonPeriod) -> None:
    frame = person_period.frame
    folds = splits.grouped_stratified_folds(
        frame["athlete_id"], frame["y_growth_related"], n_splits=5, seed=3
    )
    assert len(folds) == 5
    tested = np.zeros(len(frame), dtype=int)
    for fold in folds:
        train_athletes = set(frame.loc[fold.train, "athlete_id"])
        test_athletes = set(frame.loc[fold.test, "athlete_id"])
        assert train_athletes & test_athletes == set()  # split overlap = 0
        assert (fold.train ^ fold.test).all()
        tested += fold.test
    assert (tested == 1).all()  # every athlete-week is tested exactly once
    sizes = [frame.loc[f.test, "athlete_id"].nunique() for f in folds]
    assert max(sizes) - min(sizes) <= 1
    # Stratified: athletes with an event are spread evenly.
    with_event = [
        frame.loc[f.test].groupby("athlete_id")["y_growth_related"].max().sum() for f in folds
    ]
    assert max(with_event) - min(with_event) <= 1


def test_folds_are_reproducible_and_seed_dependent(person_period: PersonPeriod) -> None:
    frame = person_period.frame
    make = lambda seed: splits.grouped_stratified_folds(  # noqa: E731
        frame["athlete_id"], frame["y_growth_related"], n_splits=5, seed=seed
    )
    assert np.array_equal(make(1)[0].test, make(1)[0].test)
    assert not np.array_equal(make(1)[0].test, make(2)[0].test)
    with pytest.raises(ValueError, match="Cannot make"):
        splits.grouped_stratified_folds(
            frame["athlete_id"].head(50), frame["y_growth_related"].head(50), n_splits=5, seed=1
        )


def test_overlap_check_catches_a_row_level_split(person_period: PersonPeriod) -> None:
    """The v2 mistake: splitting rows at random puts each athlete on both sides."""
    frame = person_period.frame
    rng = np.random.default_rng(0)
    test = rng.random(len(frame)) < 0.2
    with pytest.raises(AssertionError, match="both train and test"):
        splits.assert_no_group_overlap(frame["athlete_id"], splits.Split("rows", ~test, test))


def test_nested_folds_never_touch_outer_test_athletes(person_period: PersonPeriod) -> None:
    frame = person_period.frame
    count = 0
    for outer, inner in splits.nested_folds(
        frame["athlete_id"], frame["y_all_non_contact"], n_outer=3, n_inner=3, seed=5
    ):
        outer_test = set(frame.loc[outer.test, "athlete_id"])
        assert len(inner) == 3
        covered = np.zeros(len(frame), dtype=int)
        for fold in inner:
            used = set(frame.loc[fold.train | fold.test, "athlete_id"])
            assert used & outer_test == set()
            assert (
                set(frame.loc[fold.train, "athlete_id"]) & set(frame.loc[fold.test, "athlete_id"])
                == set()
            )
            covered += fold.test
        assert np.array_equal(covered == 1, outer.train)
        count += 1
    assert count == 3


def test_temporal_holdout(person_period: PersonPeriod) -> None:
    frame = person_period.frame
    split = splits.temporal_holdout(frame["week_start"], fraction=0.25, horizon_weeks=4)
    train_weeks, test_weeks = (
        frame.loc[split.train, "week_start"],
        frame.loc[split.test, "week_start"],
    )
    first_test = test_weeks.min()
    assert split.test.mean() == pytest.approx(0.25, abs=0.05)
    # No training outcome window reaches into the held-out period.
    assert (train_weeks + pd.Timedelta(weeks=4) < first_test).all()
    assert not (split.train & split.test).any()
    assert (~split.train & ~split.test).sum() > 0  # the buffer weeks are in neither part
    with pytest.raises(ValueError, match="between 0 and 1"):
        splits.temporal_holdout(frame["week_start"], fraction=1.5, horizon_weeks=4)


def test_leave_one_academy_out(person_period: PersonPeriod) -> None:
    frame = person_period.frame
    folds = splits.leave_one_academy_out(frame["academy_id"], frame["athlete_id"])
    assert [f.name for f in folds] == [
        "academy_academy_1",
        "academy_academy_2",
        "academy_academy_3",
    ]
    for fold in folds:
        assert frame.loc[fold.test, "academy_id"].nunique() == 1
        assert fold.name.removeprefix("academy_") not in set(frame.loc[fold.train, "academy_id"])
