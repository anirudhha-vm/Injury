"""Synthetic cohort assembly: reproducibility, size, labelling, truth separation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from youthtwin.io.schema import TABLES
from youthtwin.sim.cohort import (
    METADATA_FILE,
    TRUTH_DIR,
    SyntheticCohort,
    generate_cohort,
    write_cohort,
)

TRUTH_ONLY_COLUMNS = {"true_aphv", "theta", "true_maturity_offset", "true_load_proxy", "at_risk"}


def _small(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    small = copy.deepcopy(config)
    small.update({"n_athletes": 25, "follow_up_weeks": 26})
    small.update(overrides)
    return small


def test_default_cohort_is_300_athletes_for_12_months(cohort: SyntheticCohort) -> None:
    assert len(cohort.tables["athletes"]) == 300
    assert cohort.tables["athletes"]["id"].is_unique
    weeks = cohort.tables["exposure_weekly"].groupby("athlete_id").size()
    assert (weeks == 52).all() and len(weeks) == 300
    visits = cohort.tables["anthropometrics"].groupby("athlete_id").size()
    assert len(visits) == 300 and visits.max() <= 9  # every 6-8 weeks over a year


def test_same_seed_gives_identical_cohort(cohort_config: dict[str, Any]) -> None:
    first, second = (generate_cohort(_small(cohort_config)) for _ in range(2))
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name], second.tables[name])
    for name in first.truth:
        pd.testing.assert_frame_equal(first.truth[name], second.truth[name])
    assert first.metadata["config_hash"] == second.metadata["config_hash"]


def test_different_seed_gives_different_cohort(cohort_config: dict[str, Any]) -> None:
    first = generate_cohort(_small(cohort_config, seed=1))
    second = generate_cohort(_small(cohort_config, seed=2))
    assert not first.tables["athletes"]["id"].equals(second.tables["athletes"]["id"])
    assert not first.truth["athlete_truth"]["true_aphv"].equals(
        second.truth["athlete_truth"]["true_aphv"]
    )
    assert first.metadata["config_hash"] != second.metadata["config_hash"]


def test_everything_is_flagged_synthetic(cohort: SyntheticCohort) -> None:
    assert cohort.metadata["synthetic"] is True
    assert cohort.metadata["seed"] == 20261009
    for frame in (*cohort.tables.values(), *cohort.truth.values()):
        assert frame["synthetic"].all()


def test_config_must_declare_synthetic(cohort_config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="synthetic"):
        generate_cohort(_small(cohort_config, synthetic=False))


def test_observable_tables_contain_no_truth(cohort: SyntheticCohort) -> None:
    """Truth must not be reachable from the tables the modelling pipeline reads (N1, N2)."""
    for name, frame in cohort.tables.items():
        assert set(frame.columns) == set(TABLES[name].model_fields), name
        assert not TRUTH_ONLY_COLUMNS & set(frame.columns)
        assert not [c for c in frame.columns if c.startswith(("hazard_", "true_", "trait_"))]


def test_ages_stay_in_study_range(cohort: SyntheticCohort) -> None:
    age = cohort.truth["athlete_weeks_truth"]["age"]
    assert age.min() >= 8.0 and age.max() <= 18.0
    assert cohort.tables["athletes"]["dob_month"].str.fullmatch(r"\d{4}-\d{2}").all()


def test_write_cohort_round_trip(cohort_config: dict[str, Any], tmp_path: Path) -> None:
    small = generate_cohort(_small(cohort_config))
    out = write_cohort(small, tmp_path / "cohort")
    metadata = json.loads((out / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["synthetic"] is True
    assert metadata["n_rows"]["athletes"] == 25
    for name, frame in small.tables.items():
        loaded = pd.read_parquet(out / f"{name}.parquet")
        assert len(loaded) == len(frame) and loaded["synthetic"].all()
    assert sorted(p.stem for p in (out / TRUTH_DIR).glob("*.parquet")) == sorted(small.truth)
    assert not list(out.glob("*truth*.parquet"))  # truth only under truth/


def test_write_refuses_unflagged_cohort(cohort_config: dict[str, Any], tmp_path: Path) -> None:
    small = generate_cohort(_small(cohort_config))
    unflagged = SyntheticCohort(small.tables, small.truth, {**small.metadata, "synthetic": False})
    with pytest.raises(ValueError, match="not flagged synthetic"):
        write_cohort(unflagged, tmp_path / "bad")
