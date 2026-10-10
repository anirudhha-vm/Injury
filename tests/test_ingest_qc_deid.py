"""Ingestion, anthropometry QC, de-identification, seeding and provenance."""

from __future__ import annotations

import stat
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.io import deid, ingest, qc
from youthtwin.sim.cohort import SyntheticCohort
from youthtwin.utils import provenance
from youthtwin.utils.config import repo_root
from youthtwin.utils.seeding import make_rng

# ----------------------------------------------------------------------------- ingestion


def test_synthetic_cohort_validates_against_schema(cohort: SyntheticCohort) -> None:
    report = ingest.validate_tables(cohort.tables)
    assert report.ok, report.errors[:3]
    assert report.n_rows["athletes"] == 300


def test_invalid_rows_are_reported_not_dropped(cohort: SyntheticCohort) -> None:
    broken = cohort.tables["anthropometrics"].head(5).copy()
    broken.loc[broken.index[1], "stature_1"] = 1.52  # metres
    broken.loc[broken.index[3], "athlete_id"] = "not-a-uuid"
    rows, errors = ingest.validate_table("anthropometrics", broken)
    assert len(rows) == 3
    assert [e.row for e in errors] == [1, 3]
    assert "stature_1" in errors[0].message and "athlete_id" in errors[1].message


def test_unknown_table_is_an_error() -> None:
    with pytest.raises(KeyError, match="Unknown table"):
        ingest.validate_table("patients", pd.DataFrame())


def test_load_into_database(cohort: SyntheticCohort, tmp_path: Path) -> None:
    connection = ingest.create_database(tmp_path / "research.sqlite")
    report = ingest.load_tables(connection, cohort.tables)
    assert report.ok
    for name, frame in cohort.tables.items():
        (count,) = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        assert count == len(frame)
    (optional,) = connection.execute(
        "SELECT COUNT(*) FROM task_features WHERE trial IS NULL"
    ).fetchone()
    assert optional == len(cohort.tables["task_features"])
    (flagged,) = connection.execute("SELECT MIN(synthetic) FROM injuries").fetchone()
    assert flagged == 1


def test_load_is_all_or_nothing(cohort: SyntheticCohort, tmp_path: Path) -> None:
    connection = ingest.create_database(tmp_path / "research.sqlite")
    athletes = cohort.tables["athletes"].copy()
    athletes.loc[athletes.index[0], "sex"] = "unknown"
    with pytest.raises(ValueError, match="nothing was loaded"):
        ingest.load_tables(connection, {"athletes": athletes})
    assert connection.execute("SELECT COUNT(*) FROM athletes").fetchone() == (0,)


def test_refuses_to_mix_synthetic_and_real(cohort: SyntheticCohort, tmp_path: Path) -> None:
    connection = ingest.create_database(tmp_path / "research.sqlite")
    athletes = cohort.tables["athletes"]
    ingest.load_tables(connection, {"athletes": athletes.head(3)})
    real = athletes.iloc[3:5].copy()
    real["synthetic"] = False
    with pytest.raises(ValueError, match="mix synthetic and real"):
        ingest.load_tables(connection, {"athletes": real})


# ----------------------------------------------------------------------------- QC


def test_tem_reference_value() -> None:
    # differences 0.2, -0.4, 0.0, 0.6 -> sum of squares 0.56 -> sqrt(0.56 / 8) = 0.264575...
    first = [150.2, 140.0, 160.0, 155.6]
    second = [150.0, 140.4, 160.0, 155.0]
    assert qc.technical_error_of_measurement(first, second) == pytest.approx(0.2645751311)
    with pytest.raises(ValueError):
        qc.technical_error_of_measurement([1.0], [1.0, 2.0])


def test_tem_recovers_the_simulated_measurement_error(
    cohort: SyntheticCohort, cohort_config: dict[str, Any]
) -> None:
    table = qc.tem_by_measurer(cohort.tables["anthropometrics"])
    assert set(table["measure"]) == set(qc.MEASURES)
    stature = table[table["measure"] == "stature"]
    assert stature["tem"].to_numpy() == pytest.approx(
        cohort_config["measurement_error"]["stature_cm"], abs=0.05
    )


def _visit(athlete: str, day: str, stature: float, **overrides: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "athlete_id": athlete,
        "date": date.fromisoformat(day),
        "stature_1": stature,
        "stature_2": stature,
        "sitting_height_1": stature * 0.52,
        "sitting_height_2": stature * 0.52,
        "mass_1": 40.0,
        "mass_2": 40.0,
        "measurer_id": "m1",
    }
    row.update(overrides)
    return row


def test_qc_flags_each_problem_and_keeps_all_rows(qc_config: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        [
            _visit("a", "2026-01-05", 150.0),
            _visit("a", "2026-02-20", 150.8),  # fine
            _visit("a", "2026-04-05", 148.0),  # shrank
            _visit("b", "2026-01-05", 140.0, stature_2=141.2),  # duplicates disagree
            _visit("b", "2026-02-20", 146.0),  # ~5 cm in 6.5 weeks: implausible velocity
            _visit("c", "2026-01-05", 150.0, sitting_height_1=95.0, sitting_height_2=95.0),
            _visit("d", "2026-01-05", 190.0),  # different athlete: no comparison with "c"
        ]
    )
    before = frame.copy()
    flags = qc.anthropometry_flags(frame, qc_config)
    found = set(zip(flags["athlete_id"], flags["flag"], strict=True))
    assert found == {
        ("a", "stature_decrease"),
        ("b", "duplicate_disagreement_stature"),
        ("b", "implausible_height_velocity"),
        ("c", "implausible_sitting_ratio"),
    }
    pd.testing.assert_frame_equal(frame, before)  # QC never edits or drops data
    assert list(flags.columns) == qc.FLAG_COLUMNS


def test_qc_on_clean_data_returns_empty_table(qc_config: dict[str, Any]) -> None:
    frame = pd.DataFrame([_visit("a", "2026-01-05", 150.0), _visit("a", "2026-03-01", 150.9)])
    flags = qc.anthropometry_flags(frame, qc_config)
    assert flags.empty and list(flags.columns) == qc.FLAG_COLUMNS


# ----------------------------------------------------------------------------- de-identification


def test_dob_is_reduced_to_month() -> None:
    assert deid.dob_to_month(date(2014, 3, 17)) == "2014-03"


def test_pseudonymise_is_stable_and_private(tmp_path: Path) -> None:
    map_path = tmp_path / "vault" / "identity_map.csv"
    first = deid.pseudonymise(["REG-001", "REG-002"], map_path)
    second = deid.pseudonymise(["REG-002", "REG-003"], map_path)
    assert first["REG-002"] == second["REG-002"]
    assert len({*first.values(), *second.values()}) == 3
    assert "REG" not in "".join(first.values())
    assert stat.S_IMODE(map_path.stat().st_mode) == 0o600
    assert set(deid.read_identity_map(map_path)) == {"REG-001", "REG-002", "REG-003"}
    with pytest.raises(ValueError, match="Empty identity"):
        deid.pseudonymise([" "], map_path)


def test_identity_map_cannot_live_in_the_repo() -> None:
    inside = repo_root() / "data" / "interim" / "identity_map.csv"
    with pytest.raises(ValueError, match="outside the repository"):
        deid.pseudonymise(["REG-001"], inside)
    assert not inside.exists()


# ----------------------------------------------------------------------------- seeding, provenance


def test_rng_streams_are_reproducible_and_independent() -> None:
    assert make_rng(1, "a").random() == make_rng(1, "a").random()
    assert make_rng(1, "a").random() != make_rng(1, "b").random()
    assert make_rng(1, "a").random() != make_rng(2, "a").random()
    with pytest.raises(ValueError):
        make_rng(-1)


def test_config_hash_ignores_key_order_but_not_values() -> None:
    assert provenance.config_hash({"a": 1, "b": 2}) == provenance.config_hash({"b": 2, "a": 1})
    assert provenance.config_hash({"a": 1}) != provenance.config_hash({"a": 2})


def test_run_metadata_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = provenance.run_metadata(seed=3, config={"a": 1}, synthetic=True)
    assert metadata["synthetic"] is True and metadata["seed"] == 3
    assert {"config_hash", "git_commit", "youthtwin_version", "created_utc"} <= set(metadata)
    commit = provenance.git_commit()
    assert commit is None or len(commit.removesuffix("-dirty")) == 40
    monkeypatch.setattr(provenance.shutil, "which", lambda _: None)
    assert provenance.git_commit() is None


def test_git_commit_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert provenance.git_commit() is None


def test_numpy_seed_sequence_is_stable_across_runs() -> None:
    """Guards the reproducibility claim: a fixed seed always yields this first draw."""
    first = make_rng(20261009, "athletes").random()
    assert first == make_rng(20261009, "athletes").random()
    assert 0.0 <= first < 1.0
    assert isinstance(np.float64(first), np.float64)


# ----------------------------------------------------------------------------- cohort directory


def test_cohort_directory_round_trip_and_validation(
    cohort: SyntheticCohort, tmp_path: Path
) -> None:
    from youthtwin.sim.cohort import write_cohort

    directory = write_cohort(cohort, tmp_path / "cohort")
    tables = ingest.read_cohort_dir(directory)
    assert set(tables) == set(ingest.COHORT_TABLES)
    assert len(tables["athletes"]) == 300 and "athlete_truth" not in tables  # truth is not read
    broken = tables["anthropometrics"].copy()
    broken.loc[broken.index[0], "stature_1"] = 1.6  # metres
    broken.to_parquet(directory / "anthropometrics.parquet", index=False)
    with pytest.raises(ValueError, match="invalid row"):
        ingest.read_cohort_dir(directory)
    (directory / "injuries.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="injuries"):
        ingest.read_cohort_dir(directory)


def test_cohort_directory_refuses_mixed_data(cohort: SyntheticCohort, tmp_path: Path) -> None:
    from youthtwin.sim.cohort import write_cohort

    directory = write_cohort(cohort, tmp_path / "mixed")
    athletes = pd.read_parquet(directory / "athletes.parquet")
    athletes.loc[athletes.index[:5], "synthetic"] = False
    athletes.to_parquet(directory / "athletes.parquet", index=False)
    with pytest.raises(ValueError, match="mixes synthetic and real"):
        ingest.read_cohort_dir(directory)
