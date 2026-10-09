"""Schema: the SQL DDL and the pydantic models stay in step, and enforce the privacy rules."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from youthtwin.io.ingest import DDL_PATH
from youthtwin.io.schema import TABLES, Athlete, Injury, OstrcWeekly, Trial
from youthtwin.utils.config import repo_root

ID = str(uuid4())


@pytest.fixture(scope="module")
def database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript((repo_root() / DDL_PATH).read_text(encoding="utf-8"))
    return connection


def test_ddl_tables_match_models(database: sqlite3.Connection) -> None:
    names = {
        row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert names == set(TABLES)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_ddl_columns_match_model_fields(database: sqlite3.Connection, table: str) -> None:
    columns = [row[1] for row in database.execute(f"PRAGMA table_info({table})")]
    assert sorted(columns) == sorted(TABLES[table].model_fields)


def _athlete(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": ID,
        "sex": "female",
        "dob_month": "2014-03",
        "sport": "football",
        "academy_id": "academy_1",
        "dominant_limb": "right",
    }
    base.update(overrides)
    return base


def test_valid_athlete_defaults_to_real_data_flag() -> None:
    athlete = Athlete.model_validate(_athlete())
    assert athlete.synthetic is False
    assert athlete.mid_parent_height is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"dob_month": "2014-03-17"},  # full date of birth is not allowed
        {"dob_month": "2014-13"},
        {"id": "athlete-7"},  # not a pseudonymous UUID
        {"sex": "F"},
        {"dominant_limb": "both"},
        {"mid_parent_height": 1.7},  # metres instead of cm
        {"name": "A. Person"},  # no identifying extra fields
    ],
)
def test_invalid_athletes_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Athlete.model_validate(_athlete(**overrides))


def test_trial_refuses_a_video_uri() -> None:
    base = {
        "id": ID,
        "session_id": ID,
        "task": "drop_vertical_jump",
        "limb": "both",
        "trial_no": 1,
        "quality": 0.9,
    }
    Trial.model_validate({**base, "keypoints_uri": "data/keypoints/a.parquet"})
    with pytest.raises(ValidationError, match="never to raw video"):
        Trial.model_validate({**base, "keypoints_uri": "data/raw_video_LOCAL_ONLY/a.MP4"})


def test_injury_requires_a_diagnoser_and_a_known_group() -> None:
    base = {
        "id": ID,
        "athlete_id": ID,
        "onset_date": date(2026, 3, 1),
        "osiics_code": "X",
        "body_area": "knee",
        "tissue": "apophysis",
        "mechanism": "non_contact",
        "time_loss_days": 10,
        "outcome_group": "growth_related",
        "diagnosed_by": "physio_1",
    }
    Injury.model_validate(base)
    with pytest.raises(ValidationError):
        Injury.model_validate({**base, "diagnosed_by": ""})
    with pytest.raises(ValidationError):
        Injury.model_validate({**base, "outcome_group": "high_risk_by_model"})
    with pytest.raises(ValidationError):
        Injury.model_validate({**base, "time_loss_days": -1})


def test_ostrc_severity_must_be_the_item_sum() -> None:
    base = {"athlete_id": ID, "week_start": date(2026, 1, 5), "q1": 8, "q2": 6, "q3": 0, "q4": 8}
    assert OstrcWeekly.model_validate({**base, "severity": 22}).severity == 22
    with pytest.raises(ValidationError, match="severity"):
        OstrcWeekly.model_validate({**base, "severity": 50})
