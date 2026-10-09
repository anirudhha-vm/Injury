"""Row models for the research database (brief §6.10). Mirrors ``data/schema/schema.sql``.

Units: lengths in cm, mass in kg, durations in hours or days. Athlete IDs are pseudonymous
UUIDs. Date of birth is stored at month precision only (``YYYY-MM``). Every table carries a
``synthetic`` flag. ``tests/test_schema.py`` keeps these models and the DDL in step.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Sex = Literal["male", "female"]
Limb = Literal["left", "right"]
TrialLimb = Literal["left", "right", "both"]
OutcomeGroup = Literal[
    "growth_related", "non_contact_soft_tissue", "other_non_contact", "contact", "other"
]
"""Stored groups. The analysis outcome "all non-contact" (N6c) is the union of the first three."""

NON_CONTACT_GROUPS: tuple[str, ...] = (
    "growth_related",
    "non_contact_soft_tissue",
    "other_non_contact",
)


def _uuid_text(value: str) -> str:
    UUID(value)
    return value


UuidStr = Annotated[str, AfterValidator(_uuid_text)]
Stature = Annotated[float, Field(ge=90.0, le=210.0)]
SittingHeight = Annotated[float, Field(ge=40.0, le=126.0)]
Mass = Annotated[float, Field(gt=10.0, le=200.0)]
WeeklyHours = Annotated[float, Field(ge=0.0, le=80.0)]
OstrcItem = Annotated[int, Field(ge=0, le=25)]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    synthetic: bool = False


class Athlete(Row):
    id: UuidStr
    sex: Sex
    dob_month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    sport: str
    academy_id: str
    dominant_limb: Limb
    mid_parent_height: float | None = Field(default=None, ge=130.0, le=210.0)


class Anthropometric(Row):
    """One visit; every measure taken twice and both raw values stored."""

    athlete_id: UuidStr
    date: date
    stature_1: Stature
    stature_2: Stature
    sitting_height_1: SittingHeight
    sitting_height_2: SittingHeight
    mass_1: Mass
    mass_2: Mass
    measurer_id: str


class ScreeningSession(Row):
    id: UuidStr
    athlete_id: UuidStr
    date: date
    camera_setup: str


class Trial(Row):
    id: UuidStr
    session_id: UuidStr
    task: str
    limb: TrialLimb
    trial_no: int = Field(ge=1)
    keypoints_uri: str
    quality: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _no_video_uri(self) -> Trial:
        if self.keypoints_uri.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")):
            raise ValueError("keypoints_uri must point to keypoints, never to raw video")
        return self


class TaskFeature(Row):
    """Long format. ``trial`` is None for a session-level summary."""

    athlete_id: UuidStr
    session_id: UuidStr
    task: str
    trial: int | None = Field(default=None, ge=1)
    limb: TrialLimb
    feature: str
    value: float
    quality: float = Field(ge=0.0, le=1.0)


class ExposureWeekly(Row):
    athlete_id: UuidStr
    week_start: date
    hours_training: WeeklyHours
    hours_match: WeeklyHours
    sessions: int = Field(ge=0, le=30)


class OstrcWeekly(Row):
    athlete_id: UuidStr
    week_start: date
    q1: OstrcItem
    q2: OstrcItem
    q3: OstrcItem
    q4: OstrcItem
    severity: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _severity_is_item_sum(self) -> OstrcWeekly:
        if self.severity != self.q1 + self.q2 + self.q3 + self.q4:
            raise ValueError("severity must equal q1 + q2 + q3 + q4")
        return self


class Injury(Row):
    """Physio-diagnosed injury: the only source of outcome labels (non-negotiable N1)."""

    id: UuidStr
    athlete_id: UuidStr
    onset_date: date
    osiics_code: str = Field(min_length=1)
    body_area: str
    tissue: str
    mechanism: str
    time_loss_days: int = Field(ge=0)
    outcome_group: OutcomeGroup
    diagnosed_by: str = Field(min_length=1)


class GrowthPosterior(Row):
    athlete_id: UuidStr
    fit_date: date
    uri: str


TABLES: dict[str, type[Row]] = {
    "athletes": Athlete,
    "anthropometrics": Anthropometric,
    "screening_sessions": ScreeningSession,
    "trials": Trial,
    "task_features": TaskFeature,
    "exposure_weekly": ExposureWeekly,
    "ostrc_weekly": OstrcWeekly,
    "injuries": Injury,
    "growth_posteriors": GrowthPosterior,
}
