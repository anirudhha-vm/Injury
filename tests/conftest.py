"""Shared fixtures. All cohort data in tests is synthetic."""

from __future__ import annotations

from typing import Any

import pytest

from cohort_settings import signal_cohort_config
from youthtwin.features.person_period import PersonPeriod, build_person_period
from youthtwin.sim.cohort import SyntheticCohort, generate_cohort
from youthtwin.utils.config import load_config


@pytest.fixture(scope="session")
def cohort_config() -> dict[str, Any]:
    """Realistic cohort config plus the strong-signal overrides in sim/test_signal.yaml."""
    return signal_cohort_config()


@pytest.fixture(scope="session")
def cohort(cohort_config: dict[str, Any]) -> SyntheticCohort:
    """The full default synthetic cohort (300 athletes x 52 weeks)."""
    return generate_cohort(cohort_config)


@pytest.fixture(scope="session")
def qc_config() -> dict[str, Any]:
    return load_config("io/qc.yaml")


@pytest.fixture(scope="session")
def person_period_config() -> dict[str, Any]:
    return load_config("features/person_period.yaml")


@pytest.fixture(scope="session")
def hazard_config() -> dict[str, Any]:
    return load_config("models/hazard.yaml")


@pytest.fixture(scope="session")
def person_period(
    cohort: SyntheticCohort, person_period_config: dict[str, Any], hazard_config: dict[str, Any]
) -> PersonPeriod:
    """Athlete-week table of the default synthetic cohort (no growth-twin features)."""
    return build_person_period(
        cohort.tables, config=person_period_config, hazard_config=hazard_config
    )
