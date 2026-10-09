"""Shared fixtures. All cohort data in tests is synthetic."""

from __future__ import annotations

from typing import Any

import pytest

from youthtwin.sim.cohort import SyntheticCohort, generate_cohort
from youthtwin.utils.config import load_config


@pytest.fixture(scope="session")
def cohort_config() -> dict[str, Any]:
    return load_config("sim/cohort.yaml")


@pytest.fixture(scope="session")
def cohort(cohort_config: dict[str, Any]) -> SyntheticCohort:
    """The full default synthetic cohort (300 athletes x 52 weeks)."""
    return generate_cohort(cohort_config)


@pytest.fixture(scope="session")
def qc_config() -> dict[str, Any]:
    return load_config("io/qc.yaml")
