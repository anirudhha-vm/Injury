"""Cohort settings for tests: the realistic config plus the strong-signal overrides (D-120)."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from youthtwin.utils.config import load_config


def signal_cohort_config() -> dict[str, Any]:
    """``sim/cohort.yaml`` with ``sim/test_signal.yaml`` merged in. Pipeline tests only."""
    merged = OmegaConf.merge(load_config("sim/cohort.yaml"), load_config("sim/test_signal.yaml"))
    config: dict[str, Any] = OmegaConf.to_container(merged)  # type: ignore[assignment]
    return config
