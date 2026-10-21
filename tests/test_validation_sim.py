"""Synthetic markerless-vs-mocap study: pairs from the real pipeline; error grows with offsets."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from youthtwin.sim import validation_sim
from youthtwin.utils.config import load_config
from youthtwin.validation.agreement import agreement_table

CONFIG = load_config("sim/validation.yaml")


@pytest.fixture(scope="module")
def pipeline() -> dict[str, Any]:
    return {
        "filter": load_config("pose/filter.yaml"),
        "events": load_config("pose/events.yaml"),
        "features": load_config("features/task_features.yaml"),
    }


def test_pairs_structure(pipeline: dict[str, Any]) -> None:
    groups = {"pre_phv": 3, "post_phv": 3}
    pairs = validation_sim.simulate_study(CONFIG, pipeline, seed=0, offset_sd_px=3.0, groups=groups)
    assert pairs["synthetic"].all() and pairs["athlete_id"].nunique() == 6
    assert set(pairs["maturity_group"]) == set(groups)
    assert set(zip(pairs["variable"], pairs["event"], strict=True)) == {
        ("knee_flexion", "max"),
        ("knee_fppa", "peak_knee_flexion"),
        ("knee_fppa", "max"),
        ("hip_flexion", "max"),
    }
    assert (pairs["n_trials"] >= 1).all()
    table = agreement_table(pairs, load_config("validation/agreement.yaml"), by=["maturity_group"])
    assert table["rmse"].notna().all()


def test_error_grows_with_offset_and_is_reproducible(pipeline: dict[str, Any]) -> None:
    groups = {"post_phv": 6}

    def mae(offset: float) -> float:
        pairs = validation_sim.simulate_study(
            CONFIG, pipeline, seed=1, offset_sd_px=offset, groups=groups
        )
        return validation_sim.mean_absolute_error(pairs, "knee_flexion_max")

    small, large = mae(0.0), mae(8.0)
    assert small < 1.0 < large
    assert mae(8.0) == pytest.approx(large)  # same seed, same study


def test_smaller_bodies_get_larger_angle_errors(pipeline: dict[str, Any]) -> None:
    """The same pixel error costs more degrees in a child who spans fewer pixels."""
    config = {**CONFIG, "body_scale": {"pre_phv": 0.6, "post_phv": 1.0}}
    pairs = validation_sim.simulate_study(
        config, pipeline, seed=2, offset_sd_px=5.0, groups={"pre_phv": 15, "post_phv": 15}
    )
    errors: pd.Series = (
        (pairs["markerless"] - pairs["mocap"]).abs().groupby(pairs["maturity_group"]).mean()
    )
    assert errors["pre_phv"] > errors["post_phv"]
