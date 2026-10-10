"""Event detection on signals and synthetic sequences with known event frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from youthtwin.pose import events as ev
from youthtwin.sim import skeleton_sim as sk
from youthtwin.utils.config import load_config
from youthtwin.utils.seeding import make_rng

# Thresholded contact detection fires slightly before touch-down and after lift-off.
FRAME_TOLERANCE = 5  # frames at 120 fps (about 40 ms)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return load_config("pose/events.yaml")


def _dvj_signals(trial: sk.SyntheticTrial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = ev.normalised_foot_height(trial.sagittal, ("left", "right"))
    return height, trial.angles["knee_flexion_left"], trial.sagittal.timestamps_ms


def test_foot_height_is_normalised_by_leg_length() -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(1, "h"))
    height = ev.normalised_foot_height(trial.sagittal, ("left", "right"))
    np.testing.assert_allclose(height, trial.foot_height, atol=0.02)
    assert height[:20].mean() == pytest.approx(0.30, abs=0.02)  # the box
    # The leg-length estimate is the median over frames, slightly short of the true segment
    # sum only through noise; here there is none.
    assert ev.leg_length_px(trial.sagittal, ("left",)) == pytest.approx(sk.LEG_LENGTH)


@pytest.mark.parametrize("noise_px", [0.0, 1.5])
def test_drop_vertical_jump_events_are_recovered(config: dict[str, Any], noise_px: float) -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(2, "dvj"), noise_px=noise_px)
    found = ev.detect_drop_vertical_jump(*_dvj_signals(trial), config)
    assert found.missing == {}
    assert set(found.events) == set(ev.DVJ_EVENTS)
    for name, truth in trial.events.items():
        assert abs(found.frame(name) - truth) <= FRAME_TOLERANCE, name
    order = [found.frame(name) for name in ev.DVJ_EVENTS if name != "peak_knee_flexion"]
    assert order == sorted(order)
    assert (
        found.frame("initial_contact") < found.frame("peak_knee_flexion") < found.frame("take_off")
    )
    assert found.min_confidence > 0.8
    event = found.events["initial_contact"]
    assert event.time_ms == trial.sagittal.timestamps_ms[event.frame] and event.source == "auto"


def test_missing_events_are_reported_not_invented(config: dict[str, Any]) -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(3, "cut"))
    height, flexion, stamps = _dvj_signals(trial)

    # Clip ends during the first stance: no take-off, no second landing.
    cut = trial.events["peak_knee_flexion"] + 5
    partial = ev.detect_drop_vertical_jump(height[:cut], flexion[:cut], stamps[:cut], config)
    assert {"initial_contact", "box_drop", "peak_knee_flexion"} <= set(partial.events)
    assert set(partial.missing) == {"take_off", "second_landing"}
    assert partial.min_confidence == 0.0

    # Clip ends in flight: take-off found, no second landing.
    cut = trial.events["take_off"] + 20
    flight = ev.detect_drop_vertical_jump(height[:cut], flexion[:cut], stamps[:cut], config)
    assert "take_off" in flight.events and set(flight.missing) == {"second_landing"}

    # Athlete starts on the floor: not a drop jump.
    start = trial.events["initial_contact"] + 5
    floor = ev.detect_drop_vertical_jump(height[start:], flexion[start:], stamps[start:], config)
    assert floor.events == {} and set(floor.missing) == set(ev.DVJ_EVENTS)
    assert "no box phase" in floor.missing["initial_contact"]

    # Feet never come down.
    never = ev.detect_drop_vertical_jump(np.full(100, 0.3), flexion[:100], stamps[:100], config)
    assert "never reach the floor" in never.missing["take_off"]

    # Knee flexion unavailable during stance.
    blind = ev.detect_drop_vertical_jump(height, np.full_like(flexion, np.nan), stamps, config)
    assert "peak_knee_flexion" in blind.missing and "initial_contact" in blind.events


def test_gaps_lower_event_confidence(config: dict[str, Any]) -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(4, "gap"))
    height, flexion, stamps = _dvj_signals(trial)
    clean = ev.detect_drop_vertical_jump(height, flexion, stamps, config)
    holed = height.copy()
    contact = trial.events["initial_contact"]
    holed[contact - 12 : contact - 2] = np.nan
    gappy = ev.detect_drop_vertical_jump(holed, flexion, stamps, config)
    assert gappy.events["initial_contact"].confidence < clean.events["initial_contact"].confidence


@pytest.mark.parametrize("limb", ["left", "right"])
def test_single_leg_squat_events(config: dict[str, Any], limb: str) -> None:
    trial = sk.simulate_single_leg_squat(make_rng(5, limb), limb=limb)  # type: ignore[arg-type]
    flexion = trial.angles[f"knee_flexion_{limb}"]
    found = ev.detect_single_leg_squat(flexion, trial.sagittal.timestamps_ms, config)
    assert found.missing == {} and set(found.events) == set(ev.SQUAT_EVENTS)
    assert abs(found.frame("deepest_point") - trial.events["deepest_point"]) <= FRAME_TOLERANCE
    # Start and return are defined as crossing (standing baseline + onset threshold).
    level = 5.0 + config["single_leg_squat"]["onset_deg"]
    deepest = trial.events["deepest_point"]
    expected_start = int(np.flatnonzero(flexion[:deepest] <= level)[-1])
    expected_return = deepest + int(np.flatnonzero(flexion[deepest:] <= level)[0])
    assert found.frame("start") == expected_start and found.frame("return") == expected_return
    assert trial.events["start"] < found.frame("start") < deepest < found.frame("return")


def test_single_leg_squat_failures(config: dict[str, Any]) -> None:
    stamps = np.arange(200) * 1000.0 / 120.0
    shallow = 5.0 + 10.0 * np.sin(np.linspace(0, np.pi, 200))
    result = ev.detect_single_leg_squat(shallow, stamps, config)
    assert result.events == {} and "below the minimum" in result.missing["deepest_point"]
    nothing = ev.detect_single_leg_squat(np.full(200, np.nan), stamps, config)
    assert "unavailable" in nothing.missing["start"]
    deep = 5.0 + 60.0 * np.sin(np.linspace(0, np.pi, 200))
    no_return = ev.detect_single_leg_squat(deep[:120], stamps[:120], config)
    assert "return" in no_return.missing and "deepest_point" in no_return.events
    already_down = ev.detect_single_leg_squat(
        np.concatenate([np.full(10, 5.0), deep[60:]])[10:], stamps[:140], config
    )
    assert "start" in already_down.missing


def test_hop_events(config: dict[str, Any]) -> None:
    stamps = np.arange(240) * 1000.0 / 120.0
    height = np.zeros(240)
    flight = slice(80, 130)
    phase = np.linspace(0, 1, 50)
    height[flight] = 0.4 * 4 * phase * (1 - phase)
    found = ev.detect_hop(height, stamps, config)
    assert abs(found.frame("take_off") - 80) <= FRAME_TOLERANCE
    assert abs(found.frame("landing") - 130) <= FRAME_TOLERANCE
    assert found.task == "single_leg_hop_distance" and found.missing == {}

    assert set(ev.detect_hop(np.zeros(100), stamps[:100], config).missing) == set(ev.HOP_EVENTS)
    airborne = ev.detect_hop(height[:110], stamps[:110], config)
    assert "take_off" in airborne.events and "landing" in airborne.missing


def test_manual_overrides(config: dict[str, Any], tmp_path: Path) -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(6, "manual"))
    height, flexion, stamps = _dvj_signals(trial)
    cut = trial.events["take_off"] + 20
    detected = ev.detect_drop_vertical_jump(height[:cut], flexion[:cut], stamps[:cut], config)
    table = tmp_path / "overrides.csv"
    table.write_text(
        "trial_id,event,frame\nT1,initial_contact,91\nT1,second_landing,160\nT2,start,10\n",
        encoding="utf-8",
    )
    overrides = ev.read_overrides(table)
    assert overrides == {"T1": {"initial_contact": 91, "second_landing": 160}, "T2": {"start": 10}}
    fixed = ev.apply_overrides(detected, overrides["T1"], stamps[:cut])
    assert fixed.frame("initial_contact") == 91 and fixed.missing == {}
    assert fixed.events["second_landing"].source == "manual"
    assert fixed.events["second_landing"].confidence == 1.0
    assert fixed.events["take_off"].source == "auto"
    assert detected.frame("initial_contact") != 91  # the original is not modified
    with pytest.raises(ValueError, match="outside the clip"):
        ev.apply_overrides(detected, {"take_off": 10_000}, stamps[:cut])
