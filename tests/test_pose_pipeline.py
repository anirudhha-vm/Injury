"""Keypoints to features, end to end, on synthetic sequences; keypoint I/O; quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from youthtwin.pose import task_features as tf
from youthtwin.pose.estimators import base
from youthtwin.pose.keypoints import KeypointSequence
from youthtwin.sim import skeleton_sim as sk
from youthtwin.utils.config import load_config
from youthtwin.utils.seeding import make_rng


@pytest.fixture(scope="module")
def configs() -> dict[str, Any]:
    return {
        "filter_config": load_config("pose/filter.yaml"),
        "event_config": load_config("pose/events.yaml"),
        "feature_config": load_config("features/task_features.yaml"),
    }


def _input(simulated: sk.SyntheticTrial, number: int = 1, **overrides: Any) -> tf.TrialInput:
    fields: dict[str, Any] = {
        "athlete_id": "athlete",
        "session_id": "session",
        "task": simulated.task,
        "trial": number,
        "limb": simulated.limb,
        "sagittal": simulated.sagittal,
        "frontal": simulated.frontal,
    }
    fields.update(overrides)
    return tf.TrialInput(**fields)


# ----------------------------------------------------------------------------- keypoints


def test_keypoint_sequence_properties_and_round_trip(tmp_path: Path) -> None:
    sequence = sk.simulate_drop_vertical_jump(make_rng(1, "io"), noise_px=1.0).sagittal
    assert sequence.fps == pytest.approx(120.0)
    assert sequence.time_s[0] == 0.0 and sequence.joint("left", "knee").shape == (264, 2)
    stored = sequence.save(tmp_path / "athlete" / "trial1_sagittal")
    assert stored.suffix == ".parquet"
    loaded = KeypointSequence.load(tmp_path / "athlete" / "trial1_sagittal")
    np.testing.assert_allclose(loaded.xy, sequence.xy)
    np.testing.assert_allclose(loaded.conf, sequence.conf)
    np.testing.assert_allclose(loaded.timestamps_ms, sequence.timestamps_ms)
    assert (loaded.camera, loaded.skeleton, loaded.synthetic) == ("sagittal", "blazepose", True)
    # Nothing in the stored files points back at a video.
    text = (tmp_path / "athlete" / "trial1_sagittal.json").read_text(encoding="utf-8").lower()
    assert "video" not in text and ".mp4" not in text
    assert sequence.to_frame()["synthetic"].all()


def test_keypoint_sequence_validation() -> None:
    good = sk.simulate_drop_vertical_jump(make_rng(2, "bad")).sagittal
    fields = dict(good.__dict__)
    with pytest.raises(ValueError, match="shape"):
        KeypointSequence(**{**fields, "xy": good.xy[:, :10]})
    with pytest.raises(ValueError, match="expects 33"):
        KeypointSequence(**{**fields, "xy": good.xy[:, :10], "conf": good.conf[:, :10]})
    with pytest.raises(ValueError, match="one entry per frame"):
        KeypointSequence(**{**fields, "timestamps_ms": good.timestamps_ms[:-1]})
    with pytest.raises(ValueError, match="strictly increasing"):
        KeypointSequence(**{**fields, "timestamps_ms": good.timestamps_ms[::-1].copy()})
    with pytest.raises(ValueError, match="z must"):
        KeypointSequence(**{**fields, "z": np.zeros((3, 3))})
    with_depth = KeypointSequence(**{**fields, "z": np.zeros_like(good.conf)})
    assert "z" in with_depth.to_frame().columns
    assert good.with_xy(good.xy + 1.0).xy[0, 25, 0] == good.xy[0, 25, 0] + 1.0


# ----------------------------------------------------------------------------- one trial


def test_drop_vertical_jump_features_match_the_truth(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(3, "dvj"), noise_px=1.0)
    result = tf.trial_features(_input(simulated), **configs)
    assert result.included and result.quality > 0.9
    peak = simulated.events["peak_knee_flexion"]
    for side in ("left", "right"):
        for name in ("knee_flexion", "hip_flexion", "ankle_dorsiflexion", "knee_fppa"):
            estimate = result.features[(side, f"{name}_at_peak_knee_flexion")]
            assert estimate == pytest.approx(simulated.angles[f"{name}_{side}"][peak], abs=1.5)
        assert result.features[(side, "knee_flexion_max")] == pytest.approx(85.0, abs=1.5)
        assert result.features[(side, "knee_flexion_rom")] > 50.0
        # Roughly 65 degrees of flexion in 0.2 s, eased: a few hundred degrees per second.
        assert 300.0 < result.features[(side, "knee_flexion_peak_velocity")] < 900.0
        curve = result.curves[(side, "knee_flexion")]
        assert curve.shape == (101,) and curve.max() == pytest.approx(85.0, abs=1.5)
        assert (side, "knee_fppa_peak_velocity") not in result.features
    assert result.features[("both", "window_duration_s")] == pytest.approx(0.45, abs=0.08)
    assert result.features[("both", "knee_ankle_separation_min")] < 1.0  # knees move inward


def test_velocity_does_not_depend_on_frame_rate(configs: dict[str, Any]) -> None:
    """The same movement filmed at 60 and 240 fps gives the same deg/s (v2: deg/frame)."""
    velocity = []
    for fps in (60.0, 240.0):
        simulated = sk.simulate_drop_vertical_jump(make_rng(4, "fps"), fps=fps)
        result = tf.trial_features(_input(simulated), **configs)
        velocity.append(result.features[("left", "knee_flexion_peak_velocity")])
    assert velocity[0] == pytest.approx(velocity[1], rel=0.08)


def test_single_leg_squat_features(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_single_leg_squat(make_rng(5, "sls"), limb="right", noise_px=1.0)
    result = tf.trial_features(_input(simulated), **configs)
    assert result.included
    assert result.features[("right", "knee_flexion_at_deepest_point")] == pytest.approx(
        65.0, abs=1.5
    )
    assert result.features[("right", "knee_fppa_at_deepest_point")] == pytest.approx(10.0, abs=1.5)
    assert result.features[("right", "pelvic_drop_at_deepest_point")] == pytest.approx(6.0, abs=1.0)
    assert result.features[("right", "trunk_lateral_lean_at_deepest_point")] == pytest.approx(
        5.0, abs=1.0
    )
    assert not [key for key in result.features if key[0] == "left"]


def test_sagittal_only_and_frontal_clock_offset(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(6, "views"))
    only_side = tf.trial_features(_input(simulated, frontal=None), **configs)
    assert only_side.included and ("left", "knee_flexion_max") in only_side.features
    assert not [key for key in only_side.features if "fppa" in key[1]]

    # The frontal phone started 500 ms earlier: its clock reads 500 ms more at the same instant.
    shifted = KeypointSequence(
        **{**simulated.frontal.__dict__, "timestamps_ms": simulated.frontal.timestamps_ms + 500.0}
    )
    aligned = tf.trial_features(
        _input(simulated, frontal=shifted, frontal_offset_ms=-500.0), **configs
    )
    reference = tf.trial_features(_input(simulated), **configs)
    key = ("left", "knee_fppa_at_peak_knee_flexion")
    assert aligned.features[key] == pytest.approx(reference.features[key], abs=1e-6)
    unaligned = tf.trial_features(_input(simulated, frontal=shifted), **configs)
    assert unaligned.features[key] != pytest.approx(reference.features[key], abs=0.5)


def test_uneven_frame_timing_is_handled(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(7, "vfr"))
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(simulated.sagittal.n_frames, 220, replace=False))  # dropped frames
    sagittal = simulated.sagittal
    uneven = KeypointSequence(
        **{
            **sagittal.__dict__,
            "xy": sagittal.xy[keep],
            "conf": sagittal.conf[keep],
            "timestamps_ms": sagittal.timestamps_ms[keep],
        }
    )
    result = tf.trial_features(_input(simulated, sagittal=uneven, frontal=None), **configs)
    assert result.included
    assert result.features[("left", "knee_flexion_max")] == pytest.approx(85.0, abs=2.0)


# ----------------------------------------------------------------------------- quality control


def test_low_quality_and_unusable_trials_are_logged_not_dropped(configs: dict[str, Any]) -> None:
    good = sk.simulate_drop_vertical_jump(make_rng(8, "good"), noise_px=1.0)
    dropout = np.zeros((264, 33))
    dropout[::2] = 1  # every other frame lost
    poor = sk.simulate_drop_vertical_jump(make_rng(8, "poor"), dropout=dropout)
    cut = good.events["peak_knee_flexion"] + 3
    truncated = KeypointSequence(
        **{
            **good.sagittal.__dict__,
            "xy": good.sagittal.xy[:cut],
            "conf": good.sagittal.conf[:cut],
            "timestamps_ms": good.sagittal.timestamps_ms[:cut],
        }
    )
    results = [
        tf.trial_features(_input(good, 1), **configs),
        tf.trial_features(_input(poor, 2), **configs),
        tf.trial_features(_input(good, 3, sagittal=truncated, frontal=None), **configs),
        tf.trial_features(_input(good, 4, sagittal=None), **configs),
        tf.trial_features(_input(good, 5, task="tuck_jump"), **configs),
    ]
    assert [r.included for r in results] == [True, False, False, False, False]
    table, excluded = tf.session_features(results, configs["feature_config"])

    assert list(table.columns) == tf.FEATURE_COLUMNS and set(table["trial"].dropna()) == {1}
    assert list(excluded.columns) == tf.EXCLUSION_COLUMNS
    assert excluded["trial"].tolist() == [2, 3, 4, 5]  # every excluded trial is accounted for
    reasons = excluded.set_index("trial")["reason"]
    assert "below threshold" in reasons[2]
    assert "events not found" in reasons[3] and "take_off" in reasons[3]
    assert "no sagittal view" in reasons[4]
    assert "not supported" in reasons[5]
    assert table["synthetic"].all()
    assert len(results) == table["trial"].dropna().nunique() + len(excluded)


def test_session_of_only_excluded_trials(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(9, "none"))
    result = tf.trial_features(_input(simulated, sagittal=None), **configs)
    table, excluded = tf.session_features([result], configs["feature_config"])
    assert table.empty and list(table.columns) == tf.FEATURE_COLUMNS and len(excluded) == 1


def test_manual_override_rescues_a_trial(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(10, "override"))
    cut = simulated.events["take_off"] - 5  # clip ends just before take-off
    sagittal = simulated.sagittal
    short = KeypointSequence(
        **{
            **sagittal.__dict__,
            "xy": sagittal.xy[:cut],
            "conf": sagittal.conf[:cut],
            "timestamps_ms": sagittal.timestamps_ms[:cut],
        }
    )
    failed = tf.trial_features(_input(simulated, sagittal=short, frontal=None), **configs)
    assert not failed.included
    rescued = tf.trial_features(
        _input(simulated, sagittal=short, frontal=None, event_overrides={"take_off": cut - 1}),
        **configs,
    )
    assert rescued.features and rescued.events.events["take_off"].source == "manual"


def test_time_normalise() -> None:
    values = np.arange(50, dtype=np.float64)
    curve = tf.time_normalise(values, 10, 30, 101)
    assert curve.shape == (101,) and curve[0] == 10.0 and curve[-1] == 30.0
    assert curve[50] == pytest.approx(20.0)
    assert np.isnan(tf.time_normalise(values, 10, 10, 101)).all()


# ----------------------------------------------------------------------------- estimators


def test_estimator_helpers(tmp_path: Path) -> None:
    with pytest.raises(ImportError, match="uv sync --extra pose"):
        base.require("a_module_that_does_not_exist")
    assert base.require("json").__name__ == "json"
    with pytest.raises(FileNotFoundError, match="Video not found"):
        next(base.read_frames(tmp_path / "missing.mp4"))
    stamps = base.monotonic_timestamps([0.0, 8.3, 8.3, 16.7, 16.0])
    assert np.all(np.diff(stamps) > 0) and stamps[0] == 0.0 and stamps[1] == 8.3


def test_mediapipe_estimator_needs_a_model_file(tmp_path: Path) -> None:
    from youthtwin.pose.estimators.mediapipe import MediaPipePoseEstimator

    config = load_config("pose/estimator.yaml")
    config["mediapipe"]["model_path"] = str(tmp_path / "absent.task")
    estimator = MediaPipePoseEstimator(config)
    with pytest.raises(FileNotFoundError, match=r"TODO\(human\)"):
        estimator.process(tmp_path / "clip.mp4", "sagittal")
