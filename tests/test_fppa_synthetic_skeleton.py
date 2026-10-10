"""Knee valgus as a frontal-plane projection angle (v2 regression).

v2 claimed "valgus" but computed a 3D included angle that mixes sagittal flexion into the
result. Here the FPPA is recovered from a synthetic skeleton whose frontal-plane angle is set
independently of its knee flexion.
"""

from __future__ import annotations

import numpy as np
import pytest

from youthtwin.pose import kinematics as kin
from youthtwin.pose import task_features as tf
from youthtwin.sim import skeleton_sim as sk
from youthtwin.utils.seeding import make_rng


def p(x: float, y: float) -> np.ndarray:
    return np.array([[x, y]], dtype=np.float64)


def test_fppa_hand_built_geometry() -> None:
    # Athlete faces the camera: their LEFT leg is on the image right. Knee pulled 20 px toward
    # the midline (image -x) between a hip and an ankle that are vertically aligned.
    hip, knee, ankle = p(60, 0), p(40, 100), p(60, 200)
    expected = 2 * np.degrees(np.arctan2(20, 100))
    assert kin.knee_fppa(hip, knee, ankle, side="left")[0] == pytest.approx(expected)
    # The same picture read as a RIGHT leg would be a knee moving away from the midline.
    assert kin.knee_fppa(hip, knee, ankle, side="right")[0] == pytest.approx(-expected)
    # Mirror image: right leg on the image left, knee toward the midline (image +x).
    assert kin.knee_fppa(p(-60, 0), p(-40, 100), p(-60, 200), side="right")[0] == pytest.approx(
        expected
    )
    assert kin.knee_fppa(hip, p(60, 100), ankle, side="left")[0] == pytest.approx(0.0)
    assert kin.knee_fppa(hip, knee, ankle, side="left", facing_camera=False)[0] == pytest.approx(
        -expected
    )


@pytest.mark.parametrize("peak_fppa", [(15.0, 4.0), (0.0, 10.0), (-8.0, 8.0)])
def test_fppa_recovered_on_both_limbs_at_every_frame(peak_fppa: tuple[float, float]) -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(1, "fppa"), peak_fppa=peak_fppa)
    for side in ("left", "right"):
        estimate = tf.frontal_angles(trial.frontal, side, unilateral=False)["knee_fppa"]
        np.testing.assert_allclose(estimate, trial.angles[f"knee_fppa_{side}"], atol=1e-6)


def test_fppa_does_not_depend_on_knee_flexion() -> None:
    """Same frontal-plane angle, very different sagittal flexion: FPPA must not change."""
    shallow = sk.simulate_drop_vertical_jump(make_rng(1, "a"), peak_knee_flexion=40.0)
    deep = sk.simulate_drop_vertical_jump(make_rng(1, "b"), peak_knee_flexion=110.0)
    frame_shallow, frame_deep = (t.events["peak_knee_flexion"] for t in (shallow, deep))
    fppa_shallow = tf.frontal_angles(shallow.frontal, "left", unilateral=False)["knee_fppa"]
    fppa_deep = tf.frontal_angles(deep.frontal, "left", unilateral=False)["knee_fppa"]
    assert fppa_shallow[frame_shallow] == pytest.approx(12.0, abs=1e-6)
    assert fppa_deep[frame_deep] == pytest.approx(12.0, abs=1e-6)
    # ...whereas the sagittal view sees the difference.
    flexion = tf.sagittal_angles(deep.sagittal, "left")["knee_flexion"][frame_deep]
    assert flexion == pytest.approx(110.0, abs=1e-6)


def test_sagittal_angles_recovered_exactly() -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(2, "sagittal"))
    for side in ("left", "right"):
        angles = tf.sagittal_angles(trial.sagittal, side)
        for name in tf.SAGITTAL_ANGLES:
            np.testing.assert_allclose(angles[name], trial.angles[f"{name}_{side}"], atol=1e-6)


@pytest.mark.parametrize("limb", ["left", "right"])
def test_unilateral_frontal_angles_recovered(limb: str) -> None:
    trial = sk.simulate_single_leg_squat(make_rng(3, limb), limb=limb)  # type: ignore[arg-type]
    angles = tf.frontal_angles(trial.frontal, limb, unilateral=True)  # type: ignore[arg-type]
    np.testing.assert_allclose(angles["knee_fppa"], trial.angles[f"knee_fppa_{limb}"], atol=1e-6)
    np.testing.assert_allclose(angles["pelvic_drop"], trial.angles["pelvic_drop"], atol=1e-6)
    np.testing.assert_allclose(
        angles["trunk_lateral_lean"], trial.angles["trunk_lateral_lean"], atol=1e-6
    )
    assert angles["pelvic_drop"].max() == pytest.approx(6.0, abs=1e-6)


def test_wrong_camera_is_refused() -> None:
    trial = sk.simulate_drop_vertical_jump(make_rng(4, "camera"))
    with pytest.raises(ValueError, match="front camera"):
        tf.frontal_angles(trial.sagittal, "left", unilateral=False)
    with pytest.raises(ValueError, match="side camera"):
        tf.sagittal_angles(trial.frontal, "left")
