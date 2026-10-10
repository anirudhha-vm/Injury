"""Angle conventions (v2 regression).

Reference geometries are simple enough to check by eye. Image coordinates: x right, y DOWN.
"""

from __future__ import annotations

import numpy as np
import pytest

from youthtwin.pose import kinematics as kin


def p(x: float, y: float) -> np.ndarray:
    return np.array([[x, y]], dtype=np.float64)


# ----------------------------------------------------------------------------- sagittal


def test_included_angle_references() -> None:
    assert kin.included_angle(p(0, 0), p(0, 1), p(0, 2))[0] == pytest.approx(180.0)
    assert kin.included_angle(p(1, 0), p(0, 0), p(0, 1))[0] == pytest.approx(90.0)
    assert kin.included_angle(p(1, 0), p(0, 0), p(1, 1))[0] == pytest.approx(45.0)
    assert np.isnan(kin.included_angle(p(0, 0), p(0, 0), p(1, 1))[0])  # zero-length segment


def test_knee_flexion_is_zero_at_full_extension() -> None:
    """v2 compared the ~180 degree included angle with 30-60 degree flexion thresholds."""
    hip, knee, ankle = p(0, 0), p(0, 100), p(0, 200)  # straight leg, vertical
    assert kin.included_angle(hip, knee, ankle)[0] == pytest.approx(180.0)
    assert kin.knee_flexion(hip, knee, ankle)[0] == pytest.approx(0.0)


def test_knee_flexion_equals_180_minus_included_angle() -> None:
    hip, knee = p(0, 0), p(0, 100)
    for flexion in (30.0, 60.0, 90.0, 120.0):
        # Shank rotated backward from straight-down by `flexion`.
        angle = np.radians(flexion)
        ankle = p(-100 * np.sin(angle), 100 + 100 * np.cos(angle))
        assert kin.knee_flexion(hip, knee, ankle)[0] == pytest.approx(flexion)
        assert kin.included_angle(hip, knee, ankle)[0] == pytest.approx(180.0 - flexion)


def test_hip_flexion_convention() -> None:
    shoulder, hip = p(0, 0), p(0, 100)
    assert kin.hip_flexion(shoulder, hip, p(0, 200))[0] == pytest.approx(0.0)  # standing
    assert kin.hip_flexion(shoulder, hip, p(100, 100))[0] == pytest.approx(90.0)  # thigh forward
    assert kin.hip_flexion(p(50, 13.4), hip, p(0, 200))[0] == pytest.approx(30.0, abs=0.05)


def test_ankle_dorsiflexion_convention() -> None:
    ankle, toe = p(0, 100), p(100, 100)  # foot horizontal, pointing +x
    assert kin.ankle_dorsiflexion(p(0, 0), ankle, toe)[0] == pytest.approx(0.0)  # shank vertical
    forward = np.radians(20.0)
    knee_forward = p(100 * np.sin(forward), 100 - 100 * np.cos(forward))
    assert kin.ankle_dorsiflexion(knee_forward, ankle, toe)[0] == pytest.approx(20.0)
    knee_back = p(-100 * np.sin(forward), 100 - 100 * np.cos(forward))
    assert kin.ankle_dorsiflexion(knee_back, ankle, toe)[0] == pytest.approx(-20.0)  # plantar


def test_angles_are_vectorised_over_frames() -> None:
    hip = np.zeros((5, 2))
    knee = np.tile([0.0, 100.0], (5, 1))
    ankle = np.stack(
        [
            -100 * np.sin(np.radians(np.arange(5) * 10.0)),
            100 + 100 * np.cos(np.radians(np.arange(5) * 10.0)),
        ],
        axis=1,
    )
    np.testing.assert_allclose(kin.knee_flexion(hip, knee, ankle), np.arange(5) * 10.0, atol=1e-9)


# ----------------------------------------------------------------------------- frontal


def test_pelvic_drop_sign() -> None:
    # Facing the camera: left hip on the image right. Image y is down.
    left_hip, right_hip_lower = p(60, 100), p(-60, 110)
    drop = kin.pelvic_drop(left_hip, right_hip_lower, stance="left")
    assert drop[0] == pytest.approx(np.degrees(np.arctan2(10, 120)))
    assert kin.pelvic_drop(left_hip, right_hip_lower, stance="right")[0] == pytest.approx(-drop[0])
    assert kin.pelvic_drop(p(60, 100), p(-60, 100), stance="left")[0] == pytest.approx(0.0)


def test_trunk_lean_is_positive_toward_the_stance_limb() -> None:
    hips = (p(60, 300), p(-60, 300))  # left, right
    lean_to_left = (p(60 + 85, 100), p(60 - 85, 100))  # shoulders shifted toward image +x
    toward_left = kin.trunk_lateral_lean(*lean_to_left, *hips, stance="left")[0]
    assert toward_left == pytest.approx(np.degrees(np.arctan2(60, 200)))
    assert kin.trunk_lateral_lean(*lean_to_left, *hips, stance="right")[0] == pytest.approx(
        -toward_left
    )
    behind = kin.trunk_lateral_lean(*lean_to_left, *hips, stance="left", facing_camera=False)
    assert behind[0] == pytest.approx(-toward_left)


def test_knee_ankle_separation_ratio() -> None:
    ratio = kin.knee_ankle_separation_ratio(p(40, 0), p(-40, 0), p(50, 0), p(-50, 0))
    assert ratio[0] == pytest.approx(0.8)
    assert np.isnan(kin.knee_ankle_separation_ratio(p(40, 0), p(-40, 0), p(0, 0), p(0, 0))[0])
