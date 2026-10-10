"""Two-dimensional joint angles with explicit conventions. All angles in degrees.

Inputs are joint trajectories of shape (T, 2) in **image coordinates** (pixels, x right,
**y down**). Sagittal angles need the side camera; frontal-plane angles need the front camera.
These are 2D projection angles, not 3D joint angles: they are valid only when the movement plane
is roughly perpendicular to the camera axis (see ``docs/protocol/02_camera_setup_sop.md``).

Conventions
-----------
Sagittal (side camera):

- **knee flexion** = 180 - included angle(hip, knee, ankle). 0 at full extension, positive in
  flexion. (v2 compared the ~180 degree included angle to flexion thresholds.)
- **hip flexion** = 180 - included angle(shoulder, hip, knee). 0 when trunk and thigh are in
  line, positive in flexion. It is a trunk-thigh angle, so it includes trunk lean.
- **ankle dorsiflexion** = 90 - included angle(knee, ankle, foot_index). Positive when the shank
  moves over the foot. The ankle and toe landmarks are not on the anatomical axes, so there is
  a constant offset: use changes from a standing reference, not absolute values.

Frontal (front camera, athlete facing the camera):

- **knee FPPA** (frontal-plane projection angle) = 180 - included angle(hip, knee, ankle), signed
  so that **positive = knee medial to the hip-ankle line (valgus direction)** for either limb.
  (v2 reported "valgus" from a 3D included angle that mixed planes.)
- **pelvic drop**: angle of the hip-hip line from horizontal, positive when the hip opposite the
  stance limb is lower.
- **trunk lateral lean**: angle of the mid-hip to mid-shoulder line from vertical, positive
  toward the stance limb.
- **knee-to-ankle separation ratio**: horizontal knee distance / horizontal ankle distance;
  below 1 means the knees are closer together than the ankles.

The athlete's left side appears on the image right when they face the camera. Pass
``facing_camera=False`` if they are filmed from behind.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose.skeletons import Side

FloatArray = NDArray[np.float64]

_STRAIGHT = 180.0
_RIGHT_ANGLE = 90.0


def included_angle(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    """Angle at ``b`` between the segments ``b->a`` and ``b->c`` (0-180 degrees)."""
    u, v = a - b, c - b
    norms = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(u * v, axis=-1) / norms
    return np.asarray(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), dtype=np.float64)


def knee_flexion(hip: FloatArray, knee: FloatArray, ankle: FloatArray) -> FloatArray:
    """Sagittal knee flexion: 0 at full extension, positive in flexion."""
    return _STRAIGHT - included_angle(hip, knee, ankle)


def hip_flexion(shoulder: FloatArray, hip: FloatArray, knee: FloatArray) -> FloatArray:
    """Sagittal hip flexion as the trunk-thigh angle: 0 when in line, positive in flexion."""
    return _STRAIGHT - included_angle(shoulder, hip, knee)


def ankle_dorsiflexion(knee: FloatArray, ankle: FloatArray, foot_index: FloatArray) -> FloatArray:
    """Sagittal ankle dorsiflexion: positive when the shank moves over the foot."""
    return _RIGHT_ANGLE - included_angle(knee, ankle, foot_index)


def _medial_sign(side: Side, facing_camera: bool) -> float:
    """+1 if image +x is medial for this limb, else -1."""
    # Facing the camera, the athlete's left limb is on the image right, so medial is -x.
    sign = -1.0 if side == "left" else 1.0
    return sign if facing_camera else -sign


def knee_fppa(
    hip: FloatArray,
    knee: FloatArray,
    ankle: FloatArray,
    *,
    side: Side,
    facing_camera: bool = True,
) -> FloatArray:
    """Frontal-plane projection angle; positive = knee medial to the hip-ankle line (valgus)."""
    magnitude = _STRAIGHT - included_angle(hip, knee, ankle)
    # Horizontal offset of the knee from the straight line joining hip and ankle.
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = (knee[..., 1] - hip[..., 1]) / (ankle[..., 1] - hip[..., 1])
    line_x = hip[..., 0] + fraction * (ankle[..., 0] - hip[..., 0])
    medial = (knee[..., 0] - line_x) * _medial_sign(side, facing_camera)
    return np.asarray(np.sign(medial) * magnitude, dtype=np.float64)


def pelvic_drop(left_hip: FloatArray, right_hip: FloatArray, *, stance: Side) -> FloatArray:
    """Pelvic obliquity; positive when the hip opposite the stance limb is lower."""
    stance_hip, other_hip = (left_hip, right_hip) if stance == "left" else (right_hip, left_hip)
    lower_by = other_hip[..., 1] - stance_hip[..., 1]  # image y is down
    width = np.abs(other_hip[..., 0] - stance_hip[..., 0])
    return np.asarray(np.degrees(np.arctan2(lower_by, width)), dtype=np.float64)


def trunk_lateral_lean(
    left_shoulder: FloatArray,
    right_shoulder: FloatArray,
    left_hip: FloatArray,
    right_hip: FloatArray,
    *,
    stance: Side,
    facing_camera: bool = True,
) -> FloatArray:
    """Trunk angle from vertical; positive toward the stance limb."""
    mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
    mid_hip = 0.5 * (left_hip + right_hip)
    sideways = mid_shoulder[..., 0] - mid_hip[..., 0]
    upward = mid_hip[..., 1] - mid_shoulder[..., 1]  # image y is down
    toward_stance = -_medial_sign(stance, facing_camera)  # lateral for the stance limb
    return np.asarray(np.degrees(np.arctan2(sideways * toward_stance, upward)), dtype=np.float64)


def knee_ankle_separation_ratio(
    left_knee: FloatArray, right_knee: FloatArray, left_ankle: FloatArray, right_ankle: FloatArray
) -> FloatArray:
    """Horizontal knee separation / horizontal ankle separation (frontal camera)."""
    knees = np.abs(left_knee[..., 0] - right_knee[..., 0])
    ankles = np.abs(left_ankle[..., 0] - right_ankle[..., 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(ankles > 0, knees / ankles, np.nan), dtype=np.float64)
