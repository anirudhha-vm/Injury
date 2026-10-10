"""Stylised synthetic skeleton sequences with exactly known angles and events.

A planar stick figure is posed from prescribed joint-angle time courses and projected to a
sagittal and a frontal camera in the BlazePose layout. It exists to test the pose pipeline
(angles, events, features) against a known truth; it is not a model of how children move, and
nothing derived from it is a finding. Sequences are flagged ``synthetic=True``.

Construction (world: x forward, y up; pixels):

- Sagittal: shank tilted forward by ``a``, thigh back by ``b``, trunk forward by ``c``, with
  knee flexion = a + b, hip flexion = b + c, ankle dorsiflexion = a (the toe is placed level
  with the ankle so the foot segment is horizontal).
- Frontal: thigh and shank each tilted by half the prescribed FPPA, medially then laterally, so
  the frontal-plane projection angle equals the prescribed value at every knee flexion. Pelvic
  drop tilts the hip line; trunk lean tilts the trunk.
- The athlete faces the frontal camera, so their left side is on the image right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose.keypoints import Camera, KeypointSequence
from youthtwin.pose.skeletons import BLAZEPOSE, SIDES, Side

FloatArray = NDArray[np.float64]

# Stylised body, in pixels.
SHANK, THIGH, TRUNK = 180.0, 190.0, 230.0
LEG_LENGTH = SHANK + THIGH
HIP_HALF_WIDTH, SHOULDER_HALF_WIDTH = 60.0, 85.0
HEEL_BACK, TOE_FORWARD = 30.0, 80.0
IMAGE_SIZE = (1080, 1920)
FLOOR_Y, CENTRE_X = 1700.0, 540.0
# How total knee flexion is shared between shank and thigh tilt, and trunk lean per unit flexion.
SHANK_SHARE, TRUNK_SHARE = 0.45, 0.30


@dataclass(frozen=True)
class SyntheticTrial:
    """Two camera views of one trial plus the truth used to build them."""

    task: str
    limb: str  # "left", "right" or "both"
    sagittal: KeypointSequence
    frontal: KeypointSequence
    events: dict[str, int]  # true event frames
    angles: dict[str, FloatArray]  # true angle time courses, degrees, keyed "<name>_<side>"
    foot_height: FloatArray  # of the reference limb, fraction of leg length


def _ease(times: FloatArray, key_times: list[float], key_values: list[float]) -> FloatArray:
    """Smooth (cosine-eased) interpolation through key frames."""
    out = np.full(times.shape, key_values[-1], dtype=np.float64)
    out[times < key_times[0]] = key_values[0]
    for t0, t1, v0, v1 in zip(
        key_times[:-1], key_times[1:], key_values[:-1], key_values[1:], strict=True
    ):
        inside = (times >= t0) & (times < t1)
        phase = (times[inside] - t0) / (t1 - t0)
        out[inside] = v0 + (v1 - v0) * 0.5 * (1.0 - np.cos(np.pi * phase))
    return out


def _pose(
    *,
    knee_flexion: dict[Side, FloatArray],
    fppa: dict[Side, FloatArray],
    reference: Side,
    foot_height_px: FloatArray,
    pelvic_drop: FloatArray,
    trunk_lean: FloatArray,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Joint positions in sagittal and frontal pixel coordinates, each (T, 2)."""
    rad = np.radians
    a = {s: rad(SHANK_SHARE * knee_flexion[s]) for s in SIDES}
    b = {s: rad((1.0 - SHANK_SHARE) * knee_flexion[s]) for s in SIDES}
    c = rad(TRUNK_SHARE * knee_flexion[reference])
    pelvis_height = foot_height_px + SHANK * np.cos(a[reference]) + THIGH * np.cos(b[reference])
    # Reference ankle at x = 0: the knee is ahead of it, the pelvis behind the knee.
    pelvis_x = SHANK * np.sin(a[reference]) - THIGH * np.sin(b[reference])
    sagittal: dict[str, FloatArray] = {}
    frontal: dict[str, FloatArray] = {}
    other: Side = "right" if reference == "left" else "left"
    drop_px = 2.0 * HIP_HALF_WIDTH * np.tan(rad(pelvic_drop))

    def image(x: FloatArray | float, height: FloatArray | float) -> FloatArray:
        columns = np.broadcast_arrays(CENTRE_X + x, FLOOR_Y - height)
        return np.stack(columns, axis=-1).astype(np.float64)

    for side in SIDES:
        # --- sagittal chain, from the pelvis down ---
        hip = (pelvis_x, pelvis_height)
        knee = (hip[0] + THIGH * np.sin(b[side]), hip[1] - THIGH * np.cos(b[side]))
        ankle = (knee[0] - SHANK * np.sin(a[side]), knee[1] - SHANK * np.cos(a[side]))
        shoulder = (hip[0] + TRUNK * np.sin(c), hip[1] + TRUNK * np.cos(c))
        sagittal[f"{side}_hip"] = image(*hip)
        sagittal[f"{side}_knee"] = image(*knee)
        sagittal[f"{side}_ankle"] = image(*ankle)
        sagittal[f"{side}_heel"] = image(ankle[0] - HEEL_BACK, ankle[1])
        sagittal[f"{side}_foot_index"] = image(ankle[0] + TOE_FORWARD, ankle[1])
        sagittal[f"{side}_shoulder"] = image(*shoulder)

        # --- frontal: left side on the image right; medial is toward the centre ---
        outward = 1.0 if side == "left" else -1.0
        half = rad(fppa[side]) / 2.0
        hip_height = pelvis_height - (drop_px if side == other else 0.0)
        hip_x = outward * HIP_HALF_WIDTH
        thigh_drop, shank_drop = THIGH * np.cos(b[side]), SHANK * np.cos(a[side])
        knee_x = hip_x - outward * thigh_drop * np.tan(half)  # medial
        ankle_x = knee_x + outward * shank_drop * np.tan(half)  # back out laterally
        ankle_height = hip_height - thigh_drop - shank_drop
        frontal[f"{side}_hip"] = image(hip_x, hip_height)
        frontal[f"{side}_knee"] = image(knee_x, hip_height - thigh_drop)
        frontal[f"{side}_ankle"] = image(ankle_x, ankle_height)
        frontal[f"{side}_heel"] = image(ankle_x, ankle_height)
        frontal[f"{side}_foot_index"] = image(ankle_x + outward * 10.0, ankle_height)

    # Trunk in the frontal view: lean is positive toward the reference (stance) limb.
    toward = 1.0 if reference == "left" else -1.0
    trunk_up = TRUNK * np.cos(c)
    mid_hip_height = pelvis_height - 0.5 * drop_px
    mid_shoulder_x = toward * trunk_up * np.tan(rad(trunk_lean))
    for side in SIDES:
        outward = 1.0 if side == "left" else -1.0
        frontal[f"{side}_shoulder"] = image(
            mid_shoulder_x + outward * SHOULDER_HALF_WIDTH, mid_hip_height + trunk_up
        )
    return sagittal, frontal


def _sequence(
    joints: dict[str, FloatArray],
    camera: Camera,
    timestamps_ms: FloatArray,
    rng: np.random.Generator,
    noise_px: float,
    dropout: FloatArray | None,
) -> KeypointSequence:
    n_frames = timestamps_ms.size
    xy = np.full((n_frames, BLAZEPOSE.n_landmarks, 2), np.nan)
    conf = np.zeros((n_frames, BLAZEPOSE.n_landmarks))
    for name, index in BLAZEPOSE.landmarks.items():
        xy[:, index] = joints[name] + rng.normal(0.0, noise_px, (n_frames, 2))
        conf[:, index] = 0.95
    if dropout is not None:
        conf[dropout.astype(bool)] = 0.1
    return KeypointSequence(
        xy=xy,
        conf=conf,
        timestamps_ms=timestamps_ms,
        skeleton=BLAZEPOSE.name,
        camera=camera,
        image_size=IMAGE_SIZE,
        estimator="synthetic",
        synthetic=True,
    )


def _build(
    task: str,
    limb: str,
    reference: Side,
    times: FloatArray,
    rng: np.random.Generator,
    noise_px: float,
    dropout: FloatArray | None,
    events: dict[str, int],
    **pose: Any,
) -> SyntheticTrial:
    sagittal, frontal = _pose(reference=reference, **pose)
    timestamps_ms = 1000.0 * times
    flexion = pose["knee_flexion"]
    angles: dict[str, FloatArray] = {
        "pelvic_drop": pose["pelvic_drop"],
        "trunk_lateral_lean": pose["trunk_lean"],
    }
    for side in SIDES:
        angles[f"knee_flexion_{side}"] = flexion[side]
        angles[f"hip_flexion_{side}"] = (1.0 - SHANK_SHARE) * flexion[side] + (
            TRUNK_SHARE * flexion[reference]
        )
        angles[f"ankle_dorsiflexion_{side}"] = SHANK_SHARE * flexion[side]
        angles[f"shank_inclination_{side}"] = SHANK_SHARE * flexion[side]
        angles[f"thigh_inclination_{side}"] = (1.0 - SHANK_SHARE) * flexion[side]
        angles[f"knee_fppa_{side}"] = pose["fppa"][side]
    return SyntheticTrial(
        task=task,
        limb=limb,
        sagittal=_sequence(sagittal, "sagittal", timestamps_ms, rng, noise_px, dropout),
        frontal=_sequence(frontal, "frontal", timestamps_ms, rng, noise_px, dropout),
        events=events,
        angles=angles,
        foot_height=np.asarray(pose["foot_height_px"] / LEG_LENGTH, dtype=np.float64),
    )


def simulate_drop_vertical_jump(
    rng: np.random.Generator,
    *,
    fps: float = 120.0,
    peak_knee_flexion: float = 85.0,
    peak_fppa: tuple[float, float] = (12.0, 6.0),  # (left, right), degrees, positive = valgus
    box_height: float = 0.30,  # fraction of leg length
    jump_height: float = 0.25,
    noise_px: float = 0.0,
    dropout: FloatArray | None = None,
) -> SyntheticTrial:
    """A drop vertical jump: on the box, drop, land, jump, land again."""
    t_drop, t_contact, t_peak, t_take_off, t_land, t_end = 0.50, 0.75, 0.95, 1.20, 1.60, 2.20
    times = np.arange(0.0, t_end, 1.0 / fps)
    height = np.zeros_like(times)
    height[times < t_drop] = box_height
    falling = (times >= t_drop) & (times < t_contact)
    height[falling] = box_height * (1.0 - ((times[falling] - t_drop) / (t_contact - t_drop)) ** 2)
    flight = (times >= t_take_off) & (times < t_land)
    phase = (times[flight] - t_take_off) / (t_land - t_take_off)
    height[flight] = jump_height * 4.0 * phase * (1.0 - phase)
    flexion = _ease(
        times,
        [t_drop, t_contact, t_peak, t_take_off, t_land, t_land + 0.25, t_end],
        [8.0, 20.0, peak_knee_flexion, 10.0, 22.0, 60.0, 15.0],
    )
    share = flexion / peak_knee_flexion
    stance = (times >= t_contact) & (times < t_take_off)
    events = {
        "box_drop": int(np.argmax(height < box_height * 0.85)),
        "initial_contact": int(np.argmax(times >= t_contact)),
        "peak_knee_flexion": int(np.flatnonzero(stance)[np.argmax(flexion[stance])]),
        "take_off": int(np.argmax(times >= t_take_off)),
        "second_landing": int(np.argmax(times >= t_land)),
    }
    zeros = np.zeros_like(times)
    return _build(
        "drop_vertical_jump",
        "both",
        "left",
        times,
        rng,
        noise_px,
        dropout,
        events,
        knee_flexion={"left": flexion, "right": flexion},
        fppa={"left": peak_fppa[0] * share, "right": peak_fppa[1] * share},
        foot_height_px=height * LEG_LENGTH,
        pelvic_drop=zeros,
        trunk_lean=zeros,
    )


def simulate_single_leg_squat(
    rng: np.random.Generator,
    *,
    limb: Side,
    fps: float = 120.0,
    peak_knee_flexion: float = 65.0,
    peak_fppa: float = 10.0,
    peak_pelvic_drop: float = 6.0,
    peak_trunk_lean: float = 5.0,
    noise_px: float = 0.0,
    dropout: FloatArray | None = None,
) -> SyntheticTrial:
    """A single-leg squat on ``limb``; the other leg is held flexed off the floor."""
    t_start, t_deep, t_return, t_end = 0.50, 1.50, 2.50, 3.00
    times = np.arange(0.0, t_end, 1.0 / fps)
    flexion = _ease(times, [t_start, t_deep, t_return], [5.0, peak_knee_flexion, 5.0])
    share = (flexion - 5.0) / (peak_knee_flexion - 5.0)
    other: Side = "right" if limb == "left" else "left"
    zeros = np.zeros_like(times)
    events = {
        "start": int(np.argmax(times >= t_start)),
        "deepest_point": int(np.argmax(flexion)),
        "return": int(np.argmax(times >= t_return)),
    }
    return _build(
        "single_leg_squat",
        limb,
        limb,
        times,
        rng,
        noise_px,
        dropout,
        events,
        knee_flexion={limb: flexion, other: flexion + 60.0},
        fppa={limb: peak_fppa * share, other: zeros},
        foot_height_px=zeros,
        pelvic_drop=peak_pelvic_drop * share,
        trunk_lean=peak_trunk_lean * share,
    )
