"""From keypoints to tidy task features, with quality control.

Per trial: fill gaps and filter keypoints, compute joint angles, detect events, then take
angles at events, ranges of motion and peak angular velocities within the task window, and
101-point time-normalised curves. Per session: the mean over valid trials and between-limb
asymmetry.

Asymmetry (v2 defect: frame-wise left/right differences during non-bilateral tasks) is computed
**only from event- and window-level features**: within the same trial for bilateral tasks, and
between left-limb trials and right-limb trials for unilateral tasks. There is no frame-wise
asymmetry function.

Quality control never drops data silently: every trial gets a quality score and, if excluded, a
row in the exclusion table saying why.

Output rows: ``athlete_id, session_id, task, trial, limb, feature, value, quality`` (+
``synthetic``), with ``trial`` empty for session-level rows.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.pose import events as ev
from youthtwin.pose import filtering, kinematics
from youthtwin.pose.keypoints import KeypointSequence
from youthtwin.pose.skeletons import JOINTS, SIDES, Side

FloatArray = NDArray[np.float64]

SAGITTAL_ANGLES = (
    "knee_flexion",
    "hip_flexion",
    "ankle_dorsiflexion",
    "thigh_inclination",
    "shank_inclination",
)
FEATURE_COLUMNS = [
    "athlete_id",
    "session_id",
    "task",
    "trial",
    "limb",
    "feature",
    "value",
    "quality",
    "synthetic",
]
EXCLUSION_COLUMNS = ["athlete_id", "session_id", "task", "trial", "limb", "reason", "quality"]
# Clip-level body ratios: stored per limb but not movement features, so no asymmetry is formed.
BODY_PROPORTIONS = ("thigh_length_fraction", "forefoot_heel_lever_ratio")
# Task -> (window start event, window end event).
TASK_WINDOWS: dict[str, tuple[str, str]] = {
    "drop_vertical_jump": ("initial_contact", "take_off"),
    "single_leg_squat": ("start", "return"),
    "single_leg_hop_distance": ("take_off", "landing"),
}


@dataclass(frozen=True)
class TrialInput:
    athlete_id: str
    session_id: str
    task: str
    trial: int
    limb: str  # "left", "right" or "both"
    sagittal: KeypointSequence | None = None
    frontal: KeypointSequence | None = None
    frontal_offset_ms: float = 0.0  # added to frontal timestamps to match the sagittal clock
    event_overrides: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    trial: TrialInput
    features: dict[tuple[str, str], float]  # (limb, feature) -> value
    curves: dict[tuple[str, str], FloatArray]  # (limb, angle) -> 101-point curve
    events: ev.EventSet | None
    quality: float
    excluded_reason: str | None

    @property
    def included(self) -> bool:
        return self.excluded_reason is None


# ------------------------------------------------------------------------------ preprocessing


def preprocess(
    sequence: KeypointSequence, config: Mapping[str, Any]
) -> tuple[KeypointSequence, float]:
    """Gap-fill and low-pass every canonical joint. Returns the sequence and its gap fraction.

    The gap fraction is the share of (frame, joint) samples that were low-confidence or missing
    before filling. Unevenly timed sequences are resampled to an even grid first.
    """
    gap, butter = config["gap_fill"], config["butterworth"]
    layout = sequence.layout
    used = [layout.index(side, joint) for side in SIDES for joint in JOINTS]
    xy, conf, timestamps_ms = sequence.xy, sequence.conf, sequence.timestamps_ms
    if not filtering.is_uniform(sequence.time_s):
        xy, grid = filtering.resample_uniform(xy, timestamps_ms / 1000.0)
        nearest = np.clip(np.searchsorted(timestamps_ms / 1000.0, grid), 0, conf.shape[0] - 1)
        conf, timestamps_ms = conf[nearest], grid * 1000.0
    fs = 1000.0 / float(np.median(np.diff(timestamps_ms)))
    out = np.full_like(xy, np.nan)
    missing_share: list[float] = []
    for index in used:
        filled, missing = filtering.fill_gaps(
            xy[:, index],
            conf[:, index],
            min_confidence=float(gap["min_confidence"]),
            max_gap=int(gap["max_linear_gap_frames"]),
        )
        out[:, index] = filtering.butterworth_lowpass(
            filled, fs=fs, cutoff_hz=float(butter["cutoff_hz"])
        )
        missing_share.append(float(missing.mean()))
    cleaned = KeypointSequence(
        xy=out,
        conf=conf,
        timestamps_ms=timestamps_ms,
        skeleton=sequence.skeleton,
        camera=sequence.camera,
        image_size=sequence.image_size,
        estimator=sequence.estimator,
        synthetic=sequence.synthetic,
    )
    return cleaned, float(np.mean(missing_share))


def sagittal_angles(sequence: KeypointSequence, side: Side) -> dict[str, FloatArray]:
    """Knee flexion, hip flexion and ankle dorsiflexion (degrees) from the side camera."""
    if sequence.camera != "sagittal":
        raise ValueError("Sagittal angles need the side camera")
    j = sequence.joint
    return {
        "knee_flexion": kinematics.knee_flexion(j(side, "hip"), j(side, "knee"), j(side, "ankle")),
        "hip_flexion": kinematics.hip_flexion(j(side, "shoulder"), j(side, "hip"), j(side, "knee")),
        "ankle_dorsiflexion": kinematics.ankle_dorsiflexion(
            j(side, "knee"), j(side, "ankle"), j(side, "foot_index")
        ),
        "thigh_inclination": kinematics.segment_inclination(j(side, "hip"), j(side, "knee")),
        "shank_inclination": kinematics.segment_inclination(j(side, "knee"), j(side, "ankle")),
    }


def body_proportions(sequence: KeypointSequence, side: Side) -> dict[str, float]:
    """Camera-independent body ratios from the side view (medians over the clip).

    - ``thigh_length_fraction``: thigh / (thigh + shank) projected length.
    - ``forefoot_heel_lever_ratio``: horizontal ankle-to-toe distance / horizontal ankle-to-heel
      distance, a geometric stand-in for forefoot lever / Achilles lever.
    """
    j = sequence.joint
    thigh = kinematics.segment_length(j(side, "hip"), j(side, "knee"))
    shank = kinematics.segment_length(j(side, "knee"), j(side, "ankle"))
    forefoot = np.abs(j(side, "foot_index")[:, 0] - j(side, "ankle")[:, 0])
    heel = np.abs(j(side, "heel")[:, 0] - j(side, "ankle")[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        lever = np.where(heel > 0, forefoot / heel, np.nan)
    return {
        "thigh_length_fraction": float(np.nanmedian(thigh / (thigh + shank))),
        "forefoot_heel_lever_ratio": float(np.nanmedian(lever)),
    }


def frontal_angles(
    sequence: KeypointSequence, side: Side, *, unilateral: bool, facing_camera: bool = True
) -> dict[str, FloatArray]:
    """Knee FPPA, plus pelvic drop and trunk lean for unilateral tasks (stance = ``side``)."""
    if sequence.camera != "frontal":
        raise ValueError("Frontal-plane angles need the front camera")
    j = sequence.joint
    angles = {
        "knee_fppa": kinematics.knee_fppa(
            j(side, "hip"),
            j(side, "knee"),
            j(side, "ankle"),
            side=side,
            facing_camera=facing_camera,
        )
    }
    if unilateral:
        angles["pelvic_drop"] = kinematics.pelvic_drop(
            j("left", "hip"), j("right", "hip"), stance=side
        )
        angles["trunk_lateral_lean"] = kinematics.trunk_lateral_lean(
            j("left", "shoulder"),
            j("right", "shoulder"),
            j("left", "hip"),
            j("right", "hip"),
            stance=side,
            facing_camera=facing_camera,
        )
    return angles


def time_normalise(values: FloatArray, start: int, stop: int, n_points: int) -> FloatArray:
    """Resample ``values[start:stop + 1]`` to ``n_points`` evenly spaced points."""
    segment = values[start : stop + 1]
    if segment.size < 2:
        return np.full(n_points, np.nan)
    return np.asarray(
        np.interp(np.linspace(0, segment.size - 1, n_points), np.arange(segment.size), segment),
        dtype=np.float64,
    )


# ------------------------------------------------------------------------------ one trial


def _detect(
    task: str,
    foot_height: FloatArray,
    knee_flexion: FloatArray,
    timestamps_ms: FloatArray,
    config: Mapping[str, Any],
) -> ev.EventSet:
    if task == "drop_vertical_jump":
        return ev.detect_drop_vertical_jump(foot_height, knee_flexion, timestamps_ms, config)
    if task == "single_leg_squat":
        return ev.detect_single_leg_squat(knee_flexion, timestamps_ms, config)
    return ev.detect_hop(foot_height, timestamps_ms, config)


def _excluded(trial: TrialInput, reason: str, quality: float = 0.0) -> TrialResult:
    return TrialResult(trial, {}, {}, None, quality, reason)


def trial_features(
    trial: TrialInput,
    *,
    filter_config: Mapping[str, Any],
    event_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> TrialResult:
    """Features, curves, events and quality for one trial."""
    if trial.task not in TASK_WINDOWS:
        return _excluded(trial, f"task '{trial.task}' is not supported by the feature pipeline")
    if trial.sagittal is None:
        return _excluded(trial, "no sagittal view: events cannot be detected")
    unilateral = trial.limb != "both"
    sides: tuple[Side, ...] = SIDES if not unilateral else (trial.limb,)  # type: ignore[assignment]

    sagittal, gap_sagittal = preprocess(trial.sagittal, filter_config)
    time_s = sagittal.time_s
    angles = {side: sagittal_angles(sagittal, side) for side in sides}
    with warnings.catch_warnings():  # frames with no usable knee angle stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        reference_flexion = np.nanmean([angles[s]["knee_flexion"] for s in sides], axis=0)
    detected = _detect(
        trial.task,
        ev.normalised_foot_height(sagittal, sides),
        reference_flexion,
        sagittal.timestamps_ms,
        event_config,
    )
    detected = ev.apply_overrides(detected, trial.event_overrides, sagittal.timestamps_ms)

    confidence = [float(np.nanmean(trial.sagittal.joint_conf(s, j))) for s in sides for j in JOINTS]
    gaps = [gap_sagittal]
    frontal = None
    if trial.frontal is not None:
        frontal, gap_frontal = preprocess(trial.frontal, filter_config)
        gaps.append(gap_frontal)
        confidence += [
            float(np.nanmean(trial.frontal.joint_conf(s, j))) for s in sides for j in JOINTS
        ]
    quality = min(float(np.mean(confidence)), 1.0 - max(gaps), detected.min_confidence)

    window_start, window_end = TASK_WINDOWS[trial.task]
    if window_start not in detected.events or window_end not in detected.events:
        absent = [name for name in (window_start, window_end) if name not in detected.events]
        reasons = "; ".join(f"{name}: {detected.missing.get(name, 'not found')}" for name in absent)
        return TrialResult(trial, {}, {}, detected, 0.0, f"events not found ({reasons})")
    start, stop = detected.frame(window_start), detected.frame(window_end)
    n_points = int(feature_config["curve_points"])

    features: dict[tuple[str, str], float] = {}
    curves: dict[tuple[str, str], FloatArray] = {}

    def add(
        limb: str,
        series: Mapping[str, FloatArray],
        frames: Mapping[str, int],
        lo: int,
        hi: int,
        clock: FloatArray,
        with_velocity: bool,
    ) -> None:
        for name, values in series.items():
            for event_name, frame in frames.items():
                features[(limb, f"{name}_at_{event_name}")] = float(values[frame])
            window = values[lo : hi + 1]
            if np.isfinite(window).any():
                features[(limb, f"{name}_max")] = float(np.nanmax(window))
                features[(limb, f"{name}_rom")] = float(np.nanmax(window) - np.nanmin(window))
                if with_velocity:
                    velocity = filtering.derivative(values, clock)[lo : hi + 1]
                    features[(limb, f"{name}_peak_velocity")] = float(np.nanmax(np.abs(velocity)))
            curves[(limb, name)] = time_normalise(values, lo, hi, n_points)

    frames = {name: event.frame for name, event in detected.events.items()}
    for side in sides:
        add(side, angles[side], frames, start, stop, time_s, with_velocity=True)
        for name, value in body_proportions(sagittal, side).items():
            features[(side, name)] = value

    if frontal is not None:
        # Map event times on the sagittal clock to the nearest frontal frame.
        frontal_clock = frontal.timestamps_ms + trial.frontal_offset_ms
        frontal_frames = {
            name: int(np.argmin(np.abs(frontal_clock - event.time_ms)))
            for name, event in detected.events.items()
        }
        lo, hi = frontal_frames[window_start], frontal_frames[window_end]
        for side in sides:
            add(
                side,
                frontal_angles(frontal, side, unilateral=unilateral),
                frontal_frames,
                lo,
                hi,
                frontal.time_s,
                with_velocity=False,
            )
        if not unilateral:
            j = frontal.joint
            ratio = kinematics.knee_ankle_separation_ratio(
                j("left", "knee"), j("right", "knee"), j("left", "ankle"), j("right", "ankle")
            )
            for event_name, frame in frontal_frames.items():
                features[("both", f"knee_ankle_separation_at_{event_name}")] = float(ratio[frame])
            features[("both", "knee_ankle_separation_min")] = float(np.nanmin(ratio[lo : hi + 1]))

    features[(sides[0] if unilateral else "both", "window_duration_s")] = float(
        time_s[stop] - time_s[start]
    )
    threshold = float(feature_config["quality"]["min_trial_quality"])
    reason = None if quality >= threshold else f"quality {quality:.2f} below threshold {threshold}"
    return TrialResult(trial, features, curves, detected, quality, reason)


# ------------------------------------------------------------------------------ asymmetry


def symmetry_angle(left: float, right: float) -> float:
    """Symmetry angle (Zifchock et al. 2008), percent: 0 = symmetric, sign = larger side (+ right).

    ``(45 deg - atan2(left, right)) / 90 deg * 100``, wrapped so opposite-sign pairs are handled.
    """
    angle = 45.0 - float(np.degrees(np.arctan2(left, right)))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle / 90.0 * 100.0


def limb_symmetry_index(left: float, right: float) -> float:
    """Smaller / larger magnitude x 100 (100 = symmetric). NaN if signs differ or both are zero."""
    if left * right <= 0:
        return float("nan")
    return 100.0 * min(abs(left), abs(right)) / max(abs(left), abs(right))


def asymmetry_value(left: float, right: float, index: str) -> float:
    """Between-limb asymmetry of one event- or window-level feature."""
    if index == "symmetry_angle":
        return symmetry_angle(left, right)
    if index == "lsi":
        return limb_symmetry_index(left, right)
    if index == "difference":
        return left - right
    raise ValueError(f"Unknown asymmetry index '{index}'")


# ------------------------------------------------------------------------------ one session


def session_features(
    results: Sequence[TrialResult], feature_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial rows, session means, event-based asymmetry, and the table of excluded trials."""
    index_name = str(feature_config["asymmetry"]["index"])
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for result in results:
        trial = result.trial
        base = {
            "athlete_id": trial.athlete_id,
            "session_id": trial.session_id,
            "task": trial.task,
            "synthetic": _synthetic(trial),
        }
        if not result.included:
            exclusions.append(
                {
                    **{k: base[k] for k in ("athlete_id", "session_id", "task")},
                    "trial": trial.trial,
                    "limb": trial.limb,
                    "reason": result.excluded_reason,
                    "quality": result.quality,
                }
            )
            continue
        for (limb, feature), value in result.features.items():
            rows.append(
                {
                    **base,
                    "trial": trial.trial,
                    "limb": limb,
                    "feature": feature,
                    "value": value,
                    "quality": result.quality,
                }
            )
    trials = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    excluded = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    if trials.empty:
        return trials, excluded

    keys = ["athlete_id", "session_id", "task", "limb", "feature"]
    means = trials.groupby(keys, as_index=False).agg(
        value=("value", "mean"), quality=("quality", "mean"), synthetic=("synthetic", "all")
    )
    means["trial"] = None
    comparable = means["limb"].isin(SIDES) & ~means["feature"].isin(BODY_PROPORTIONS)
    sided = means[comparable].pivot_table(
        index=["athlete_id", "session_id", "task", "feature"], columns="limb", values="value"
    )
    asymmetry_rows: list[dict[str, Any]] = []
    if {"left", "right"} <= set(sided.columns):
        quality = means.groupby(["athlete_id", "session_id", "task"])["quality"].mean()
        flags = means.groupby(["athlete_id", "session_id", "task"])["synthetic"].all()
        for key, pair in sided.dropna().iterrows():
            athlete, session, task, feature = cast(tuple[str, str, str, str], key)
            asymmetry_rows.append(
                {
                    "athlete_id": athlete,
                    "session_id": session,
                    "task": task,
                    "trial": None,
                    "limb": "both",
                    "feature": f"asymmetry_{feature}",
                    "value": asymmetry_value(float(pair["left"]), float(pair["right"]), index_name),
                    "quality": float(quality.loc[(athlete, session, task)]),
                    "synthetic": bool(flags.loc[(athlete, session, task)]),
                }
            )
    table = pd.concat(
        [trials, means[FEATURE_COLUMNS], pd.DataFrame(asymmetry_rows, columns=FEATURE_COLUMNS)],
        ignore_index=True,
    )
    return table, excluded


def _synthetic(trial: TrialInput) -> bool:
    views = [v for v in (trial.sagittal, trial.frontal) if v is not None]
    return bool(views) and all(v.synthetic for v in views)
