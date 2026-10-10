"""Task event detection from foot height and knee flexion.

Detectors return frame indices with a confidence in [0, 1]. An event that cannot be found is
reported in ``EventSet.missing`` with the reason; it is never guessed. A manual-override table
can replace any event (confidence 1, source ``manual``).

Signals
-------
- ``foot_height``: height of the feet above the floor as a fraction of leg length (see
  ``normalised_foot_height``), so thresholds do not depend on camera distance.
- ``knee_flexion``: sagittal knee flexion in degrees.

Tasks
-----
- drop vertical jump: ``box_drop, initial_contact, peak_knee_flexion, take_off, second_landing``
- single-leg squat: ``start, deepest_point, return``
- single-leg hop: ``take_off, landing``

Thresholds come from ``configs/pose/events.yaml``.
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose.keypoints import KeypointSequence
from youthtwin.pose.skeletons import Side

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DVJ_EVENTS = ("box_drop", "initial_contact", "peak_knee_flexion", "take_off", "second_landing")
SQUAT_EVENTS = ("start", "deepest_point", "return")
HOP_EVENTS = ("take_off", "landing")
_FLOOR_PERCENTILE = 98.0  # image y is down, so the floor is at the largest foot y


@dataclass(frozen=True)
class Event:
    name: str
    frame: int
    time_ms: float
    confidence: float
    source: str = "auto"


@dataclass(frozen=True)
class EventSet:
    task: str
    events: dict[str, Event] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    def frame(self, name: str) -> int:
        return self.events[name].frame

    @property
    def min_confidence(self) -> float:
        """Lowest event confidence; 0 if any expected event is missing."""
        if self.missing or not self.events:
            return 0.0
        return min(event.confidence for event in self.events.values())


def leg_length_px(sequence: KeypointSequence, sides: Sequence[Side]) -> float:
    """Median hip-knee-ankle length in pixels over frames and the given sides."""
    lengths = [
        np.linalg.norm(sequence.joint(s, "hip") - sequence.joint(s, "knee"), axis=1)
        + np.linalg.norm(sequence.joint(s, "knee") - sequence.joint(s, "ankle"), axis=1)
        for s in sides
    ]
    return float(np.nanmedian(np.concatenate(lengths)))


def normalised_foot_height(sequence: KeypointSequence, sides: Sequence[Side]) -> FloatArray:
    """Height of the feet above the floor, as a fraction of leg length.

    Uses the mean of heel and toe of the given sides. The floor is the lowest level the feet
    reach in the clip.
    """
    with warnings.catch_warnings():  # frames where every foot landmark is missing stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        foot_y = np.nanmean(
            [sequence.joint(s, part)[:, 1] for s in sides for part in ("heel", "foot_index")],
            axis=0,
        )
    floor_y = float(np.nanpercentile(foot_y, _FLOOR_PERCENTILE))
    return np.asarray((floor_y - foot_y) / leg_length_px(sequence, sides), dtype=np.float64)


def _runs(mask: BoolArray, min_length: int) -> list[tuple[int, int]]:
    """``(start, stop)`` of every run of True at least ``min_length`` long (stop exclusive)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    return [
        (int(a), int(b))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if b - a >= min_length
    ]


def _confidence(signal: FloatArray, frame: int, window: int, margin: float) -> float:
    """Share of finite samples around ``frame`` times how clearly the threshold was crossed."""
    near = signal[max(0, frame - window) : frame + window + 1]
    return float(np.mean(np.isfinite(near)) * np.clip(margin, 0.0, 1.0))


def _event(name: str, frame: int, timestamps_ms: FloatArray, confidence: float) -> Event:
    return Event(name, int(frame), float(timestamps_ms[frame]), float(confidence))


def detect_drop_vertical_jump(
    foot_height: FloatArray,
    knee_flexion: FloatArray,
    timestamps_ms: FloatArray,
    config: Mapping[str, Any],
) -> EventSet:
    """Box drop, initial contact, peak knee flexion, take-off and second landing."""
    contact_cfg, dvj = config["contact"], config["drop_vertical_jump"]
    threshold = float(contact_cfg["height_threshold"])
    window = int(config["confidence_window_frames"])
    missing: dict[str, str] = {}
    events: dict[str, Event] = {}

    box_height = float(np.nanmedian(foot_height[: int(config["baseline_frames"])]))
    if not box_height >= float(dvj["min_box_height"]):
        reason = "feet are not above the floor at the start (no box phase found)"
        return EventSet("drop_vertical_jump", {}, dict.fromkeys(DVJ_EVENTS, reason))

    in_contact = np.nan_to_num(foot_height, nan=np.inf) < threshold
    contacts = _runs(in_contact, int(contact_cfg["min_contact_frames"]))
    if not contacts:
        reason = "the feet never reach the floor"
        return EventSet("drop_vertical_jump", {}, dict.fromkeys(DVJ_EVENTS, reason))

    first_start, first_stop = contacts[0]
    drop_margin = box_height / (2.0 * threshold)
    events["initial_contact"] = _event(
        "initial_contact",
        first_start,
        timestamps_ms,
        _confidence(foot_height, first_start, window, drop_margin),
    )
    dropped = np.flatnonzero(
        foot_height[:first_start] > box_height * (1.0 - float(dvj["box_drop_fraction"]))
    )
    if dropped.size and dropped[-1] + 1 < first_start:
        frame = int(dropped[-1] + 1)
        events["box_drop"] = _event(
            "box_drop", frame, timestamps_ms, _confidence(foot_height, frame, window, drop_margin)
        )
    else:
        missing["box_drop"] = "no descent from the box before initial contact"

    if first_stop >= foot_height.size:
        missing["take_off"] = "no take-off after landing (clip ends in contact)"
        missing["second_landing"] = missing["take_off"]
        stance_stop = foot_height.size
    else:
        stance_stop = first_stop
        later = [
            c for c in contacts[1:] if c[0] - first_stop >= int(contact_cfg["min_flight_frames"])
        ]
        flight_stop = later[0][0] if later else foot_height.size
        flight_peak = float(np.nanmax(foot_height[first_stop:flight_stop], initial=0.0))
        jump_margin = flight_peak / (2.0 * threshold)
        events["take_off"] = _event(
            "take_off",
            first_stop,
            timestamps_ms,
            _confidence(foot_height, first_stop, window, jump_margin),
        )
        if later:
            events["second_landing"] = _event(
                "second_landing",
                later[0][0],
                timestamps_ms,
                _confidence(foot_height, later[0][0], window, jump_margin),
            )
        else:
            missing["second_landing"] = "no landing after take-off"

    stance = knee_flexion[first_start:stance_stop]
    if np.isfinite(stance).any():
        frame = first_start + int(np.nanargmax(stance))
        events["peak_knee_flexion"] = _event(
            "peak_knee_flexion", frame, timestamps_ms, float(np.mean(np.isfinite(stance)))
        )
    else:
        missing["peak_knee_flexion"] = "knee flexion unavailable during stance"
    return EventSet("drop_vertical_jump", events, missing)


def detect_single_leg_squat(
    knee_flexion: FloatArray, timestamps_ms: FloatArray, config: Mapping[str, Any]
) -> EventSet:
    """Start of descent, deepest point and return to standing, from knee flexion."""
    squat = config["single_leg_squat"]
    window = int(config["confidence_window_frames"])
    reason_all = None
    if not np.isfinite(knee_flexion).any():
        reason_all = "knee flexion unavailable"
    else:
        baseline = float(np.nanmedian(knee_flexion[: int(config["baseline_frames"])]))
        deepest = int(np.nanargmax(knee_flexion))
        depth = float(knee_flexion[deepest] - baseline)
        if not depth >= float(squat["min_depth_deg"]):
            reason_all = f"squat depth {depth:.0f} deg is below the minimum"
    if reason_all is not None:
        return EventSet("single_leg_squat", {}, dict.fromkeys(SQUAT_EVENTS, reason_all))

    standing = np.nan_to_num(knee_flexion, nan=np.inf) <= baseline + float(squat["onset_deg"])
    margin = depth / (2.0 * float(squat["min_depth_deg"]))
    events = {
        "deepest_point": _event(
            "deepest_point",
            deepest,
            timestamps_ms,
            _confidence(knee_flexion, deepest, window, margin),
        )
    }
    missing: dict[str, str] = {}
    before, after = np.flatnonzero(standing[:deepest]), np.flatnonzero(standing[deepest:])
    if before.size:
        events["start"] = _event(
            "start",
            int(before[-1]),
            timestamps_ms,
            _confidence(knee_flexion, int(before[-1]), window, margin),
        )
    else:
        missing["start"] = "clip starts with the knee already flexed"
    if after.size:
        frame = deepest + int(after[0])
        events["return"] = _event(
            "return", frame, timestamps_ms, _confidence(knee_flexion, frame, window, margin)
        )
    else:
        missing["return"] = "no return to standing before the clip ends"
    return EventSet("single_leg_squat", events, missing)


def detect_hop(
    foot_height: FloatArray, timestamps_ms: FloatArray, config: Mapping[str, Any]
) -> EventSet:
    """Take-off and landing of a single-leg hop from the stance foot's height."""
    contact_cfg = config["contact"]
    threshold = float(contact_cfg["height_threshold"])
    window = int(config["confidence_window_frames"])
    in_contact = np.nan_to_num(foot_height, nan=-np.inf) < threshold
    flights = _runs(~in_contact & np.isfinite(foot_height), int(contact_cfg["min_flight_frames"]))
    flights = [f for f in flights if f[0] > 0]  # must start standing
    if not flights:
        reason = "no flight phase found"
        return EventSet("single_leg_hop_distance", {}, dict.fromkeys(HOP_EVENTS, reason))
    start, stop = flights[0]
    margin = float(np.nanmax(foot_height[start:stop])) / (2.0 * threshold)
    events = {
        "take_off": _event(
            "take_off", start, timestamps_ms, _confidence(foot_height, start, window, margin)
        )
    }
    missing: dict[str, str] = {}
    if stop < foot_height.size:
        events["landing"] = _event(
            "landing", stop, timestamps_ms, _confidence(foot_height, stop, window, margin)
        )
    else:
        missing["landing"] = "no landing before the clip ends"
    return EventSet("single_leg_hop_distance", events, missing)


def apply_overrides(
    detected: EventSet, overrides: Mapping[str, int], timestamps_ms: FloatArray
) -> EventSet:
    """Replace or add events with manually marked frames (confidence 1, source ``manual``)."""
    events, missing = dict(detected.events), dict(detected.missing)
    for name, frame in overrides.items():
        if not 0 <= frame < timestamps_ms.size:
            raise ValueError(f"Override frame {frame} for '{name}' is outside the clip")
        events[name] = Event(name, int(frame), float(timestamps_ms[frame]), 1.0, source="manual")
        missing.pop(name, None)
    return replace(detected, events=events, missing=missing)


def read_overrides(path: Path) -> dict[str, dict[str, int]]:
    """Read a manual-override CSV with columns ``trial_id, event, frame``."""
    table: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table.setdefault(row["trial_id"], {})[row["event"]] = int(row["frame"])
    return table
