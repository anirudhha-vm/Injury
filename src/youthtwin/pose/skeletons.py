"""Landmark indices: the single source of truth for every pose layout used in the project.

v2 defect: the knee was computed from landmarks 11/13/15 (shoulder, elbow, wrist) and the ankle
from 13/14. Everything downstream therefore refers to joints by **canonical name and side**
through a ``Skeleton``; raw indices appear only in this file.

Canonical joints per side: ``shoulder, hip, knee, ankle, heel, foot_index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
SIDES: tuple[Side, ...] = ("left", "right")
JOINTS: tuple[str, ...] = ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")


@dataclass(frozen=True)
class Skeleton:
    """A pose layout: its name, landmark count, and canonical joint name -> index."""

    name: str
    n_landmarks: int
    landmarks: dict[str, int]

    def __post_init__(self) -> None:
        expected = {f"{side}_{joint}" for side in SIDES for joint in JOINTS}
        if set(self.landmarks) != expected:
            raise ValueError(f"{self.name}: landmarks must be exactly {sorted(expected)}")
        indices = list(self.landmarks.values())
        if len(set(indices)) != len(indices):
            raise ValueError(f"{self.name}: duplicate landmark indices")
        if min(indices) < 0 or max(indices) >= self.n_landmarks:
            raise ValueError(f"{self.name}: landmark index out of range")

    def index(self, side: Side, joint: str) -> int:
        try:
            return self.landmarks[f"{side}_{joint}"]
        except KeyError:
            raise KeyError(f"{self.name} has no landmark '{side}_{joint}'") from None


# MediaPipe BlazePose, 33 landmarks.
BLAZEPOSE = Skeleton(
    name="blazepose",
    n_landmarks=33,
    landmarks={
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    },
)

# Halpe-26 (COCO-17 body + head/neck/hip + 6 foot points), as produced by RTMPose body-with-feet
# models. ``foot_index`` maps to the big toe.
# TODO(human): confirm these indices against the rtmlib / MMPose Halpe-26 definition before
# using RTMPose output; they were written from memory of the layout, not checked against a run.
HALPE26 = Skeleton(
    name="halpe26",
    n_landmarks=26,
    landmarks={
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot_index": 20,
        "right_foot_index": 21,
        "left_heel": 24,
        "right_heel": 25,
    },
)

SKELETONS: dict[str, Skeleton] = {s.name: s for s in (BLAZEPOSE, HALPE26)}


def get_skeleton(name: str) -> Skeleton:
    try:
        return SKELETONS[name]
    except KeyError:
        raise KeyError(f"Unknown skeleton '{name}'. Known: {sorted(SKELETONS)}") from None
