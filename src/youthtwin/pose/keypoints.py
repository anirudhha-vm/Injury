"""Keypoint sequences: the only thing the pipeline keeps from a video (privacy rule 6).

Coordinates are **pixels in image convention**: origin top-left, x to the right, **y downward**.
``timestamps_ms`` are the per-frame presentation times from the video container, so velocities
use true time even when the frame rate is uneven. ``conf`` is the estimator's per-landmark
confidence in [0, 1]. A sequence records which camera it came from and whether it is synthetic;
it never records the path of the source video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.pose.skeletons import Side, Skeleton, get_skeleton

FloatArray = NDArray[np.float64]
Camera = Literal["frontal", "sagittal"]


@dataclass(frozen=True)
class KeypointSequence:
    """T frames x J landmarks."""

    xy: FloatArray  # (T, J, 2) pixels, y down
    conf: FloatArray  # (T, J)
    timestamps_ms: FloatArray  # (T,)
    skeleton: str
    camera: Camera
    image_size: tuple[int, int]  # (width, height) in pixels
    estimator: str
    synthetic: bool
    z: FloatArray | None = None  # (T, J) estimator depth, if any (not used for angles)

    def __post_init__(self) -> None:
        n_frames, n_landmarks = self.conf.shape
        layout = get_skeleton(self.skeleton)
        if self.xy.shape != (n_frames, n_landmarks, 2):
            raise ValueError("xy must have shape (T, J, 2) matching conf (T, J)")
        if n_landmarks != layout.n_landmarks:
            raise ValueError(f"{self.skeleton} expects {layout.n_landmarks} landmarks")
        if self.timestamps_ms.shape != (n_frames,):
            raise ValueError("timestamps_ms must have one entry per frame")
        if n_frames > 1 and np.any(np.diff(self.timestamps_ms) <= 0):
            raise ValueError("timestamps_ms must be strictly increasing")
        if self.z is not None and self.z.shape != (n_frames, n_landmarks):
            raise ValueError("z must have shape (T, J)")

    @property
    def n_frames(self) -> int:
        return int(self.conf.shape[0])

    @property
    def layout(self) -> Skeleton:
        return get_skeleton(self.skeleton)

    @property
    def time_s(self) -> FloatArray:
        """Seconds from the first frame."""
        return np.asarray((self.timestamps_ms - self.timestamps_ms[0]) / 1000.0, dtype=np.float64)

    @property
    def fps(self) -> float:
        """Median frame rate from the timestamps."""
        if self.n_frames < 2:
            raise ValueError("fps needs at least two frames")
        return float(1000.0 / np.median(np.diff(self.timestamps_ms)))

    def joint(self, side: Side, name: str) -> FloatArray:
        """Trajectory of one joint: (T, 2) pixels, y down."""
        return self.xy[:, self.layout.index(side, name), :]

    def joint_conf(self, side: Side, name: str) -> FloatArray:
        return self.conf[:, self.layout.index(side, name)]

    def with_xy(self, xy: FloatArray) -> KeypointSequence:
        """Same sequence with replaced coordinates (e.g. after filtering)."""
        return KeypointSequence(
            xy=xy,
            conf=self.conf,
            timestamps_ms=self.timestamps_ms,
            skeleton=self.skeleton,
            camera=self.camera,
            image_size=self.image_size,
            estimator=self.estimator,
            synthetic=self.synthetic,
            z=self.z,
        )

    # --- persistence: parquet (long format) + JSON sidecar -----------------------------

    def to_frame(self) -> pd.DataFrame:
        n_frames, n_landmarks = self.conf.shape
        frame = pd.DataFrame(
            {
                "frame": np.repeat(np.arange(n_frames), n_landmarks),
                "timestamp_ms": np.repeat(self.timestamps_ms, n_landmarks),
                "landmark": np.tile(np.arange(n_landmarks), n_frames),
                "x": self.xy[:, :, 0].ravel(),
                "y": self.xy[:, :, 1].ravel(),
                "conf": self.conf.ravel(),
            }
        )
        if self.z is not None:
            frame["z"] = self.z.ravel()
        frame["synthetic"] = self.synthetic
        return frame

    def save(self, path: Path) -> Path:
        """Write ``<path>.parquet`` and ``<path>.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(path.with_suffix(".parquet"), index=False)
        info = {
            "skeleton": self.skeleton,
            "camera": self.camera,
            "image_size": list(self.image_size),
            "estimator": self.estimator,
            "synthetic": self.synthetic,
            "n_frames": self.n_frames,
        }
        path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return path.with_suffix(".parquet")

    @classmethod
    def load(cls, path: Path) -> KeypointSequence:
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        frame = pd.read_parquet(path.with_suffix(".parquet")).sort_values(["frame", "landmark"])
        n_frames = int(info["n_frames"])
        n_landmarks = len(frame) // n_frames

        def grid(column: str) -> FloatArray:
            return frame[column].to_numpy(dtype=np.float64).reshape(n_frames, n_landmarks)

        return cls(
            xy=np.stack([grid("x"), grid("y")], axis=-1),
            conf=grid("conf"),
            timestamps_ms=grid("timestamp_ms")[:, 0],
            skeleton=info["skeleton"],
            camera=info["camera"],
            image_size=(int(info["image_size"][0]), int(info["image_size"][1])),
            estimator=info["estimator"],
            synthetic=bool(info["synthetic"]),
            z=grid("z") if "z" in frame.columns else None,
        )
