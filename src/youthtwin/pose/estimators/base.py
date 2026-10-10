"""Common interface of pose estimators, and video decoding shared by them.

An estimator turns one local video file into a ``KeypointSequence``. Nothing from the video is
kept: no frames, no path (privacy rule 6). Heavy dependencies (OpenCV, MediaPipe, rtmlib) are an
optional extra (``uv sync --extra pose``) and are imported only when an estimator is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose.keypoints import Camera, KeypointSequence

INSTALL_HINT = "Install the optional pose dependencies: `uv sync --extra pose`."


class PoseEstimator(Protocol):
    """Anything that can turn a video file into keypoints."""

    name: str

    def process(self, video_path: Path, camera: Camera) -> KeypointSequence: ...


def require(module: str) -> Any:
    """Import an optional dependency or fail with an actionable message."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(f"'{module}' is required for pose estimation. {INSTALL_HINT}") from error


def read_frames(video_path: Path) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """Yield ``(timestamp_ms, RGB frame)`` using the container's presentation timestamps."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cv2 = require("cv2")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path.name}")
    try:
        while True:
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            ok, frame = capture.read()
            if not ok:
                break
            yield timestamp_ms, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def monotonic_timestamps(timestamps_ms: list[float]) -> NDArray[np.float64]:
    """Container timestamps, nudged where needed so they are strictly increasing.

    Some containers repeat a timestamp for consecutive frames; a 0.001 ms step keeps order
    without changing timing meaningfully.
    """
    out = np.asarray(timestamps_ms, dtype=np.float64)
    for k in range(1, out.size):
        if out[k] <= out[k - 1]:
            out[k] = out[k - 1] + 1e-3
    return out
