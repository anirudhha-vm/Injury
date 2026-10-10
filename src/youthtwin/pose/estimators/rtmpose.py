"""RTMPose (body with feet, Halpe-26) through ``rtmlib``; the comparator estimator for Paper A.

UNVERIFIED: written from the rtmlib documentation and not yet run. Before use, confirm (a) the
``BodyWithFeet`` call signature of the installed rtmlib version and (b) the Halpe-26 indices in
``youthtwin.pose.skeletons`` against a real output. TODO(human): provide test clips.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from youthtwin.pose.estimators.base import monotonic_timestamps, read_frames, require
from youthtwin.pose.keypoints import Camera, KeypointSequence
from youthtwin.pose.skeletons import HALPE26


class RTMPoseEstimator:
    """Halpe-26 keypoints per frame, in pixels, with the model score as confidence."""

    name = "rtmpose_body_with_feet"

    def __init__(self, config: Mapping[str, Any]) -> None:
        settings = config["rtmpose"]
        self._mode = str(settings["mode"])
        self._backend = str(settings["inference_backend"])
        self._device = str(settings["device"])

    def process(self, video_path: Path, camera: Camera) -> KeypointSequence:
        rtmlib = require("rtmlib")
        cv2 = require("cv2")
        model = rtmlib.BodyWithFeet(mode=self._mode, backend=self._backend, device=self._device)
        xy: list[np.ndarray[Any, Any]] = []
        conf: list[np.ndarray[Any, Any]] = []
        stamps: list[float] = []
        size = (0, 0)
        for timestamp_ms, frame in read_frames(video_path):
            height, width = frame.shape[:2]
            size = (width, height)
            keypoints, scores = model(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            stamps.append(timestamp_ms)
            if len(keypoints):
                best = int(np.argmax(np.mean(scores, axis=1)))  # the most confident person
                xy.append(np.asarray(keypoints[best], dtype=np.float64))
                conf.append(np.asarray(scores[best], dtype=np.float64))
            else:
                xy.append(np.full((HALPE26.n_landmarks, 2), np.nan))
                conf.append(np.zeros(HALPE26.n_landmarks))
        if not stamps:
            raise ValueError(f"No frames could be read from {video_path.name}")
        return KeypointSequence(
            xy=np.stack(xy),
            conf=np.clip(np.stack(conf), 0.0, 1.0),
            timestamps_ms=monotonic_timestamps(stamps),
            skeleton=HALPE26.name,
            camera=camera,
            image_size=size,
            estimator=self.name,
            synthetic=False,
        )
