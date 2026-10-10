"""MediaPipe pose estimation through the Tasks ``PoseLandmarker`` API.

The legacy ``mp.solutions.pose`` interface is deprecated and is not used. The model file
(``pose_landmarker_heavy.task`` for research) must be downloaded once by a human and its path
set in ``configs/pose/estimator.yaml``.

UNVERIFIED AGAINST REAL VIDEO: written to the documented Tasks API but not yet run, because no
test clip or model file is available. TODO(human): provide 2-3 consented or self-recorded adult
test clips and the model file, then run ``scripts/run_pose.py`` on them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from youthtwin.pose.estimators.base import monotonic_timestamps, read_frames, require
from youthtwin.pose.keypoints import Camera, KeypointSequence
from youthtwin.pose.skeletons import BLAZEPOSE


class MediaPipePoseEstimator:
    """BlazePose landmarks (33) per frame, in pixels, with visibility as confidence."""

    name = "mediapipe_pose_landmarker"

    def __init__(self, config: Mapping[str, Any]) -> None:
        settings = config["mediapipe"]
        self._model_path = Path(str(settings["model_path"])).expanduser()
        self._min_detection = float(settings["min_pose_detection_confidence"])
        self._min_tracking = float(settings["min_tracking_confidence"])

    def process(self, video_path: Path, camera: Camera) -> KeypointSequence:
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at {self._model_path}. TODO(human): download the "
                "pose landmarker model and set mediapipe.model_path in configs/pose/estimator.yaml."
            )
        mp = require("mediapipe")
        vision = require("mediapipe.tasks.python.vision")
        tasks = require("mediapipe.tasks.python")
        options = vision.PoseLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self._min_detection,
            min_tracking_confidence=self._min_tracking,
        )
        xy: list[np.ndarray[Any, Any]] = []
        conf: list[np.ndarray[Any, Any]] = []
        depth: list[np.ndarray[Any, Any]] = []
        stamps: list[float] = []
        size = (0, 0)
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for timestamp_ms, frame in read_frames(video_path):
                height, width = frame.shape[:2]
                size = (width, height)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                # The Tasks API needs integer, strictly increasing timestamps in ms.
                stamp = max(round(timestamp_ms), (round(stamps[-1]) + 1) if stamps else 0)
                result = landmarker.detect_for_video(image, stamp)
                stamps.append(float(stamp))
                if result.pose_landmarks:
                    points = result.pose_landmarks[0]
                    xy.append(np.array([[p.x * width, p.y * height] for p in points]))
                    conf.append(np.array([p.visibility for p in points]))
                    depth.append(np.array([p.z for p in points]))
                else:  # nobody detected: keep the frame, mark every landmark missing
                    xy.append(np.full((BLAZEPOSE.n_landmarks, 2), np.nan))
                    conf.append(np.zeros(BLAZEPOSE.n_landmarks))
                    depth.append(np.full(BLAZEPOSE.n_landmarks, np.nan))
        if not stamps:
            raise ValueError(f"No frames could be read from {video_path.name}")
        return KeypointSequence(
            xy=np.stack(xy).astype(np.float64),
            conf=np.stack(conf).astype(np.float64),
            timestamps_ms=monotonic_timestamps(stamps),
            skeleton=BLAZEPOSE.name,
            camera=camera,
            image_size=size,
            estimator=self.name,
            synthetic=False,
            z=np.stack(depth).astype(np.float64),
        )
