"""Run pose estimation on local videos and keep only keypoints.

Reads a trial manifest (CSV) describing which local video belongs to which trial and camera,
writes one keypoint file per video under ``data/keypoints/`` and, with ``--features``, the tidy
task-feature table and the exclusion log. Videos are read from disk and never copied, uploaded
or referenced in any output (privacy rule 6).

Manifest columns: ``athlete_id, session_id, task, trial, limb, camera, video``, where ``video``
is a path relative to the video directory and ``camera`` is ``frontal`` or ``sagittal``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from youthtwin.pose import task_features as tf
from youthtwin.pose.estimators.base import PoseEstimator
from youthtwin.pose.keypoints import KeypointSequence
from youthtwin.utils.config import load_config, repo_root

MANIFEST_COLUMNS = ["athlete_id", "session_id", "task", "trial", "limb", "camera", "video"]


def _estimator(name: str) -> PoseEstimator:
    config = load_config("pose/estimator.yaml")
    if name == "mediapipe":
        from youthtwin.pose.estimators.mediapipe import MediaPipePoseEstimator

        return MediaPipePoseEstimator(config)
    if name == "rtmpose":
        from youthtwin.pose.estimators.rtmpose import RTMPoseEstimator

        return RTMPoseEstimator(config)
    raise typer.BadParameter(f"Unknown estimator '{name}'")


def main(
    manifest: Path,
    video_dir: Path | None = None,
    out_dir: Path | None = None,
    estimator: str | None = None,
    features: bool = True,
) -> None:
    root = repo_root()
    videos = video_dir or root / "data" / "raw_video_LOCAL_ONLY"
    target = out_dir or root / "data" / "keypoints"
    name = estimator or str(load_config("pose/estimator.yaml")["default"])
    model = _estimator(name)
    table = pd.read_csv(manifest)
    absent = [c for c in MANIFEST_COLUMNS if c not in table.columns]
    if absent:
        raise typer.BadParameter(f"Manifest is missing columns: {absent}")

    views: dict[tuple[str, str, str, int, str], dict[str, KeypointSequence]] = {}
    for row in table.itertuples(index=False):
        sequence = model.process(videos / str(row.video), str(row.camera))  # type: ignore[arg-type]
        stem = f"{row.session_id}_{row.task}_{row.limb}_{row.trial}_{row.camera}_{name}"
        sequence.save(target / str(row.athlete_id) / stem)
        key = (
            str(row.athlete_id),
            str(row.session_id),
            str(row.task),
            int(row.trial),
            str(row.limb),
        )
        views.setdefault(key, {})[str(row.camera)] = sequence
        typer.echo(f"keypoints: {stem} ({sequence.n_frames} frames, {sequence.fps:.0f} fps)")

    if not features:
        return
    configs = {
        "filter_config": load_config("pose/filter.yaml"),
        "event_config": load_config("pose/events.yaml"),
        "feature_config": load_config("features/task_features.yaml"),
    }
    results = [
        tf.trial_features(
            tf.TrialInput(
                athlete, session, task, trial, limb, cameras.get("sagittal"), cameras.get("frontal")
            ),
            **configs,
        )
        for (athlete, session, task, trial, limb), cameras in views.items()
    ]
    feature_table, exclusions = tf.session_features(results, configs["feature_config"])
    feature_table.to_parquet(target / "task_features.parquet", index=False)
    exclusions.to_csv(target / "excluded_trials.csv", index=False)
    typer.echo(f"{len(feature_table)} feature rows; {len(exclusions)} trial(s) excluded (see log)")


if __name__ == "__main__":
    typer.run(main)
