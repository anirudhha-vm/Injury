"""Synthetic markerless-vs-mocap validation study (Paper A layer; D-121).

The same true drop-jump movement (``skeleton_sim``) is observed twice: by a marker-based
reference (small marker-placement offsets and noise) and by a smartphone pose estimator (larger
per-participant and per-trial landmark offsets plus frame noise). Both go through the real
pose pipeline (``pose.task_features``), so the agreement statistics reflect the pipeline's
filtering, event detection and angle definitions, not just added noise.

Errors are set in pixels at adult size and scaled by ``1 / body_scale``: at a fixed camera
distance a smaller child spans fewer pixels. Whether that makes agreement worse before PHV is
therefore an output of the simulation, not an input. The markerless offset size is calibrated
to a published knee-flexion error (config). Everything here is SYNTHETIC.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.pose import task_features as tf
from youthtwin.pose.keypoints import KeypointSequence
from youthtwin.sim import skeleton_sim

TASK = "drop_vertical_jump"
SIDES = ("left", "right")


def _observe(
    sequence: KeypointSequence,
    offset: np.ndarray,
    frame_noise_px: float,
    scale: float,
    rng: np.random.Generator,
) -> KeypointSequence:
    """``sequence`` with constant landmark offsets and frame noise, both in adult pixels."""
    noise = rng.normal(0.0, frame_noise_px, sequence.xy.shape)
    xy = sequence.xy + (offset[None, :, :] + noise) / scale
    return dataclasses.replace(sequence, xy=xy, estimator="synthetic-observed")


def _event_label(feature: str) -> tuple[str, str]:
    if "_at_" in feature:
        name, event = feature.split("_at_", 1)
        return name, event
    name, _, event = feature.rpartition("_")
    return name, event


def simulate_study(
    config: Mapping[str, Any],
    configs: Mapping[str, Any],
    *,
    seed: int,
    offset_sd_px: float,
    groups: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Paired values, one row per participant x side x variable (means over trials).

    ``configs`` holds ``filter, events, features`` (the pose pipeline's YAML blocks).
    ``groups`` overrides ``participants_per_group``.
    """
    rng = np.random.default_rng(seed)
    truth, mocap, markerless = config["truth"], config["mocap"], config["markerless"]
    counts = groups or config["participants_per_group"]
    rows: list[dict[str, Any]] = []
    participant = 0
    for group, n in counts.items():
        scale = float(config["body_scale"][group])
        for _ in range(int(n)):
            participant += 1
            athlete = f"val-{participant:03d}"
            sex = "female" if rng.random() < float(config["proportion_female"]) else "male"
            flexion = rng.normal(
                truth["peak_knee_flexion_deg"]["mean"], truth["peak_knee_flexion_deg"]["sd"]
            )
            fppa = rng.normal(truth["peak_fppa_deg"]["mean"], truth["peak_fppa_deg"]["sd"], 2)
            # Standard normal draws, scaled below: the same draws for every offset size, so
            # the calibration target is a smooth function of offset_sd_px.
            participant_unit = {v: rng.normal(0.0, 1.0, (33, 2)) for v in ("sagittal", "frontal")}
            reference_offset = {
                v: rng.normal(0.0, mocap["participant_offset_sd_px"], (33, 2))
                for v in ("sagittal", "frontal")
            }
            values: dict[tuple[str, str], list[tuple[float, float]]] = {}
            for trial in range(int(config["trials_per_participant"])):
                clean = skeleton_sim.simulate_drop_vertical_jump(
                    rng,
                    fps=float(config["fps"]),
                    peak_knee_flexion=float(
                        np.clip(flexion + rng.normal(0, truth["trial_sd_deg"]), 40, 120)
                    ),
                    peak_fppa=tuple(float(f + rng.normal(0, truth["trial_sd_deg"])) for f in fppa),  # type: ignore[arg-type]
                    noise_px=0.0,
                )
                observed = {}
                for kind in ("mocap", "markerless"):
                    views = {}
                    for view in ("sagittal", "frontal"):
                        sequence = getattr(clean, view)
                        if kind == "mocap":
                            offset = reference_offset[view]
                            noise = float(mocap["frame_noise_sd_px"])
                        else:
                            trial_unit = rng.normal(0.0, 1.0, (33, 2))
                            offset = offset_sd_px * (
                                participant_unit[view]
                                + float(markerless["trial_share"]) * trial_unit
                            )
                            noise = float(markerless["frame_noise_sd_px"])
                        views[view] = _observe(sequence, offset, noise, scale, rng)
                    result = tf.trial_features(
                        tf.TrialInput(
                            athlete, "s1", TASK, trial, "both", views["sagittal"], views["frontal"]
                        ),
                        filter_config=configs["filter"],
                        event_config=configs["events"],
                        feature_config=configs["features"],
                    )
                    observed[kind] = result.features if result.included else {}
                for side in SIDES:
                    for variable in config["variables"]:
                        key = (side, variable)
                        if key in observed["mocap"] and key in observed["markerless"]:
                            values.setdefault((side, variable), []).append(
                                (observed["markerless"][key], observed["mocap"][key])
                            )
            for (side, variable), pairs in values.items():
                name, event = _event_label(variable)
                method, reference = np.mean(np.asarray(pairs), axis=0)
                rows.append(
                    {
                        "athlete_id": athlete,
                        "side": side,
                        "variable": name,
                        "task": TASK,
                        "event": event,
                        "markerless": float(method),
                        "mocap": float(reference),
                        "maturity_group": group,
                        "sex": sex,
                        "n_trials": len(pairs),
                        "synthetic": True,
                    }
                )
    return pd.DataFrame(rows)


def mean_absolute_error(pairs: pd.DataFrame, feature: str) -> float:
    name, event = _event_label(feature)
    part = pairs[(pairs["variable"] == name) & (pairs["event"] == event)]
    return float((part["markerless"] - part["mocap"]).abs().mean())
