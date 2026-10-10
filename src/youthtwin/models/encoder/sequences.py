"""Encoder inputs: fixed-length multi-channel sequences from screening trials.

A trial becomes an array of shape (channels, time): the time-normalised joint-angle curves over
the task window produced by the pose pipeline, plus their angular velocities in degrees per
second. Channels are listed in a fixed order so that the same index always means the same
quantity. A session is a stack of its valid trials.

Normalisation statistics (per-channel mean and SD) are computed from **training athletes only**
and stored in a ``ChannelScaler`` that travels with the model (non-negotiable N4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose.task_features import TrialResult

FloatArray = NDArray[np.float64]

SAGITTAL = ("knee_flexion", "hip_flexion", "ankle_dorsiflexion")
FRONTAL = ("knee_fppa",)
_EPS = 1e-8


def channel_names(sides: Sequence[str], *, frontal: bool) -> list[str]:
    """Channel order: per side, each angle followed by its velocity."""
    angles = (*SAGITTAL, *(FRONTAL if frontal else ()))
    return [
        f"{side}_{angle}{suffix}"
        for side in sides
        for angle in angles
        for suffix in ("", "_velocity")
    ]


def trial_channels(result: TrialResult, sides: Sequence[str], *, frontal: bool) -> FloatArray:
    """(channels, time) array for one included trial.

    Velocity is the derivative of the time-normalised curve converted to degrees per second
    with the real duration of the task window, so it does not depend on frame rate or on how
    many points the curve was resampled to.
    """
    if not result.included:
        raise ValueError("Excluded trials have no curves")
    duration = next(
        value for (_, name), value in result.features.items() if name == "window_duration_s"
    )
    angles = (*SAGITTAL, *(FRONTAL if frontal else ()))
    rows: list[FloatArray] = []
    for side in sides:
        for angle in angles:
            curve = result.curves[(side, angle)]
            seconds = np.linspace(0.0, duration, curve.size)
            rows.append(curve)
            rows.append(np.gradient(curve, seconds))
    return np.stack(rows).astype(np.float64)


@dataclass(frozen=True)
class SessionSequences:
    """All sessions' trials, padded to the same number of trials.

    ``trials`` is (sessions, max_trials, channels, time); ``mask`` is (sessions, max_trials) and
    is 1 where a real trial is present.
    """

    session_ids: tuple[str, ...]
    athlete_ids: tuple[str, ...]
    trials: FloatArray
    mask: FloatArray
    channels: tuple[str, ...]

    def index_of(self) -> dict[str, int]:
        return {session: k for k, session in enumerate(self.session_ids)}


def stack_sessions(
    per_session: Mapping[str, Sequence[FloatArray]],
    athlete_of: Mapping[str, str],
    channels: Sequence[str],
) -> SessionSequences:
    """Pad every session to the largest trial count. Sessions without valid trials are dropped."""
    kept = {session: list(trials) for session, trials in per_session.items() if len(trials) > 0}
    if not kept:
        raise ValueError("No session has a valid trial")
    max_trials = max(len(trials) for trials in kept.values())
    n_channels, n_time = next(iter(kept.values()))[0].shape
    data = np.zeros((len(kept), max_trials, n_channels, n_time))
    mask = np.zeros((len(kept), max_trials))
    for k, trials in enumerate(kept.values()):
        data[k, : len(trials)] = np.stack(trials)
        mask[k, : len(trials)] = 1.0
    session_ids = tuple(kept)
    return SessionSequences(
        session_ids=session_ids,
        athlete_ids=tuple(athlete_of[s] for s in session_ids),
        trials=data,
        mask=mask,
        channels=tuple(channels),
    )


@dataclass(frozen=True)
class ChannelScaler:
    """Per-channel mean and SD, fitted on the training athletes' trials only."""

    mean: FloatArray  # (channels,)
    sd: FloatArray  # (channels,)

    @classmethod
    def fit(cls, sequences: SessionSequences, train_athletes: set[str]) -> ChannelScaler:
        rows = np.array([a in train_athletes for a in sequences.athlete_ids])
        if not rows.any():
            raise ValueError("No training sessions to fit the channel scaler on")
        real = sequences.mask[rows].astype(bool)  # (sessions, trials)
        values = sequences.trials[rows][real]  # (n_real_trials, channels, time)
        return cls(mean=values.mean(axis=(0, 2)), sd=values.std(axis=(0, 2)) + _EPS)

    def transform(self, trials: FloatArray) -> FloatArray:
        """Standardise (..., channels, time)."""
        return np.asarray((trials - self.mean[:, None]) / self.sd[:, None], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "sd": self.sd.tolist()}
