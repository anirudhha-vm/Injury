"""Time alignment of the two phones from a shared event (a clap or an LED flash).

Both cameras record the same sharp event at the start of a trial. Each recording is reduced to
a one-dimensional signal sampled at a common rate (audio envelope for a clap, mean frame
brightness for a flash); the offset is the lag that maximises their normalised
cross-correlation, refined to sub-sample precision.

Extracting the audio track or brightness trace from a video file is done by the caller (it
needs the video, which stays local); this module only aligns signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyncResult:
    """``offset_s`` is added to camera B's clock to express its times on camera A's clock."""

    offset_s: float
    peak_correlation: float  # 0-1; low values mean the shared event was not found


def estimate_offset(signal_a: FloatArray, signal_b: FloatArray, fs: float) -> SyncResult:
    """Offset between two recordings of the same event, both sampled at ``fs`` Hz.

    If the event occurs at time ``t_a`` in A and ``t_b`` in B (each from its own start), the
    result is ``t_a - t_b``.
    """
    a = np.asarray(signal_a, dtype=np.float64) - float(np.mean(signal_a))
    b = np.asarray(signal_b, dtype=np.float64) - float(np.mean(signal_b))
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale == 0.0:
        raise ValueError("Cannot synchronise a constant signal")
    correlation = signal.correlate(a, b, mode="full") / scale
    lags = signal.correlation_lags(a.size, b.size, mode="full")
    k = int(np.argmax(correlation))
    shift = 0.0
    if 0 < k < correlation.size - 1:
        left, mid, right = correlation[k - 1], correlation[k], correlation[k + 1]
        curvature = left - 2.0 * mid + right
        if curvature < 0:
            shift = 0.5 * (left - right) / curvature
    return SyncResult(
        offset_s=float((lags[k] + shift) / fs), peak_correlation=float(correlation[k])
    )


def align_timestamps(timestamps_b_ms: FloatArray, result: SyncResult) -> FloatArray:
    """Camera B timestamps expressed on camera A's clock."""
    return np.asarray(timestamps_b_ms + 1000.0 * result.offset_s, dtype=np.float64)
