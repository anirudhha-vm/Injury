"""Gap filling, low-pass filtering and differentiation of trajectories.

v2 defect: velocities were unfiltered and in degrees per frame. Here:

- low-confidence samples are removed; gaps up to ``max_gap`` frames are filled linearly, longer
  gaps stay NaN and are never bridged by the filter;
- the low-pass is a **zero-lag 4th-order Butterworth**: a 2nd-order filter run forward and
  backward. Two passes lower the -3 dB point, so the design cut-off is raised by
  ``1 / (2^(1/2) - 1)^(1/4)`` (about 1.247; Winter, *Biomechanics and Motor Control of Human
  Movement*) to put -3 dB at the requested cut-off;
- derivatives use the true timestamps and are per second.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_PASSES = 2
_DUAL_PASS_CORRECTION = (2.0 ** (1.0 / _PASSES) - 1.0) ** 0.25
_SINGLE_PASS_ORDER = 2
_UNIFORM_TOLERANCE = 0.02  # timestamps within 2% of the median interval count as uniform


def fill_gaps(
    values: FloatArray, conf: FloatArray, *, min_confidence: float, max_gap: int
) -> tuple[FloatArray, BoolArray]:
    """Drop low-confidence samples and linearly fill interior gaps of at most ``max_gap`` frames.

    ``values`` is (T,) or (T, D); ``conf`` is (T,). Returns ``(filled, was_missing)`` where
    ``was_missing`` marks frames that were low-confidence or NaN in the input. Gaps longer than
    ``max_gap``, and gaps touching either end, remain NaN.
    """
    data = np.array(values, dtype=np.float64, copy=True)
    flat = data.reshape(data.shape[0], -1)
    missing = (conf < min_confidence) | ~np.isfinite(flat).all(axis=1)
    flat[missing] = np.nan
    good = np.flatnonzero(~missing)
    for left, right in pairwise(good):
        gap = right - left - 1
        if 0 < gap <= max_gap:
            weight = (np.arange(left + 1, right) - left)[:, None] / (right - left)
            flat[left + 1 : right] = flat[left] + weight * (flat[right] - flat[left])
    return flat.reshape(data.shape), missing


def is_uniform(timestamps_s: FloatArray) -> bool:
    """Are the samples evenly spaced (within tolerance)?"""
    steps = np.diff(timestamps_s)
    return bool(np.all(np.abs(steps - np.median(steps)) <= _UNIFORM_TOLERANCE * np.median(steps)))


def resample_uniform(values: FloatArray, timestamps_s: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Linear resampling onto an even grid at the median frame interval. NaNs stay NaN."""
    step = float(np.median(np.diff(timestamps_s)))
    grid = np.arange(timestamps_s[0], timestamps_s[-1] + 0.5 * step, step)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((grid.size, flat.shape[1]))
    nearest = np.clip(np.searchsorted(timestamps_s, grid), 1, timestamps_s.size - 1)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(grid, timestamps_s, flat[:, column])
        bracket_nan = np.isnan(flat[nearest - 1, column]) | np.isnan(flat[nearest, column])
        out[bracket_nan, column] = np.nan
    return out.reshape((grid.size, *values.shape[1:])), grid


def butterworth_lowpass(values: FloatArray, *, fs: float, cutoff_hz: float) -> FloatArray:
    """Zero-lag 4th-order Butterworth low-pass along axis 0, with -3 dB at ``cutoff_hz``.

    Requires evenly sampled data at ``fs`` Hz. Each run of consecutive finite samples is
    filtered separately; runs too short to filter and NaN gaps are returned as NaN / unchanged
    (short runs are left unfiltered rather than invented).
    """
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({fs / 2:.1f} Hz)")
    design_cutoff = min(cutoff_hz / _DUAL_PASS_CORRECTION, 0.99 * fs / 2)
    b, a = signal.butter(_SINGLE_PASS_ORDER, design_cutoff, btype="low", fs=fs)
    pad = 3 * max(len(a), len(b))
    data = np.asarray(values, dtype=np.float64)
    flat = data.reshape(data.shape[0], -1)
    out = flat.copy()
    finite = np.isfinite(flat).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], finite, [False]]).astype(int)))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start > pad:
            out[start:stop] = signal.filtfilt(b, a, flat[start:stop], axis=0)
    out[~finite] = np.nan
    return out.reshape(data.shape)


def derivative(values: FloatArray, timestamps_s: FloatArray) -> FloatArray:
    """Time derivative along axis 0 using the true timestamps (units of ``values`` per second)."""
    if values.shape[0] != timestamps_s.shape[0]:
        raise ValueError("values and timestamps_s must have the same length")
    return np.asarray(np.gradient(values, timestamps_s, axis=0), dtype=np.float64)


def residual_analysis(
    values: FloatArray, *, fs: float, cutoffs_hz: Sequence[float]
) -> dict[float, float]:
    """RMS difference between the raw and filtered signal for each candidate cut-off.

    Plot residual against cut-off and choose where the curve leaves its linear high-frequency
    tail (Winter's residual analysis) to justify the cut-off in ``configs/pose/filter.yaml``.
    """
    finite = np.isfinite(values.reshape(values.shape[0], -1)).all(axis=1)
    residuals: dict[float, float] = {}
    for cutoff in cutoffs_hz:
        smooth = butterworth_lowpass(values, fs=fs, cutoff_hz=cutoff)
        residuals[float(cutoff)] = float(
            np.sqrt(np.nanmean((values[finite] - smooth[finite]) ** 2))
        )
    return residuals
