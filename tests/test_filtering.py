"""Filtering (v2 regression: unfiltered velocities in degrees per frame)."""

from __future__ import annotations

import numpy as np
import pytest

from youthtwin.pose import filtering, sync

FS = 120.0


def _gain(frequency: float, cutoff: float) -> float:
    t = np.arange(0, 6.0, 1.0 / FS)
    wave = np.sin(2 * np.pi * frequency * t)
    out = filtering.butterworth_lowpass(wave, fs=FS, cutoff_hz=cutoff)
    middle = slice(t.size // 4, 3 * t.size // 4)
    return float(np.std(out[middle]) / np.std(wave[middle]))


# ----------------------------------------------------------------------------- gap filling


def test_short_gaps_are_filled_linearly_and_long_gaps_stay_missing() -> None:
    values = np.arange(20, dtype=np.float64) * 2.0
    conf = np.ones(20)
    conf[3:6] = 0.1  # 3-frame gap: filled
    conf[10:17] = 0.1  # 7-frame gap: too long
    filled, missing = filtering.fill_gaps(values, conf, min_confidence=0.5, max_gap=5)
    np.testing.assert_allclose(filled[3:6], [6.0, 8.0, 10.0])
    assert np.isnan(filled[10:17]).all()
    assert np.isfinite(filled[[9, 17]]).all()
    assert missing.sum() == 10
    assert values[4] == 8.0  # input untouched


def test_gap_of_exactly_the_limit_is_filled_and_edges_are_not_extrapolated() -> None:
    values = np.linspace(0.0, 1.0, 12)
    conf = np.ones(12)
    conf[:2] = 0.0  # leading gap: nothing to interpolate from
    conf[4:9] = 0.0  # exactly 5 frames
    conf[-1] = 0.0  # trailing gap
    filled, _ = filtering.fill_gaps(values, conf, min_confidence=0.5, max_gap=5)
    assert np.isnan(filled[:2]).all() and np.isnan(filled[-1])
    np.testing.assert_allclose(filled[4:9], values[4:9])


def test_gap_filling_handles_2d_points_and_nan_input() -> None:
    xy = np.stack([np.arange(8.0), np.arange(8.0) * 3], axis=1)
    xy[2] = np.nan  # missing even though confidence is high
    filled, missing = filtering.fill_gaps(xy, np.ones(8), min_confidence=0.5, max_gap=5)
    np.testing.assert_allclose(filled[2], [2.0, 6.0])
    assert missing.tolist() == [False, False, True, False, False, False, False, False]


# ----------------------------------------------------------------------------- low-pass


@pytest.mark.parametrize("cutoff", [6.0, 8.0, 12.0])
def test_minus_3_db_at_the_requested_cutoff(cutoff: float) -> None:
    """Zero-lag (dual-pass) filtering must still put -3 dB at the configured cut-off."""
    assert _gain(cutoff, cutoff) == pytest.approx(1 / np.sqrt(2), abs=0.02)


def test_passband_is_kept_and_stopband_removed() -> None:
    assert _gain(1.0, 8.0) == pytest.approx(1.0, abs=0.01)
    assert _gain(40.0, 8.0) < 0.01
    # Two passes of a 2nd-order digital Butterworth with design cut-off f0 = fc / 0.802 have
    # gain 1 / (1 + (tan(pi f / fs) / tan(pi f0 / fs))^4). One octave above fc at 120 Hz:
    # tan(0.4189) / tan(0.2611) = 0.4452 / 0.2672 = 1.666 -> 1 / (1 + 7.71) = 0.115.
    assert _gain(16.0, 8.0) == pytest.approx(0.115, abs=0.01)


def test_filter_has_zero_phase_lag() -> None:
    t = np.arange(0, 4.0, 1.0 / FS)
    wave = np.sin(2 * np.pi * 2.0 * t)
    out = filtering.butterworth_lowpass(wave, fs=FS, cutoff_hz=8.0)
    lag = sync.estimate_offset(out[100:-100], wave[100:-100], FS).offset_s
    assert abs(lag) < 0.2 / FS  # well under one frame


def test_noise_is_reduced_and_gaps_are_not_bridged() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 3.0, 1.0 / FS)
    clean = 40 * np.sin(2 * np.pi * 1.5 * t)
    noisy = clean + rng.normal(0, 2.0, t.size)
    noisy[150:170] = np.nan
    out = filtering.butterworth_lowpass(noisy, fs=FS, cutoff_hz=8.0)
    assert np.isnan(out[150:170]).all()
    finite = np.isfinite(out)
    finite[:20] = finite[-20:] = False
    finite[130:190] = False  # ignore filter edge effects next to the gap
    assert np.std(out[finite] - clean[finite]) < 0.5 * 2.0


def test_short_runs_are_left_unfiltered_and_bad_cutoff_is_rejected() -> None:
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    out = filtering.butterworth_lowpass(values, fs=FS, cutoff_hz=8.0)
    np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], values[[0, 1, 3, 4, 5]])
    with pytest.raises(ValueError, match="Nyquist"):
        filtering.butterworth_lowpass(np.zeros(100), fs=FS, cutoff_hz=70.0)


# ----------------------------------------------------------------------------- derivatives


def test_velocity_is_in_units_per_second_not_per_frame() -> None:
    """A knee flexing at 300 deg/s is 300, at any frame rate (v2 reported deg/frame)."""
    for fs in (60.0, 120.0, 240.0):
        t = np.arange(0, 1.0, 1.0 / fs)
        np.testing.assert_allclose(filtering.derivative(300.0 * t, t), 300.0, rtol=1e-9)


def test_velocity_uses_true_timestamps_when_frames_are_uneven() -> None:
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.005, 0.012, 200))
    velocity = filtering.derivative(50.0 * t + 3.0, t)
    np.testing.assert_allclose(velocity, 50.0, rtol=1e-9)
    assert not filtering.is_uniform(t) and filtering.is_uniform(np.arange(50) / FS)
    with pytest.raises(ValueError, match="same length"):
        filtering.derivative(np.zeros(5), np.zeros(4))


def test_resampling_to_an_even_grid() -> None:
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.006, 0.011, 300))
    values = np.stack([np.sin(t), np.cos(t)], axis=1)
    values[100] = np.nan
    resampled, grid = filtering.resample_uniform(values, t)
    assert filtering.is_uniform(grid)
    finite = np.isfinite(resampled[:, 0])
    np.testing.assert_allclose(resampled[finite, 0], np.sin(grid[finite]), atol=1e-4)
    assert (~finite).sum() >= 1  # the missing sample is not papered over


def test_residual_analysis_decreases_with_cutoff() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(0, 4.0, 1.0 / FS)
    noisy = 30 * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 1.0, t.size)
    residuals = filtering.residual_analysis(noisy, fs=FS, cutoffs_hz=[4.0, 8.0, 16.0, 32.0])
    assert list(residuals) == [4.0, 8.0, 16.0, 32.0]
    values = list(residuals.values())
    assert values == sorted(values, reverse=True)
    assert values[1] == pytest.approx(1.0, abs=0.2)  # at 8 Hz the residual is about the noise


# ----------------------------------------------------------------------------- two-phone sync


def test_offset_between_two_recordings_of_a_clap() -> None:
    rng = np.random.default_rng(4)
    fs = 1000.0
    clap = np.exp(-np.arange(200) / 20.0)

    def recording(at_s: float) -> np.ndarray:
        out = rng.normal(0, 0.02, 3000)
        start = int(at_s * fs)
        out[start : start + clap.size] += clap
        return out

    result = sync.estimate_offset(recording(1.250), recording(0.400), fs)
    assert result.offset_s == pytest.approx(0.850, abs=0.002)
    assert result.peak_correlation > 0.8
    aligned = sync.align_timestamps(np.array([400.0]), result)
    assert aligned[0] == pytest.approx(1250.0, abs=2.0)
    with pytest.raises(ValueError, match="constant"):
        sync.estimate_offset(np.ones(10), recording(0.4), fs)
