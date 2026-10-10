"""Reading marker-based motion capture and aligning it in time with markerless output.

Supported inputs:

- **TRC** (OpenSim marker trajectories, tab-separated text): read natively.
- **C3D**: needs the optional ``ezc3d`` package. TODO(human): confirm which system (Vicon /
  Qualisys) and export format the lab will provide.

Turning marker trajectories into joint angles depends on the lab's marker set and model, which
are not decided yet, so it is not implemented: this module reads trajectories and aligns
already-computed angle curves from the two systems.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from youthtwin.pose import sync

FloatArray = NDArray[np.float64]

_TRC_HEADER_LINES = 5
_MM_TO_M = 0.001


@dataclass(frozen=True)
class MarkerData:
    """Marker trajectories: ``positions`` is (T, n_markers, 3) in metres."""

    names: tuple[str, ...]
    time_s: FloatArray
    positions: FloatArray
    rate_hz: float

    def marker(self, name: str) -> FloatArray:
        try:
            return self.positions[:, self.names.index(name), :]
        except ValueError:
            raise KeyError(f"No marker '{name}'. Available: {list(self.names)}") from None


def read_trc(path: Path) -> MarkerData:
    """Read an OpenSim ``.trc`` file. Positions are converted to metres."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= _TRC_HEADER_LINES or not lines[0].startswith("PathFileType"):
        raise ValueError(f"{path.name} is not a TRC file")
    keys, values = lines[1].split("\t"), lines[2].split("\t")
    header = dict(zip(keys, values, strict=False))
    names = tuple(name for name in lines[3].split("\t")[2:] if name.strip())
    rows = [line.split("\t") for line in lines[_TRC_HEADER_LINES:] if line.strip()]
    table = np.array(
        [
            [float(cell) if cell.strip() else np.nan for cell in row[: 2 + 3 * len(names)]]
            for row in rows
        ]
    )
    positions = table[:, 2:].reshape(len(rows), len(names), 3)
    if header.get("Units", "mm").strip().lower() == "mm":
        positions = positions * _MM_TO_M
    return MarkerData(names, table[:, 1], positions, float(header["DataRate"]))


def write_trc(data: MarkerData, path: Path) -> Path:
    """Write marker trajectories as TRC in millimetres (used for tests and data exchange)."""
    n_frames, n_markers = data.positions.shape[:2]
    axes = "\t".join(f"X{k}\tY{k}\tZ{k}" for k in range(1, n_markers + 1))
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data.rate_hz}\t{data.rate_hz}\t{n_frames}\t{n_markers}\tmm\t{data.rate_hz}\t1\t{n_frames}",
        "Frame#\tTime\t" + "\t\t\t".join(data.names),
        "\t\t" + axes,
    ]
    millimetres = data.positions / _MM_TO_M
    for frame in range(n_frames):
        cells = "\t".join("" if np.isnan(v) else f"{v:.5f}" for v in millimetres[frame].ravel())
        lines.append(f"{frame + 1}\t{data.time_s[frame]:.6f}\t{cells}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_c3d(path: Path) -> MarkerData:
    """Read marker trajectories from a C3D file (needs the optional ``ezc3d`` package)."""
    try:
        ezc3d = importlib.import_module("ezc3d")
    except ImportError as error:
        raise ImportError(
            "Reading C3D needs the optional 'ezc3d' package (`uv pip install ezc3d`), or export "
            "TRC from the mocap software instead."
        ) from error
    file = ezc3d.c3d(str(path))
    points = file["data"]["points"]  # (4, n_markers, T), usually millimetres
    parameters = file["parameters"]["POINT"]
    rate = float(parameters["RATE"]["value"][0])
    scale = _MM_TO_M if str(parameters["UNITS"]["value"][0]).strip().lower() == "mm" else 1.0
    positions = np.transpose(points[:3], (2, 1, 0)) * scale
    time_s = np.arange(positions.shape[0]) / rate
    return MarkerData(tuple(parameters["LABELS"]["value"]), time_s, positions, rate)


def resample(values: FloatArray, time_s: FloatArray, new_time_s: FloatArray) -> FloatArray:
    """Linear interpolation of a 1D curve onto another clock; NaN outside the recorded span."""
    return np.asarray(
        np.interp(new_time_s, time_s, values, left=np.nan, right=np.nan), dtype=np.float64
    )


@dataclass(frozen=True)
class Alignment:
    """``offset_s`` is added to mocap time to express it on the markerless clock."""

    offset_s: float
    peak_correlation: float
    mocap_on_markerless_clock: FloatArray  # mocap curve resampled at the markerless frames


def align_curves(
    markerless: FloatArray,
    markerless_time_s: FloatArray,
    mocap: FloatArray,
    mocap_time_s: FloatArray,
    *,
    max_lag_s: float,
) -> Alignment:
    """Align a mocap angle curve to the markerless one by cross-correlation (e.g. knee angle).

    Both curves are resampled to the markerless frame rate, the lag maximising their
    correlation is found, and the mocap curve is returned on the markerless clock.
    """
    step = float(np.median(np.diff(markerless_time_s)))
    fs = 1.0 / step
    grid_a = np.arange(markerless_time_s[0], markerless_time_s[-1] + 0.5 * step, step)
    grid_b = np.arange(mocap_time_s[0], mocap_time_s[-1] + 0.5 * step, step)
    a = resample(markerless, markerless_time_s, grid_a)
    b = resample(mocap, mocap_time_s, grid_b)
    a = np.where(np.isfinite(a), a, float(np.nanmean(markerless)))
    b = np.where(np.isfinite(b), b, float(np.nanmean(mocap)))
    result = sync.estimate_offset(a, b, fs)
    # estimate_offset works on sample indices from each recording's own start.
    offset = result.offset_s + float(grid_a[0] - grid_b[0])
    if abs(offset) > max_lag_s:
        raise ValueError(
            f"Estimated lag {offset:.2f} s exceeds the allowed {max_lag_s} s; check that both "
            "recordings contain the same trial"
        )
    aligned = resample(mocap, mocap_time_s + offset, markerless_time_s)
    return Alignment(offset, result.peak_correlation, aligned)
