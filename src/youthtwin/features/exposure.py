"""Training-exposure features per athlete-week.

All features for week ``w`` use weeks up to and including ``w`` only, so they are known at the
prediction time (the end of week ``w``).

- ``hours_total``: training + match hours in the week.
- ``ewma_hours``: exponentially weighted moving average of weekly hours with a configurable
  half-life, ``e_w = (1 - d) * hours_w + d * e_{w-1}`` with ``d = 0.5 ** (1 / half_life)``,
  started at the first observed week.
- ``hours_change``: this week's hours minus the previous week's EWMA (a spike measure in
  hours, not a ratio).

The acute:chronic workload ratio is not computed: the brief excludes it as a primary feature,
and ratios of two noisy loads behave badly when the chronic load is small.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EXPOSURE_FEATURES = ["hours_total", "sessions", "ewma_hours", "hours_change"]


def ewma(values: FloatArray, half_life_weeks: float) -> FloatArray:
    """Exponentially weighted moving average along a 1D series; missing weeks carry forward."""
    if half_life_weeks <= 0:
        raise ValueError("half_life_weeks must be positive")
    decay = 0.5 ** (1.0 / half_life_weeks)
    out = np.empty(values.size, dtype=np.float64)
    state = np.nan
    for k, value in enumerate(values):
        if np.isfinite(value):
            state = value if np.isnan(state) else (1.0 - decay) * value + decay * state
        out[k] = state
    return out


def exposure_features(exposure_weekly: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Exposure features for every athlete-week in ``exposure_weekly``.

    Returns ``athlete_id, week_start`` plus ``EXPOSURE_FEATURES``.
    """
    half_life = float(config["ewma_half_life_weeks"])
    frame = exposure_weekly.sort_values(["athlete_id", "week_start"]).reset_index(drop=True)
    frame["hours_total"] = frame["hours_training"] + frame["hours_match"]
    smoothed = np.empty(len(frame), dtype=np.float64)
    change = np.empty(len(frame), dtype=np.float64)
    for rows in frame.groupby("athlete_id", sort=False).indices.values():
        hours = frame["hours_total"].to_numpy(dtype=np.float64)[rows]
        average = ewma(hours, half_life)
        smoothed[rows] = average
        change[rows] = hours - np.concatenate([[np.nan], average[:-1]])
    frame["ewma_hours"] = smoothed
    frame["hours_change"] = change
    return frame[["athlete_id", "week_start", *EXPOSURE_FEATURES]]
