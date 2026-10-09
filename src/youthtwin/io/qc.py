"""Anthropometry quality control. Flags problems; never drops or corrects data.

Technical error of measurement (TEM) for duplicate measures uses the Dahlberg formula:
``TEM = sqrt(sum(d^2) / (2 n))`` where ``d`` is the difference within each pair.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

MEASURES: dict[str, str] = {
    "stature": "stature_cm",
    "sitting_height": "sitting_height_cm",
    "mass": "mass_kg",
}
"""Measure name -> key of its limit in ``configs/io/qc.yaml``."""

FLAG_COLUMNS = ["athlete_id", "date", "flag", "detail"]
DAYS_PER_YEAR = 365.25


def technical_error_of_measurement(first: ArrayLike, second: ArrayLike) -> float:
    """Dahlberg TEM for paired duplicate measurements (same unit as the input)."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("TEM needs two non-empty arrays of equal shape")
    return float(np.sqrt(np.sum((a - b) ** 2) / (2.0 * a.size)))


def tem_by_measurer(anthropometrics: pd.DataFrame) -> pd.DataFrame:
    """TEM per measurer per measure, with the number of pairs it is based on."""
    records: list[dict[str, Any]] = []
    for measurer, group in anthropometrics.groupby("measurer_id"):
        for measure in MEASURES:
            records.append(
                {
                    "measurer_id": measurer,
                    "measure": measure,
                    "n_pairs": len(group),
                    "tem": technical_error_of_measurement(
                        group[f"{measure}_1"].to_numpy(), group[f"{measure}_2"].to_numpy()
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=["measurer_id", "measure", "n_pairs", "tem"])


def anthropometry_flags(anthropometrics: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Return one row per QC problem: ``athlete_id, date, flag, detail``."""
    frame = anthropometrics.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    flags: list[pd.DataFrame] = []

    def add(mask: pd.Series[bool], flag: str, detail: pd.Series[str]) -> None:
        if mask.any():
            hit = frame.loc[mask, ["athlete_id", "date"]].copy()
            hit["flag"] = flag
            hit["detail"] = detail[mask]
            flags.append(hit)

    for measure, limit_key in MEASURES.items():
        difference = (frame[f"{measure}_1"] - frame[f"{measure}_2"]).abs()
        limit = float(config["duplicate_limit"][limit_key])
        add(
            difference > limit,
            f"duplicate_disagreement_{measure}",
            "difference " + difference.round(2).astype(str) + f" > {limit}",
        )

    stature = frame[["stature_1", "stature_2"]].mean(axis=1)
    sitting = frame[["sitting_height_1", "sitting_height_2"]].mean(axis=1)
    ratio = sitting / stature
    low, high = config["sitting_to_stature_ratio"]
    add(
        (ratio < low) | (ratio > high),
        "implausible_sitting_ratio",
        "ratio " + ratio.round(3).astype(str),
    )

    same_athlete = frame["athlete_id"].eq(frame["athlete_id"].shift())
    change = stature.diff()
    days = pd.to_datetime(frame["date"]).diff().dt.days
    tolerance = float(config["stature_decrease_tolerance_cm"])
    add(
        same_athlete & (change < -tolerance),
        "stature_decrease",
        "change " + change.round(2).astype(str) + " cm since previous visit",
    )
    velocity = change / (days / DAYS_PER_YEAR)
    add(
        same_athlete
        & (days >= int(config["min_days_between_visits_for_velocity"]))
        & (velocity > float(config["max_height_velocity_cm_per_year"])),
        "implausible_height_velocity",
        "velocity " + velocity.round(1).astype(str) + " cm/year",
    )

    if not flags:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    return pd.concat(flags, ignore_index=True)[FLAG_COLUMNS]
