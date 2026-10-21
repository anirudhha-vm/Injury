"""The Berkeley Child Guidance Study as real growth data for the twin (M10, Part 1).

136 children (66 boys, 70 girls) born 1928-29, measured 6-monthly from 8 to 21 y; distributed
in the ``sitar`` R package (GPL >= 2). Stem length is sitting height. The raw file stays in the
git-ignored ``data/external/``; only derived tables are written to ``outputs/``.

Each child's full record gives a reference age at PHV (``reference_aphv``). The twin, Moore
and Mirwald then see only a one-season window of the same child, as they would in the study.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.growth.maturity_equations import Sex, mirwald2002, moore2015

SEX_CODES = {"1": "male", "2": "female"}
ID_PREFIX = "berkeley-"


def load_berkeley(path: Path) -> pd.DataFrame:
    """Tidy long table: ``athlete_id, sex, age, stature, sitting_height, mass, synthetic``."""
    import pyreadr  # optional dependency, only needed for this public dataset

    raw = next(iter(pyreadr.read_r(str(path)).values()))
    return tidy_berkeley(raw)


def tidy_berkeley(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and recode the columns of the ``berkeley`` data frame."""
    frame = pd.DataFrame(
        {
            "athlete_id": ID_PREFIX + raw["id"].astype(str),
            "sex": raw["sex"].astype(str).map(SEX_CODES),
            "age": raw["age"].astype(float),
            "stature": raw["height"].astype(float),
            "sitting_height": raw["stem.length"].astype(float),
            "mass": raw["weight"].astype(float),
        }
    )
    if frame["sex"].isna().any():
        raise ValueError("Unexpected sex codes in the Berkeley data")
    frame["synthetic"] = False
    return frame.sort_values(["athlete_id", "age"]).reset_index(drop=True)


def draw_windows(
    frame: pd.DataFrame, athletes: list[str], config: Mapping[str, Any], seed: int
) -> pd.DataFrame:
    """One season of complete measurements per athlete, starting at a random measurement age.

    The start is drawn uniformly among the athlete's measurement ages inside
    ``start_age_range_years`` whose window contains at least ``min_measurements`` complete
    visits. Athletes without any such start are dropped.
    """
    rng = np.random.default_rng(seed)
    length = float(config["length_years"])
    low, high = (float(v) for v in config["start_age_range_years"])
    complete = frame.dropna(subset=["stature", "sitting_height", "mass"])
    parts = []
    for athlete in athletes:
        record = complete[complete["athlete_id"] == athlete]
        ages = record["age"].to_numpy()
        starts = [
            a
            for a in ages
            if low <= a <= high
            and ((ages >= a - 1e-9) & (ages <= a + length + 1e-9)).sum()
            >= int(config["min_measurements"])
        ]
        if not starts:
            continue
        start = float(rng.choice(starts))
        window = record[(ages >= start - 1e-9) & (ages <= start + length + 1e-9)].copy()
        window["window_start"] = start
        parts.append(window)
    return pd.concat(parts, ignore_index=True)


def equation_estimates(window: pd.DataFrame) -> pd.DataFrame:
    """Moore 2015 (and Mirwald 2002 for boys) APHV from each athlete's window.

    As for the twin's prior, the estimate is taken at the visit the equation places closest to
    PHV. Mirwald's girls equation stays gated until a human verifies it, so girls get NaN.
    """
    rows = []
    for (athlete, sex_value), part in window.groupby(["athlete_id", "sex"], sort=True):
        sex: Sex = "male" if sex_value == "male" else "female"
        args = (part["age"], part["stature"], part["sitting_height"])
        moore = moore2015(sex, *args)
        row: dict[str, Any] = {
            "athlete_id": athlete,
            "moore_aphv": float(moore.aphv[np.argmin(np.abs(moore.mo))]),
            "mirwald_aphv": np.nan,
        }
        if sex == "male":
            mirwald = mirwald2002(sex, *args, part["mass"], synthetic=False)
            row["mirwald_aphv"] = float(mirwald.aphv[np.argmin(np.abs(mirwald.mo))])
        rows.append(row)
    return pd.DataFrame(rows)
