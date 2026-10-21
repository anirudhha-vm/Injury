"""Berkeley loader, season windows and equation estimates (no download needed)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import berkeley
from youthtwin.growth.maturity_equations import moore2015
from youthtwin.growth.sitar_bayes import growth_data_from_measurements
from youthtwin.utils.config import load_config

CONFIG = load_config("external/berkeley.yaml")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """A small frame shaped like sitar's ``berkeley`` (id factor, sex 1/2, stem.length)."""
    ages = np.arange(8.0, 18.01, 0.5)
    rows = []
    for ident, sex, scale in (("201", "1", 1.0), ("301", "2", 0.96), ("302", "2", 0.95)):
        height = scale * (125.0 + 5.5 * (ages - 8.0))
        rows.append(
            pd.DataFrame(
                {
                    "id": ident,
                    "age": ages,
                    "height": height,
                    "stem.length": 0.52 * height,
                    "weight": 25.0 + 3.0 * (ages - 8.0),
                    "sex": sex,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame.loc[3, "height"] = np.nan  # one missing measurement
    return frame


def test_tidy_recodes_and_marks_real(raw: pd.DataFrame) -> None:
    tidy = berkeley.tidy_berkeley(raw)
    assert set(tidy["sex"]) == {"male", "female"}
    assert tidy["athlete_id"].str.startswith("berkeley-").all()
    assert not tidy["synthetic"].any()
    assert list(tidy.columns) == [
        "athlete_id",
        "sex",
        "age",
        "stature",
        "sitting_height",
        "mass",
        "synthetic",
    ]
    with pytest.raises(ValueError, match="sex codes"):
        berkeley.tidy_berkeley(raw.assign(sex="3"))


def test_windows_are_one_complete_season(raw: pd.DataFrame) -> None:
    tidy = berkeley.tidy_berkeley(raw)
    athletes = sorted(tidy["athlete_id"].unique())
    windows = berkeley.draw_windows(tidy, athletes, CONFIG["window"], seed=1)
    low, high = CONFIG["window"]["start_age_range_years"]
    for _, part in windows.groupby("athlete_id"):
        start = part["window_start"].iloc[0]
        assert low <= start <= high
        assert part["age"].between(start, start + CONFIG["window"]["length_years"]).all()
        assert len(part) >= CONFIG["window"]["min_measurements"]
        assert part[["stature", "sitting_height", "mass"]].notna().all().all()
    # Reproducible, and a different seed moves at least one window.
    again = berkeley.draw_windows(tidy, athletes, CONFIG["window"], seed=1)
    pd.testing.assert_frame_equal(windows, again)
    other = berkeley.draw_windows(tidy, athletes, CONFIG["window"], seed=2)
    assert not windows["window_start"].equals(other["window_start"])


def test_equation_estimates_match_the_equations(raw: pd.DataFrame) -> None:
    tidy = berkeley.tidy_berkeley(raw)
    windows = berkeley.draw_windows(tidy, sorted(tidy["athlete_id"].unique()), CONFIG["window"], 0)
    table = berkeley.equation_estimates(windows).set_index("athlete_id")
    boy = windows[windows["athlete_id"] == "berkeley-201"]
    moore = moore2015("male", boy["age"], boy["stature"], boy["sitting_height"])
    assert table.loc["berkeley-201", "moore_aphv"] == pytest.approx(
        moore.aphv[np.argmin(np.abs(moore.mo))]
    )
    assert np.isfinite(table.loc["berkeley-201", "mirwald_aphv"])
    assert np.isnan(table.loc["berkeley-301", "mirwald_aphv"])  # girls equation is gated


def test_real_measurements_enter_the_growth_model(raw: pd.DataFrame) -> None:
    tidy = berkeley.tidy_berkeley(raw).dropna()
    girls: Any = tidy[tidy["sex"] == "female"]
    data = growth_data_from_measurements(girls, "female", synthetic=False)
    assert data.athlete_ids == ("berkeley-301", "berkeley-302")
    assert data.n_observations == len(girls) and not data.synthetic
    assert np.bincount(data.athlete_index).tolist() == girls.groupby("athlete_id").size().tolist()
