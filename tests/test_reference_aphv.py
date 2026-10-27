"""Reference age at PHV from dense growth records (used to score the twin on real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import reference_aphv
from youthtwin.sim.growth_sim import pb1_height, pb1_peak_offset
from youthtwin.utils.config import load_config

CONFIG = load_config("external/berkeley.yaml")["reference"]
# Berkeley schedule: annual 4-8 y, then 6-monthly to 21 y.
AGES = np.concatenate([np.arange(4.0, 8.0, 1.0), np.arange(8.0, 21.01, 0.5)])


def _child(theta: float, s0: float, s1: float, noise: float, seed: int) -> tuple[np.ndarray, float]:
    one = np.array([1.0])
    height = pb1_height(AGES, 175.0 * one, 162.0 * one, s0 * one, s1 * one, theta * one)
    rng = np.random.default_rng(seed)
    true = theta + float(pb1_peak_offset(np.array([s0]), np.array([s1]))[0])
    return height + rng.normal(0.0, noise, AGES.size), true


@pytest.mark.parametrize(
    ("theta", "s0", "s1"), [(13.0, 0.11, 1.2), (11.0, 0.13, 1.0), (15.0, 0.1, 1.4)]
)
def test_preece_baines_fit_recovers_a_known_peak(theta: float, s0: float, s1: float) -> None:
    height, true = _child(theta, s0, s1, noise=0.3, seed=1)
    fit = reference_aphv.fit_preece_baines(AGES, height, initial_theta=12.0)
    assert fit.aphv == pytest.approx(true, abs=0.1)
    assert fit.rmse_cm < 0.5 and fit.peak_velocity_cm_per_year > 5.0


def test_spline_peak_agrees_with_the_parametric_peak() -> None:
    height, true = _child(13.0, 0.11, 1.2, noise=0.3, seed=2)
    assert reference_aphv.spline_peak_age(AGES, height) == pytest.approx(true, abs=0.4)


def test_reference_table_flags_exclusions() -> None:
    good, true = _child(13.0, 0.11, 1.2, noise=0.3, seed=3)
    early, _ = _child(13.0, 0.11, 1.2, noise=0.3, seed=4)
    keep = AGES <= 13.5  # record stops 0.5 y after the peak: cannot cover 2 y after it
    frame = pd.concat(
        [
            pd.DataFrame({"athlete_id": "a", "sex": "male", "age": AGES, "stature": good}),
            pd.DataFrame(
                {"athlete_id": "b", "sex": "male", "age": AGES[keep], "stature": early[keep]}
            ),
        ]
    )
    table = reference_aphv.reference_table(frame, CONFIG).set_index("athlete_id")
    assert table.loc["a", "included"] and table.loc["a", "reference_aphv"] == pytest.approx(
        true, abs=0.1
    )
    assert not table.loc["b", "included"]
    assert "coverage" in table.loc["b", "exclusion_reason"]
    assert table.loc["a", "exclusion_reason"] == ""


def test_a_curve_without_a_spurt_is_excluded() -> None:
    linear = 110.0 + 5.0 * (AGES - 4.0)  # steady growth, no spurt
    frame = pd.DataFrame({"athlete_id": "c", "sex": "female", "age": AGES, "stature": linear})
    row = reference_aphv.reference_table(frame, CONFIG).iloc[0]
    assert not row["included"] and row["exclusion_reason"]


def test_short_record_has_no_spline_reference() -> None:
    assert np.isnan(reference_aphv.spline_peak_age(AGES[:4], 120.0 + 5.0 * AGES[:4]))
