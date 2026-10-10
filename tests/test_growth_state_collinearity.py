"""Growth inputs must not be redundant copies of one number (v2 regression).

v2 used ``bio_risk``, ``phv_proximity``, ``age_norm`` and ``offset``, all derived from a single
maturity estimate. The check warns when any input's variance inflation factor exceeds 10.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from youthtwin.eval.collinearity import CollinearityWarning, variance_inflation, warn_if_collinear
from youthtwin.features.person_period import PersonPeriod


def test_vif_reference_values() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    # c = a + noise with correlation r = 1 / sqrt(1 + 0.25) -> VIF of a and c = 1 / (1 - r^2) = 5.
    frame = pd.DataFrame({"a": a, "b": b, "c": a + 0.5 * rng.normal(size=20000)})
    factors = variance_inflation(frame, ["a", "b", "c"])
    assert factors["a"] == pytest.approx(5.0, rel=0.05)
    assert factors["c"] == pytest.approx(5.0, rel=0.05)
    assert factors["b"] == pytest.approx(1.0, abs=0.01)


def test_v2_style_redundant_inputs_are_flagged() -> None:
    rng = np.random.default_rng(1)
    offset = rng.normal(0.0, 1.5, 2000)
    v2 = pd.DataFrame(
        {
            "offset": offset,
            "age_norm": (offset + 13.5) / 18.0,  # an affine function of the same number
            "phv_proximity": offset * 0.8 + rng.normal(0, 0.01, 2000),
            "training_hours": rng.normal(8.0, 2.0, 2000),
        }
    )
    with pytest.warns(CollinearityWarning, match="offset"):
        factors = warn_if_collinear(v2, list(v2.columns))
    assert np.isinf(factors["age_norm"]) and factors["phv_proximity"] > 1000
    assert factors["training_hours"] < 1.1


def test_independent_growth_quantities_do_not_warn() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(3000, 5)), columns=list("abcde"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CollinearityWarning)
        factors = warn_if_collinear(frame, list(frame.columns))
    assert factors.max() < 1.1


def test_synthetic_cohort_growth_inputs(person_period: PersonPeriod) -> None:
    """The growth inputs of the synthetic athlete-week table overlap but are not copies."""
    growth = [
        "moore_maturity_offset",
        "obs_height_velocity",
        "obs_leg_length_velocity",
        "obs_mass_velocity",
    ]
    factors = variance_inflation(person_period.frame, growth)
    assert factors.index.tolist() == growth
    assert (factors >= 1.0).all()
    # Height and leg-length velocity overlap substantially: the reason both are not used
    # together when reading off effect directions (see test_hazard_models.py).
    assert factors["obs_height_velocity"] > 2.0 and factors["obs_leg_length_velocity"] > 2.0
    assert factors["obs_mass_velocity"] < factors["obs_height_velocity"]


def test_degenerate_inputs_are_rejected() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 1.0, 5.0]})
    with pytest.raises(ValueError, match="Too few"):
        variance_inflation(frame, ["a", "b"])
    constant = pd.DataFrame({"a": np.arange(50.0), "b": np.ones(50)})
    with pytest.raises(ValueError, match="Constant"):
        variance_inflation(constant, ["a", "b"])
