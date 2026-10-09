"""Maturity-offset prediction equations (brief §2.1).

Units: age in decimal years; stature and sitting height in cm; mass in kg.
``leg_length = stature - sitting_height``. Maturity offset (MO) is years from peak height
velocity (negative = before PHV); predicted age at PHV is ``APHV = age - MO``.

Two families, kept separate and named (v2 mixed them up):

- **Moore et al. (2015)** — primary. Boys: age x sitting height. Girls: age x stature.
  ``moore2015_boys_stature`` is the stature-only boys variant, for sensitivity analysis.
- **Mirwald et al. (2002)** — secondary, for comparison. The girls coefficients are gated by a
  human-verification flag (see ``youthtwin.growth.verification``).

Known limitations (the reason the Bayesian growth twin exists):

- Predictions regress toward the population mean: APHV is underestimated at younger ages and
  overestimated at older ages; error is larger in early and late maturers (Malina & Koziel 2014)
  and grows with distance from PHV (Moore 2015; Mills et al. 2017).
- Derivation samples were predominantly of white European ancestry; generalisation to Indian
  youth is unknown.

Coefficients are read from ``configs/growth/equations.yaml`` and are point estimates only: these
functions return no uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from youthtwin.growth.verification import EQUATIONS_CONFIG, require_verified
from youthtwin.utils.config import load_config

Sex = Literal["male", "female"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MaturityEstimate:
    """Point estimate of maturity offset and age at PHV (both in years)."""

    mo: FloatArray
    aphv: FloatArray
    equation_name: str


@lru_cache(maxsize=1)
def _equations() -> dict[str, Any]:
    return load_config(EQUATIONS_CONFIG)


def _as_float(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains missing or non-finite values")
    return array


def _check_range(name: str, values: FloatArray, bounds: list[float]) -> None:
    low, high = bounds
    if np.any(values < low) or np.any(values > high):
        raise ValueError(
            f"Implausible {name}: expected {low}-{high}, "
            f"got {float(values.min()):.3f}-{float(values.max()):.3f}"
        )


def _validate(
    age: ArrayLike, stature: ArrayLike, sitting_height: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    limits = _equations()["plausibility"]
    age_a = _as_float("age", age)
    stature_a = _as_float("stature", stature)
    _check_range("age (years)", age_a, limits["age_years"])
    _check_range("stature (cm)", stature_a, limits["stature_cm"])
    if sitting_height is None:
        return age_a, stature_a, None
    sitting_a = _as_float("sitting_height", sitting_height)
    _check_range(
        "sitting height / stature ratio",
        np.asarray(sitting_a / stature_a, dtype=np.float64),
        limits["sitting_to_stature_ratio"],
    )
    return age_a, stature_a, sitting_a


def _estimate(mo: FloatArray, age: FloatArray, name: str) -> MaturityEstimate:
    mo = np.asarray(mo, dtype=np.float64)
    return MaturityEstimate(mo=mo, aphv=np.asarray(age - mo, dtype=np.float64), equation_name=name)


def moore2015(
    sex: Sex, age: ArrayLike, stature: ArrayLike, sitting_height: ArrayLike | None = None
) -> MaturityEstimate:
    """Moore et al. (2015). Boys need sitting height; girls use stature only."""
    coefficients = _equations()["moore2015"]
    if sex == "male":
        if sitting_height is None:
            raise ValueError("moore2015 boys needs sitting_height (or use moore2015_boys_stature)")
        age_a, _, sitting_a = _validate(age, stature, sitting_height)
        assert sitting_a is not None
        c = coefficients["boys"]
        mo = c["intercept"] + c["age_x_sitting_height"] * (age_a * sitting_a)
        return _estimate(mo, age_a, "moore2015_boys")
    if sex == "female":
        age_a, stature_a, _ = _validate(age, stature, sitting_height)
        c = coefficients["girls"]
        mo = c["intercept"] + c["age_x_stature"] * (age_a * stature_a)
        return _estimate(mo, age_a, "moore2015_girls")
    raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")


def moore2015_boys_stature(age: ArrayLike, stature: ArrayLike) -> MaturityEstimate:
    """Stature-only boys variant of Moore (2015); sensitivity analysis only."""
    age_a, stature_a, _ = _validate(age, stature)
    c = _equations()["moore2015"]["boys_stature"]
    mo = c["intercept"] + c["age_x_stature"] * (age_a * stature_a)
    return _estimate(mo, age_a, "moore2015_boys_stature")


def mirwald2002(
    sex: Sex,
    age: ArrayLike,
    stature: ArrayLike,
    sitting_height: ArrayLike,
    mass: ArrayLike,
    *,
    synthetic: bool,
) -> MaturityEstimate:
    """Mirwald et al. (2002). ``synthetic`` states whether the inputs are simulated data.

    The girls equation raises on real data until a human has verified its coefficients.
    """
    age_a, stature_a, sitting_a = _validate(age, stature, sitting_height)
    assert sitting_a is not None
    mass_a = _as_float("mass", mass)
    if np.any(mass_a <= 0):
        raise ValueError("mass must be positive")
    leg = stature_a - sitting_a
    weight_height_ratio = mass_a / stature_a * 100.0
    coefficients = _equations()["mirwald2002"]
    if sex == "male":
        c = coefficients["boys"]
        mo = (
            c["intercept"]
            + c["leg_length_x_sitting_height"] * (leg * sitting_a)
            + c["age_x_leg_length"] * (age_a * leg)
            + c["age_x_sitting_height"] * (age_a * sitting_a)
            + c["mass_over_stature_x100"] * weight_height_ratio
        )
        return _estimate(mo, age_a, "mirwald2002_boys")
    if sex == "female":
        require_verified("mirwald_girls", synthetic=synthetic)
        c = coefficients["girls"]
        mo = (
            c["intercept"]
            + c["leg_length_x_sitting_height"] * (leg * sitting_a)
            + c["age_x_leg_length"] * (age_a * leg)
            + c["age_x_sitting_height"] * (age_a * sitting_a)
            + c["age_x_mass"] * (age_a * mass_a)
            + c["mass_over_stature_x100"] * weight_height_ratio
        )
        return _estimate(mo, age_a, "mirwald2002_girls")
    raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
