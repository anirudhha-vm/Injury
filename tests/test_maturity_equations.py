"""Maturity-offset equations against reference values worked out by hand from brief §2.1.

Each reference value is the arithmetic written out in the comment, not the output of the code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from youthtwin.growth.khamis_roche import percent_predicted_adult_height
from youthtwin.growth.maturity_equations import (
    mirwald2002,
    moore2015,
    moore2015_boys_stature,
)
from youthtwin.growth.verification import (
    UnverifiedCoefficientsError,
    UnverifiedCoefficientsWarning,
    is_verified,
)

TOL = 1e-9


def test_moore_boys_reference() -> None:
    # -8.128741 + 0.0070346 * (13 * 80 = 1040) = -8.128741 + 7.315984 = -0.812757
    est = moore2015("male", age=13.0, stature=155.0, sitting_height=80.0)
    assert est.equation_name == "moore2015_boys"
    assert est.mo == pytest.approx(-0.812757, abs=TOL)
    assert est.aphv == pytest.approx(13.812757, abs=TOL)


def test_moore_girls_reference() -> None:
    # -7.709133 + 0.0042232 * (12 * 150 = 1800) = -7.709133 + 7.60176 = -0.107373
    est = moore2015("female", age=12.0, stature=150.0)
    assert est.equation_name == "moore2015_girls"
    assert est.mo == pytest.approx(-0.107373, abs=TOL)
    assert est.aphv == pytest.approx(12.107373, abs=TOL)


def test_moore_boys_stature_reference() -> None:
    # -7.999994 + 0.0036124 * (13 * 155 = 2015) = -7.999994 + 7.278986 = -0.721008
    est = moore2015_boys_stature(age=13.0, stature=155.0)
    assert est.equation_name == "moore2015_boys_stature"
    assert est.mo == pytest.approx(-0.721008, abs=TOL)


def test_mirwald_boys_reference() -> None:
    # leg = 155 - 80 = 75; mass/stature*100 = 45/155*100 = 29.0322580645...
    # -9.236 + 0.0002708*6000 - 0.001663*975 + 0.007216*1040 + 0.02292*29.0322580645
    # = -9.236 + 1.6248 - 1.621425 + 7.50464 + 0.6654193548 = -1.0625656452
    est = mirwald2002(
        "male", age=13.0, stature=155.0, sitting_height=80.0, mass=45.0, synthetic=False
    )
    assert est.equation_name == "mirwald2002_boys"
    assert est.mo == pytest.approx(-1.0625656452, abs=1e-9)
    assert est.aphv == pytest.approx(14.0625656452, abs=1e-9)


def test_mirwald_girls_reference_on_synthetic_data() -> None:
    # leg = 150 - 78 = 72; mass/stature*100 = 42/150*100 = 28
    # -9.376 + 0.0001882*5616 + 0.0022*864 + 0.005841*936 - 0.002658*504 + 0.07693*28
    # = -9.376 + 1.0569312 + 1.9008 + 5.467176 - 1.339632 + 2.15404 = -0.1366848
    kwargs = {"age": 12.0, "stature": 150.0, "sitting_height": 78.0, "mass": 42.0}
    if is_verified("mirwald_girls"):
        est = mirwald2002("female", synthetic=True, **kwargs)
    else:
        with pytest.warns(UnverifiedCoefficientsWarning):
            est = mirwald2002("female", synthetic=True, **kwargs)
    assert est.equation_name == "mirwald2002_girls"
    assert est.mo == pytest.approx(-0.1366848, abs=TOL)


def test_mirwald_girls_blocked_on_real_data_until_verified() -> None:
    if is_verified("mirwald_girls"):
        pytest.skip("coefficients verified by a human")
    with pytest.raises(UnverifiedCoefficientsError):
        mirwald2002(
            "female", age=12.0, stature=150.0, sitting_height=78.0, mass=42.0, synthetic=False
        )


def test_families_give_different_answers_for_the_same_girl() -> None:
    """v2 defect: the Moore girls equation was labelled 'Mirwald'."""
    moore = moore2015("female", age=12.0, stature=150.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedCoefficientsWarning)
        mirwald = mirwald2002(
            "female", age=12.0, stature=150.0, sitting_height=78.0, mass=42.0, synthetic=True
        )
    assert moore.equation_name != mirwald.equation_name
    assert abs(float(moore.mo) - float(mirwald.mo)) > 1e-3


def test_vectorised_and_aphv_identity() -> None:
    age = np.array([10.0, 12.0, 14.0, 16.0])
    stature = np.array([138.0, 150.0, 163.0, 172.0])
    sitting = stature * 0.52
    est = moore2015("male", age, stature, sitting)
    assert est.mo.shape == (4,)
    np.testing.assert_allclose(est.aphv, age - est.mo)
    assert np.all(np.diff(est.mo) > 0)  # older and taller => further past PHV


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"age": 6.0, "stature": 120.0, "sitting_height": 63.0}, "age"),
        ({"age": 21.0, "stature": 175.0, "sitting_height": 91.0}, "age"),
        ({"age": 12.0, "stature": 80.0, "sitting_height": 42.0}, "stature"),
        ({"age": 12.0, "stature": 220.0, "sitting_height": 114.0}, "stature"),
        ({"age": 12.0, "stature": 150.0, "sitting_height": 60.0}, "ratio"),
        ({"age": 12.0, "stature": 150.0, "sitting_height": 95.0}, "ratio"),
        ({"age": float("nan"), "stature": 150.0, "sitting_height": 78.0}, "non-finite"),
    ],
)
def test_implausible_inputs_raise(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        moore2015("male", **kwargs)


def test_stature_in_metres_is_caught() -> None:
    with pytest.raises(ValueError, match="stature"):
        moore2015("female", age=12.0, stature=1.50)


def test_boys_require_sitting_height_and_sex_is_checked() -> None:
    with pytest.raises(ValueError, match="sitting_height"):
        moore2015("male", age=13.0, stature=155.0)
    with pytest.raises(ValueError, match="sex"):
        moore2015("boy", age=13.0, stature=155.0, sitting_height=80.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sex"):
        mirwald2002(
            "x",  # type: ignore[arg-type]
            age=13.0,
            stature=155.0,
            sitting_height=80.0,
            mass=45.0,
            synthetic=True,
        )
    with pytest.raises(ValueError, match="mass"):
        mirwald2002("male", age=13.0, stature=155.0, sitting_height=80.0, mass=0.0, synthetic=True)


def test_khamis_roche_is_not_implemented_without_coefficients() -> None:
    with pytest.raises(NotImplementedError, match=r"TODO\(human\)"):
        percent_predicted_adult_height("male", 13.0, 155.0, 45.0, 170.0)
