"""Natural cubic spline basis: the mathematical properties the growth model relies on."""

from __future__ import annotations

import numpy as np
import pytest

from youthtwin.growth import splines

KNOTS = (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0)
LINEAR = 4.0
COEFS = np.array([1.5, -2.0, 0.7, 3.0])


def _h(z: np.ndarray) -> np.ndarray:
    return np.asarray(splines.spline_value(z, LINEAR, COEFS, KNOTS))


def _dh(z: np.ndarray) -> np.ndarray:
    return np.asarray(splines.spline_derivative(z, LINEAR, COEFS, KNOTS))


def test_basis_count_and_knot_validation() -> None:
    assert splines.n_basis(KNOTS) == 4
    assert len(splines.basis(np.zeros(3), KNOTS)) == 4
    with pytest.raises(ValueError, match="at least 3"):
        splines.check_knots([0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        splines.check_knots([0.0, 2.0, 1.0])


def test_reference_value_three_knots() -> None:
    # Knots (0, 1, 2): one basis function, scale (2 - 0)^2 = 4, gap 2 - 1 = 1.
    # N(z) = [ z+^3 - (z-1)+^3 * (2-0)/1 + (z-2)+^3 * (1-0)/1 ] / 4
    # At z = 1.5: [3.375 - 0.125 * 2 + 0] / 4 = 0.78125
    # At z = 3:   [27 - 8 * 2 + 1] / 4 = 3.0
    (term,) = splines.basis(np.array([1.5, 3.0, -1.0]), (0.0, 1.0, 2.0))
    np.testing.assert_allclose(term, [0.78125, 3.0, 0.0])
    # Derivative at 1.5: 3 * [2.25 - 0.25 * 2] / 4 = 1.3125
    (slope,) = splines.basis_derivative(np.array([1.5]), (0.0, 1.0, 2.0))
    np.testing.assert_allclose(slope, [1.3125])


def test_linear_outside_boundary_knots() -> None:
    below = np.array([-20.0, -15.0, -10.0, -7.0])
    above = np.array([4.0, 8.0, 12.0, 30.0])
    for z in (below, above):
        slopes = np.diff(_h(z)) / np.diff(z)
        np.testing.assert_allclose(slopes, slopes[0], rtol=1e-9)
        np.testing.assert_allclose(_dh(z), slopes[0], rtol=1e-9)
    # Below the first knot only the linear term is active.
    np.testing.assert_allclose(_h(below), LINEAR * below)


def test_value_and_first_two_derivatives_are_continuous_at_knots() -> None:
    eps = 1e-5
    for knot in KNOTS:
        left, right = np.array([knot - eps]), np.array([knot + eps])
        assert _h(left)[0] == pytest.approx(_h(right)[0], abs=1e-3)
        assert _dh(left)[0] == pytest.approx(_dh(right)[0], abs=1e-3)
        second_left = (_dh(left) - _dh(left - eps)) / eps
        second_right = (_dh(right + eps) - _dh(right)) / eps
        assert second_left[0] == pytest.approx(second_right[0], abs=1e-2)


def test_analytic_derivative_matches_finite_difference() -> None:
    z = np.linspace(-8.0, 5.0, 53)
    step = 1e-6
    numeric = (_h(z + step) - _h(z - step)) / (2 * step)
    np.testing.assert_allclose(_dh(z), numeric, rtol=1e-6, atol=1e-6)


def test_spans_the_same_space_as_an_independent_natural_spline() -> None:
    """Any natural cubic spline on these knots is reproduced exactly by the basis."""
    interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(0)
    reference = interpolate.CubicSpline(KNOTS, rng.normal(size=len(KNOTS)), bc_type="natural")
    z = np.linspace(KNOTS[0], KNOTS[-1], 200)
    design = np.column_stack([np.ones_like(z), z, *splines.basis(z, KNOTS)])
    solution, *_ = np.linalg.lstsq(design, reference(z), rcond=None)
    np.testing.assert_allclose(design @ solution, reference(z), atol=1e-8)


def test_numpy_and_pytensor_agree() -> None:
    pt = pytest.importorskip("pytensor.tensor")
    z = np.linspace(-8.0, 5.0, 40)
    symbolic_z = pt.dvector("z")
    value = splines.spline_value(symbolic_z, LINEAR, COEFS, KNOTS, xp=pt)
    slope = splines.spline_derivative(symbolic_z, LINEAR, COEFS, KNOTS, xp=pt)
    np.testing.assert_allclose(value.eval({symbolic_z: z}), _h(z), rtol=1e-10)
    np.testing.assert_allclose(slope.eval({symbolic_z: z}), _dh(z), rtol=1e-10)
