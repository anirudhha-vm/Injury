"""Natural (restricted) cubic spline in truncated-power form.

For knots ``t_1 < ... < t_K`` the curve is

    h(z) = linear * z + sum_j coefs[j] * N_j(z),   j = 1 .. K-2

    N_j(z) = [ (z - t_j)+^3
               - (z - t_{K-1})+^3 * (t_K - t_j) / (t_K - t_{K-1})
               + (z - t_K)+^3     * (t_{K-1} - t_j) / (t_K - t_{K-1}) ] / (t_K - t_1)^2

It is cubic between knots, has continuous first and second derivatives, and is linear outside
the boundary knots (so it extrapolates gently). There is no intercept term: the caller adds one.

The functions take the array module as ``xp`` so exactly the same code runs on NumPy arrays and
on PyTensor variables; the growth model needs the latter because the spline is evaluated at
inputs that depend on model parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np


def check_knots(knots: Sequence[float]) -> tuple[float, ...]:
    """Validate and return knots as a tuple (at least three, strictly increasing)."""
    values = tuple(float(k) for k in knots)
    if len(values) < 3:
        raise ValueError("A natural cubic spline needs at least 3 knots")
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError("Knots must be strictly increasing")
    return values


def n_basis(knots: Sequence[float]) -> int:
    """Number of non-linear basis functions (``len(knots) - 2``)."""
    return len(check_knots(knots)) - 2


def _basis_terms(z: Any, knots: Sequence[float], power: int, xp: Any) -> list[Any]:
    t = check_knots(knots)
    last, second_last = t[-1], t[-2]
    scale = (last - t[0]) ** 2
    gap = last - second_last

    def plus(knot: float) -> Any:
        return xp.maximum(z - knot, 0.0) ** power

    tail_a, tail_b = plus(second_last), plus(last)
    factor = 1.0 if power == 3 else 3.0  # d/dz of (.)+^3 is 3 (.)+^2
    return [
        factor
        * (plus(tj) - tail_a * ((last - tj) / gap) + tail_b * ((second_last - tj) / gap))
        / scale
        for tj in t[:-2]
    ]


def basis(z: Any, knots: Sequence[float], xp: Any = np) -> list[Any]:
    """The ``K - 2`` non-linear basis functions evaluated at ``z``."""
    return _basis_terms(z, knots, 3, xp)


def basis_derivative(z: Any, knots: Sequence[float], xp: Any = np) -> list[Any]:
    """First derivatives of the basis functions at ``z``."""
    return _basis_terms(z, knots, 2, xp)


def spline_value(z: Any, linear: Any, coefs: Any, knots: Sequence[float], xp: Any = np) -> Any:
    """``h(z)`` without intercept. ``coefs`` is indexable with ``K - 2`` entries."""
    value = linear * z
    for j, term in enumerate(basis(z, knots, xp)):
        value = value + coefs[j] * term
    return value


def spline_derivative(z: Any, linear: Any, coefs: Any, knots: Sequence[float], xp: Any = np) -> Any:
    """``h'(z)``: the velocity of the mean curve on the transformed age axis."""
    value = linear + 0.0 * z
    for j, term in enumerate(basis_derivative(z, knots, xp)):
        value = value + coefs[j] * term
    return value


def orthogonalising_transform(knots: Sequence[float], n_grid: int = 200) -> Any:
    """Matrix ``T`` mapping well-conditioned parameters to ``[intercept, linear, *coefs]``.

    The truncated-power basis is strongly collinear, which makes the coefficients hard to
    sample. With design matrix ``D = [1, z, N_1(z), ...]`` on an even grid over the knot range
    and its QR decomposition ``D = Q R``, setting ``[intercept, linear, *coefs] = T @ theta``
    with ``T = sqrt(n_grid) * inv(R)`` gives ``D @ T @ theta = sqrt(n_grid) * Q @ theta``: each
    ``theta`` moves the curve along an orthogonal direction, by about one unit per unit theta.
    """
    t = check_knots(knots)
    grid = np.linspace(t[0], t[-1], n_grid)
    design = np.column_stack([np.ones_like(grid), grid, *basis(grid, t)])
    _, r = np.linalg.qr(design)
    signs = np.sign(np.diag(r))  # make the decomposition unique
    return np.sqrt(n_grid) * np.linalg.inv(signs[:, None] * r)
