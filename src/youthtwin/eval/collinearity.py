"""Collinearity check for model inputs.

v2 fed four "growth" inputs to its model that were all functions of one number. The variance
inflation factor ``VIF_j = 1 / (1 - R^2_j)``, where ``R^2_j`` comes from regressing input ``j``
on the others, flags inputs that carry almost no information of their own. A VIF above the
threshold triggers a warning, not an error: prediction can tolerate collinearity, but the sign
and size of an individual input's effect cannot then be interpreted.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


class CollinearityWarning(UserWarning):
    """Emitted when model inputs are close to linear combinations of each other."""


def variance_inflation(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """VIF of each column given the others, on rows where all of them are present."""
    data = frame[list(columns)].dropna().to_numpy(dtype=np.float64)
    if data.shape[0] <= data.shape[1] + 1:
        raise ValueError("Too few complete rows to compute variance inflation factors")
    spread = data.std(axis=0)
    if np.any(spread == 0):
        constant = [c for c, s in zip(columns, spread, strict=True) if s == 0]
        raise ValueError(f"Constant columns have no variance inflation factor: {constant}")
    standardised = (data - data.mean(axis=0)) / spread
    factors: dict[str, float] = {}
    for j, name in enumerate(columns):
        others = np.delete(standardised, j, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        residual = (
            standardised[:, j] - design @ np.linalg.lstsq(design, standardised[:, j], rcond=None)[0]
        )
        unexplained = float(np.mean(residual**2))  # the target has unit variance
        factors[name] = float("inf") if unexplained <= 1e-12 else 1.0 / unexplained
    return pd.Series(factors, name="vif")


def warn_if_collinear(
    frame: pd.DataFrame, columns: Sequence[str], threshold: float = 10.0
) -> pd.Series:
    """Compute VIFs and warn about every input above ``threshold``. Returns the VIFs."""
    factors = variance_inflation(frame, columns)
    high = factors[factors > threshold]
    if not high.empty:
        listing = ", ".join(f"{name} ({value:.0f})" for name, value in high.items())
        warnings.warn(
            f"Collinear model inputs (VIF > {threshold:g}): {listing}. Individual effects of "
            "these inputs cannot be interpreted separately.",
            CollinearityWarning,
            stacklevel=2,
        )
    return factors
