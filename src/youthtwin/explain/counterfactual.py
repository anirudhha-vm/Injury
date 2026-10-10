"""What-if analysis: how the model's predicted risk changes when inputs are changed.

This is **model sensitivity, not a causal effect**. It answers "what would this model predict if
this athlete's recorded exposure were 20% lower and nothing else changed", which is a statement
about the fitted function. It does not say that reducing an athlete's training would reduce
their injury risk by that amount: the model was not built to identify causal effects, inputs
cannot really change in isolation, and unmeasured factors are not held fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from youthtwin.models.tabular import HazardModel

DISCLAIMER = "Model sensitivity to a changed input, not an estimate of a causal effect."


def scale_columns(frame: pd.DataFrame, columns: Sequence[str], factor: float) -> pd.DataFrame:
    """A copy of ``frame`` with the given numeric columns multiplied by ``factor``."""
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(f"Columns not in the table: {absent}")
    return frame.assign(**{c: frame[c] * factor for c in columns})


def what_if(model: HazardModel, frame: pd.DataFrame, changed: pd.DataFrame) -> pd.DataFrame:
    """Predicted risk before and after a change, per row, with the disclaimer attached."""
    if len(changed) != len(frame):
        raise ValueError("The changed table must have the same rows as the original")
    before, after = model.predict_proba(frame), model.predict_proba(changed)
    out = frame[["athlete_id", "week_start"]].copy()
    out["risk_before"], out["risk_after"] = before, after
    out["risk_change"] = after - before
    out.attrs["disclaimer"] = DISCLAIMER
    return out


def summarise_what_if(result: pd.DataFrame) -> Mapping[str, float | str]:
    change = result["risk_change"].to_numpy()
    return {
        "mean_risk_before": float(result["risk_before"].mean()),
        "mean_risk_after": float(result["risk_after"].mean()),
        "mean_change": float(np.mean(change)),
        "share_of_rows_with_lower_risk": float(np.mean(change < 0)),
        "interpretation": DISCLAIMER,
    }
