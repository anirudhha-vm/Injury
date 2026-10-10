"""Out-of-fold predictions: the single input to every metric, table and figure.

Each model is fitted on the training athletes of a fold and predicts that fold's held-out
athletes, so every athlete-week gets a prediction from a model that never saw that athlete.
The result is one table with the identifiers, the outcome, the columns needed for subgroup
analysis, and one ``risk_<model>`` column per model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from youthtwin.eval import splits
from youthtwin.models.tabular import HazardModel

RISK_PREFIX = "risk_"
ModelFactory = Callable[[pd.DataFrame], HazardModel]


def out_of_fold_predictions(
    frame: pd.DataFrame,
    label: str,
    factories: Mapping[str, ModelFactory],
    folds: Sequence[splits.Split],
    *,
    keep: Sequence[str] = (),
) -> pd.DataFrame:
    """Predict every row from a model trained without that row's athlete.

    ``factories`` maps a model name to a function building a fresh, unfitted model from the
    table. ``keep`` lists extra columns to carry along (for subgroups).
    """
    for fold in folds:
        splits.assert_no_group_overlap(frame["athlete_id"], fold)
    covered = np.sum([fold.test for fold in folds], axis=0)
    if not np.all(covered == 1):
        raise ValueError("Folds must put every row in exactly one test part")
    columns = ["athlete_id", "week_start", *[c for c in keep if c in frame.columns]]
    out = frame[columns].copy()
    out["outcome"] = frame[label].to_numpy()
    out["label"] = label
    out["fold"] = -1
    out["synthetic"] = frame["synthetic"].to_numpy()
    for name in factories:
        out[f"{RISK_PREFIX}{name}"] = np.nan
    for k, fold in enumerate(folds):
        out.loc[fold.test, "fold"] = k
        for name, factory in factories.items():
            model = factory(frame).fit(frame[fold.train], label)
            out.loc[fold.test, f"{RISK_PREFIX}{name}"] = model.predict_proba(frame[fold.test])
    return out.reset_index(drop=True)


def risk_columns(predictions: pd.DataFrame) -> dict[str, str]:
    """Model name -> its risk column."""
    return {
        c.removeprefix(RISK_PREFIX): c for c in predictions.columns if c.startswith(RISK_PREFIX)
    }
