"""Performance within subgroups: the fairness and robustness table.

Subgroups: sex, maturity stratum (pre / circa / post-PHV), sport and age band. A subgroup with
too few events is reported with its counts and a note; its metrics are left empty rather than
computed from a handful of events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.growth.evaluation import maturity_stratum

Metric = Callable[[Any, Any], float]


def add_strata(
    frame: pd.DataFrame, config: Mapping[str, Any], maturity_column: str
) -> pd.DataFrame:
    """Add ``age_band`` and ``maturity_stratum`` columns."""
    edges = list(config["age_band_edges_years"])
    labels = [f"{low:g}-{high:g}" for low, high in pairwise(edges)]
    out = frame.copy()
    out["age_band"] = pd.cut(
        out["age"], bins=edges, labels=labels, right=False, include_lowest=True
    ).astype("object")
    offset = out[maturity_column].to_numpy(dtype=np.float64)
    strata = maturity_stratum(np.nan_to_num(offset), float(config["maturity_band_years"]))
    labelled = pd.Series(strata, index=out.index, dtype="object")
    out["maturity_stratum"] = labelled.where(np.isfinite(offset), None)
    return out


def subgroup_table(
    predictions: pd.DataFrame,
    risk_column: str,
    by: Sequence[str],
    metrics: Mapping[str, Metric],
    *,
    min_events: int,
) -> pd.DataFrame:
    """Metrics of one model within each level of each column in ``by``.

    Returns ``subgroup, level, n_rows, n_athletes, n_events, <metrics...>, note``.
    """
    rows: list[dict[str, Any]] = []
    for column in by:
        if column not in predictions.columns:
            continue
        for level, part in predictions.dropna(subset=[column]).groupby(column, sort=True):
            outcome, risk = part["outcome"].to_numpy(), part[risk_column].to_numpy()
            events = int(outcome.sum())
            row: dict[str, Any] = {
                "subgroup": column,
                "level": str(level),
                "n_rows": len(part),
                "n_athletes": int(part["athlete_id"].nunique()),
                "n_events": events,
                "note": "",
            }
            if events < min_events or events == len(part):
                row["note"] = f"fewer than {min_events} events: metrics not computed"
                row.update(dict.fromkeys(metrics, float("nan")))
            else:
                row.update({name: metric(outcome, risk) for name, metric in metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)
