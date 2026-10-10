"""True event probabilities of a synthetic cohort, for scoring models in simulation studies.

These come from the generator's truth tables and are used **only to evaluate** predictions on
synthetic data. They are never model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.io.schema import NON_CONTACT_GROUPS
from youthtwin.sim.cohort import SyntheticCohort

FloatArray = NDArray[np.float64]
_LABEL_PREFIX = "y_"
_ALL = "all_non_contact"


def true_window_risk(
    cohort: SyntheticCohort, frame: pd.DataFrame, label: str, horizon_weeks: int
) -> FloatArray:
    """True probability of an onset of the labelled type in the ``horizon`` weeks after each row.

    Computed from the true weekly hazards as ``1 - prod(1 - h_week)`` over the following weeks.
    It ignores that an earlier injury of another type would remove the athlete from the risk
    set inside the window, so it is very slightly higher than the exact probability.
    """
    group = label.removeprefix(_LABEL_PREFIX)
    truth = cohort.truth["athlete_weeks_truth"]
    if group == _ALL:
        survive = np.prod(
            [1.0 - truth[f"hazard_{g}"].to_numpy() for g in NON_CONTACT_GROUPS], axis=0
        )
        weekly = 1.0 - survive
    else:
        weekly = truth[f"hazard_{group}"].to_numpy()
    table = pd.DataFrame(
        {
            "athlete_id": truth["athlete_id"].to_numpy(),
            "week_index": truth["week_index"].to_numpy(),
            "log_survive": np.log1p(-weekly),
        }
    ).sort_values(["athlete_id", "week_index"])
    # Sum of log-survival over weeks w+1 .. w+horizon, per athlete.
    cumulative = table.groupby("athlete_id")["log_survive"].cumsum()
    table["cumulative"] = cumulative
    ahead = table.groupby("athlete_id")["cumulative"].shift(-horizon_weeks)
    table["risk"] = 1.0 - np.exp(ahead - table["cumulative"])
    starts = pd.to_datetime(truth["week_start"]).astype("datetime64[ns]")
    table["week_start"] = starts.loc[table.index]
    keyed = table.set_index(["athlete_id", "week_start"])["risk"]
    rows = pd.MultiIndex.from_arrays(
        [frame["athlete_id"], pd.to_datetime(frame["week_start"]).astype("datetime64[ns]")]
    )
    return keyed.reindex(rows).to_numpy(dtype=np.float64)
