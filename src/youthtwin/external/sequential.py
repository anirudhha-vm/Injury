"""Sequential growth-twin evaluation on real longitudinal data (M11).

A twin is a persistent per-child state that is updated as each measurement arrives and that
forecasts the next one. This module tests exactly that, out of sample:

1. Children are split into folds by child (per sex). The shared growth curve is learned from
   the training children's full records; each training child's Moore prior comes from one
   random season, as it would at deployment (``sitar_bayes.growth_data_from_measurements``).
2. Each held-out child enters at a random measurement age. After each of up to ``max_visits``
   visits the twin is redrawn against the fold's shared draws using only the visits so far
   (``sitar_bayes.draw_athlete_effects``), and scored on age at PHV against the reference from
   the full record, and on its forecast of the next measurement.

Comparators use the same visits: Moore 2015 and Mirwald 2002 (boys) for age at PHV, and
extrapolation of the last observed velocity for the forecast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.growth import sitar_bayes
from youthtwin.growth.maturity_equations import Sex, mirwald2002, moore2015
from youthtwin.growth.sitar_bayes import SharedDraws

MEASURES = ["stature", "sitting_height", "mass"]


def assign_folds(athletes: Sequence[str], n_folds: int, seed: int) -> dict[str, int]:
    """Each athlete to one fold, balanced, in a seeded random order."""
    order = np.random.default_rng(seed).permutation(sorted(athletes))
    return {str(a): k % n_folds for k, a in enumerate(order)}


def complete(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=MEASURES).sort_values(["athlete_id", "age"])


def training_data(
    frame: pd.DataFrame,
    athletes: Sequence[str],
    sex: Sex,
    config: Mapping[str, Any],
    windows: pd.DataFrame,
) -> sitar_bayes.GrowthData:
    """Full records of the training athletes; priors from each one's season in ``windows``."""
    low, high = (float(v) for v in config["train_age_range_years"])
    rows = complete(frame)
    rows = rows[rows["athlete_id"].isin(athletes) & rows["age"].between(low, high)]
    seasons = windows[windows["athlete_id"].isin(rows["athlete_id"].unique())]
    rows = rows[rows["athlete_id"].isin(seasons["athlete_id"].unique())]
    synthetic = bool(rows["synthetic"].any()) if "synthetic" in rows else False
    return sitar_bayes.growth_data_from_measurements(
        rows, sex, synthetic=synthetic, prior_frame=seasons
    )


def entry_sequences(
    frame: pd.DataFrame, athletes: Sequence[str], config: Mapping[str, Any], seed: int
) -> dict[str, pd.DataFrame]:
    """For each held-out athlete: visits from a random entry age, up to ``max_visits + 1``.

    The extra visit is the forecast target after the last update. Athletes with fewer than two
    visits after entry are dropped (nothing to update and forecast).
    """
    rng = np.random.default_rng(seed)
    low, high = (float(v) for v in config["entry_age_range_years"])
    keep = int(config["max_visits"]) + 1
    out: dict[str, pd.DataFrame] = {}
    rows = complete(frame)
    for athlete in sorted(athletes):
        record = rows[rows["athlete_id"] == athlete]
        starts = record.loc[record["age"].between(low, high), "age"].to_numpy()
        if starts.size == 0:
            continue
        entry = float(rng.choice(starts))
        visits = record[record["age"] >= entry - 1e-9].head(keep)
        if len(visits) >= 2:
            out[str(athlete)] = visits.reset_index(drop=True)
    return out


def _equation_aphv(visits: pd.DataFrame, sex: Sex) -> tuple[float, float]:
    """Moore and (boys) Mirwald APHV at the visit each places closest to PHV."""
    args = (visits["age"], visits["stature"], visits["sitting_height"])
    moore = moore2015(sex, *args)
    moore_aphv = float(moore.aphv[np.argmin(np.abs(moore.mo))])
    if sex != "male":
        return moore_aphv, float("nan")
    mirwald = mirwald2002(sex, *args, visits["mass"], synthetic=False)
    return moore_aphv, float(mirwald.aphv[np.argmin(np.abs(mirwald.mo))])


def score_sequences(
    shared: SharedDraws,
    sequences: Mapping[str, pd.DataFrame],
    reference: Mapping[str, float],
    sex: Sex,
    model_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    """One row per held-out athlete and number of visits seen so far."""
    level = float(config["interval_level"])
    tail = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for k in range(1, int(config["max_visits"]) + 1):
        seen = {a: v.head(k) for a, v in sequences.items() if len(v) >= k}
        if not seen:
            break
        frame = pd.concat([v.assign(athlete_id=a) for a, v in seen.items()], ignore_index=True)
        data = sitar_bayes.growth_data_from_measurements(frame, sex, synthetic=False)
        twin = sitar_bayes.draw_athlete_effects(shared, data, model_config, seed=seed + k)
        aphv = twin.aphv()
        for j, athlete in enumerate(twin.athlete_ids):
            visits = seen[athlete]
            draws = aphv[:, j]
            low, high = np.quantile(draws, [tail, 1.0 - tail])
            moore, mirwald = _equation_aphv(visits, sex)
            truth = float(reference.get(athlete, np.nan))
            last = visits.iloc[-1]
            row: dict[str, Any] = {
                "athlete_id": athlete,
                "sex": sex,
                "visits": k,
                "age": float(last["age"]),
                "years_from_phv": float(last["age"]) - truth,
                "reference_aphv": truth,
                "twin_aphv": float(draws.mean()),
                "twin_low": float(low),
                "twin_high": float(high),
                "twin_covered": bool(low <= truth <= high) if np.isfinite(truth) else np.nan,
                "moore_aphv": moore,
                "mirwald_aphv": mirwald,
            }
            future = sequences[athlete]
            if len(future) > k:  # forecast the next measurement
                target = future.iloc[k]
                predicted = twin.stature(athlete, float(target["age"]))
                predicted = predicted + rng.normal(0.0, 1.0, predicted.size) * twin.sigma
                f_low, f_high = np.quantile(predicted, [tail, 1.0 - tail])
                row.update(
                    {
                        "forecast_age": float(target["age"]),
                        "observed_stature": float(target["stature"]),
                        "twin_forecast": float(predicted.mean()),
                        "forecast_low": float(f_low),
                        "forecast_high": float(f_high),
                        "forecast_covered": bool(f_low <= target["stature"] <= f_high),
                    }
                )
                if k >= 2:  # carry the last observed velocity forward
                    previous = visits.iloc[-2]
                    velocity = (last["stature"] - previous["stature"]) / (
                        last["age"] - previous["age"]
                    )
                    row["naive_forecast"] = float(
                        last["stature"] + velocity * (target["age"] - last["age"])
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Errors, coverage and widths, grouped by ``by`` (e.g. visits, sex, distance band)."""

    def one(group: pd.DataFrame) -> dict[str, Any]:
        scored = group[np.isfinite(group["reference_aphv"])]
        out = {
            "n": len(scored),
            "twin_mae": (scored["twin_aphv"] - scored["reference_aphv"]).abs().mean(),
            "moore_mae": (scored["moore_aphv"] - scored["reference_aphv"]).abs().mean(),
            "mirwald_mae": (scored["mirwald_aphv"] - scored["reference_aphv"]).abs().mean(),
            "twin_bias": (scored["twin_aphv"] - scored["reference_aphv"]).mean(),
            "twin_coverage": scored["twin_covered"].astype(float).mean(),
            "twin_width": (scored["twin_high"] - scored["twin_low"]).mean(),
        }
        if "twin_forecast" in group:
            f = group.dropna(subset=["twin_forecast"])
            out["forecast_n"] = len(f)
            out["forecast_mae"] = (f["twin_forecast"] - f["observed_stature"]).abs().mean()
            out["forecast_coverage"] = f["forecast_covered"].astype(float).mean()
            out["forecast_width"] = (f["forecast_high"] - f["forecast_low"]).mean()
            if "naive_forecast" in f:
                naive = f.dropna(subset=["naive_forecast"])
                out["naive_n"] = len(naive)
                out["naive_mae"] = (
                    (naive["naive_forecast"] - naive["observed_stature"]).abs().mean()
                )
                out["twin_mae_same_rows"] = (
                    (naive["twin_forecast"] - naive["observed_stature"]).abs().mean()
                )
        return out

    rows = []
    for key, group in scores.groupby(list(by), sort=True):
        values = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, values, strict=True)), **one(group)})
    return pd.DataFrame(rows)
