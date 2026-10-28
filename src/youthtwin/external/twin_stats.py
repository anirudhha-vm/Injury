"""Inference and secondary analyses for the sequential growth twin (M11).

Everything here works on the per-visit scores of ``scripts/real_twin_sequential.py``:

- child-level (cluster) bootstrap intervals for the headline comparisons, because a child
  contributes up to six correlated rows;
- error and bias by maturity timing (early / average / late maturers), the known failure mode
  of maturity-offset equations (Koziel & Malina 2018);
- calibration of the growth-spurt probability;
- sensitivity to the reference: the smoothing-spline age at PHV instead of Preece-Baines;
- conditional coverage: whether calibrated intervals hold within maturity-timing groups and
  within thirds of the twin's own interval width (adaptive versus constant-width intervals).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.eval import calibration, metrics

Statistic = Callable[[pd.DataFrame], float]
TIMING_GROUPS = ("early", "average", "late")


def mae_difference(estimate: str, comparator: str, truth: str = "reference_aphv") -> Statistic:
    """MAE of ``estimate`` minus MAE of ``comparator`` (negative: ``estimate`` is better)."""

    def statistic(frame: pd.DataFrame) -> float:
        rows = frame.dropna(subset=[estimate, comparator, truth])
        return float(
            (rows[estimate] - rows[truth]).abs().mean()
            - (rows[comparator] - rows[truth]).abs().mean()
        )

    return statistic


def mae(estimate: str, truth: str = "reference_aphv") -> Statistic:
    def statistic(frame: pd.DataFrame) -> float:
        rows = frame.dropna(subset=[estimate, truth])
        return float((rows[estimate] - rows[truth]).abs().mean())

    return statistic


def auc_difference(score: str, comparator_score: str, label: str = "circa_now") -> Statistic:
    def statistic(frame: pd.DataFrame) -> float:
        rows = frame.dropna(subset=[label])
        y = rows[label].astype(float).to_numpy()
        if np.unique(y).size < 2:
            return float("nan")
        return metrics.auc(y, rows[score].to_numpy()) - metrics.auc(
            y, rows[comparator_score].to_numpy()
        )

    return statistic


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Statistic,
    *,
    n_resamples: int,
    level: float,
    seed: int,
    cluster: str = "athlete_id",
) -> dict[str, float]:
    """Percentile interval of ``statistic``, resampling whole children with replacement."""
    groups = {str(key): part for key, part in frame.groupby(cluster, sort=True)}
    keys = np.array(sorted(groups))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        chosen = rng.choice(keys, size=keys.size, replace=True)
        values.append(statistic(pd.concat([groups[k] for k in chosen], ignore_index=True)))
    finite = np.asarray([v for v in values if np.isfinite(v)])
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail]) if finite.size else (np.nan, np.nan)
    return {
        "estimate": statistic(frame),
        "ci_low": float(low),
        "ci_high": float(high),
        "level": level,
        "n_clusters": int(keys.size),
        "n_resamples": int(finite.size),
    }


def timing_groups(reference: pd.DataFrame, width_sd: float) -> pd.Series:
    """Early / average / late maturers: reference APHV more than ``width_sd`` SDs from the
    mean of the same sex (computed over the children with a reference)."""
    out = pd.Series("average", index=reference.index, dtype=object)
    for _, part in reference.groupby("sex"):
        z = (part["reference_aphv"] - part["reference_aphv"].mean()) / part["reference_aphv"].std()
        out.loc[part.index[z < -width_sd]] = "early"
        out.loc[part.index[z > width_sd]] = "late"
    return out


def by_timing_group(scores: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """Error and signed bias of the twin and Moore by maturity timing group."""
    frame = scores.assign(timing=scores["athlete_id"].map(groups)).dropna(
        subset=["timing", "reference_aphv"]
    )
    rows = []
    for group in TIMING_GROUPS:
        part = frame[frame["timing"] == group]
        if part.empty:
            continue
        rows.append(
            {
                "timing": group,
                "n_children": int(part["athlete_id"].nunique()),
                "n": len(part),
                "twin_mae": float((part["twin_aphv"] - part["reference_aphv"]).abs().mean()),
                "moore_mae": float((part["moore_aphv"] - part["reference_aphv"]).abs().mean()),
                "twin_bias": float((part["twin_aphv"] - part["reference_aphv"]).mean()),
                "moore_bias": float((part["moore_aphv"] - part["reference_aphv"]).mean()),
                "twin_conf_coverage": float(part["twin_conf_covered"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def coverage_difference(covered: str, comparator: str) -> Statistic:
    """Coverage of one interval minus another, on rows where both are scored."""

    def statistic(frame: pd.DataFrame) -> float:
        rows = frame.dropna(subset=[covered, comparator])
        return float(rows[covered].astype(float).mean() - rows[comparator].astype(float).mean())

    return statistic


WIDTH_THIRDS = ("narrow", "middle", "wide")


def width_thirds(
    scores: pd.DataFrame, width: str, by: Sequence[str] = ("sex", "visits")
) -> pd.Series:
    """Thirds of the twin's interval width within each ``by`` cell (ties broken by order)."""
    ranks = scores.groupby(list(by))[width].rank(method="first", pct=True)
    return pd.cut(ranks, [0.0, 1 / 3, 2 / 3, 1.0], labels=list(WIDTH_THIRDS)).astype(object)


def circa_calibration(scores: pd.DataFrame) -> dict[str, float]:
    """Calibration intercept and slope, and Brier score, of the growth-spurt probability."""
    rows = scores.dropna(subset=["circa_now"])
    y = rows["circa_now"].astype(float).to_numpy()
    p = np.clip(rows["twin_p_circa"].to_numpy(dtype=float), 0.005, 0.995)  # 200 draws: avoid 0/1
    return {
        "calibration_intercept": calibration.calibration_intercept(y, p),
        "calibration_slope": calibration.calibration_slope(y, p),
        "brier": metrics.brier_score(y, p),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
    }


def headline_intervals(
    scores: pd.DataFrame, *, n_resamples: int, level: float, seed: int, visits: Sequence[int]
) -> pd.DataFrame:
    """Bootstrap intervals for the comparisons the paper states."""
    specs: list[tuple[str, Any, Statistic]] = [
        ("twin_mae", None, mae("twin_aphv")),
        ("moore_mae", None, mae("moore_aphv")),
        ("twin_minus_moore_mae", None, mae_difference("twin_aphv", "moore_aphv")),
        (
            "twin_minus_moore_mae_full",
            None,
            mae_difference("twin_aphv", "moore_aphv", "full_info_aphv"),
        ),
    ]
    boys = scores[scores["sex"] == "male"]
    rows = []
    for name, _, statistic in specs:
        rows.append(
            {
                "comparison": name,
                "subset": "all",
                **cluster_bootstrap(
                    scores, statistic, n_resamples=n_resamples, level=level, seed=seed
                ),
            }
        )
    rows.append(
        {
            "comparison": "twin_minus_mirwald_mae",
            "subset": "boys",
            **cluster_bootstrap(
                boys,
                mae_difference("twin_aphv", "mirwald_aphv"),
                n_resamples=n_resamples,
                level=level,
                seed=seed,
            ),
        }
    )
    for k in visits:
        part = scores[scores["visits"] == k]
        rows.append(
            {
                "comparison": "twin_minus_moore_mae",
                "subset": f"visit {k}",
                **cluster_bootstrap(
                    part,
                    mae_difference("twin_aphv", "moore_aphv"),
                    n_resamples=n_resamples,
                    level=level,
                    seed=seed,
                ),
            }
        )
    with_moore_score = scores.assign(moore_score=-scores["moore_offset_now"].abs())
    rows.append(
        {
            "comparison": "twin_minus_moore_circa_auc",
            "subset": "all",
            **cluster_bootstrap(
                with_moore_score,
                auc_difference("twin_p_circa", "moore_score"),
                n_resamples=n_resamples,
                level=level,
                seed=seed,
            ),
        }
    )
    return pd.DataFrame(rows)
