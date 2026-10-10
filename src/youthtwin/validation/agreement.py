"""Agreement between two measurement methods (Paper A: markerless vs marker-based mocap).

Differences are always ``method - reference`` (markerless minus mocap), so a positive bias means
the markerless value is larger.

- **Bland-Altman** (Bland & Altman 1986, 1999): bias, SD of the differences, limits of agreement
  ``bias +/- z * SD`` with confidence intervals for the bias and for each limit
  (``SE(limit) = SD * sqrt(1/n + z^2 / (2 (n - 1)))``), and proportional bias as the slope of
  the difference on the pair mean. Note that this slope is not zero merely because the two
  methods have different error variances, even without any true proportional bias; interpret
  it alongside the spread of each method.
- **ICC** for single measurements from a two-way ANOVA with ``k = 2`` methods:
  ICC(2,1) absolute agreement, two-way random; ICC(3,1) consistency, two-way mixed (Shrout &
  Fleiss 1979), with their F-based confidence intervals.
- **RMSE** and mean absolute error of the differences.

Each statistic answers a different question (systematic offset, spread, relative reliability),
so the table reports all of them; an ICC alone can look excellent when a large constant bias is
present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

_K = 2  # two methods
MIN_PAIRS = 3


def _paired(method: ArrayLike, reference: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(method, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("method and reference must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < MIN_PAIRS:
        raise ValueError(f"Agreement needs at least {MIN_PAIRS} complete pairs")
    return a[keep], b[keep]


def rmse(method: ArrayLike, reference: ArrayLike) -> float:
    a, b = _paired(method, reference)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_absolute_error(method: ArrayLike, reference: ArrayLike) -> float:
    a, b = _paired(method, reference)
    return float(np.mean(np.abs(a - b)))


@dataclass(frozen=True)
class BlandAltman:
    n: int
    bias: float
    bias_ci_low: float
    bias_ci_high: float
    sd: float
    loa_low: float
    loa_high: float
    loa_low_ci: tuple[float, float]
    loa_high_ci: tuple[float, float]
    proportional_slope: float  # change in difference per unit of the pair mean
    proportional_p: float


def bland_altman(
    method: ArrayLike, reference: ArrayLike, *, level: float = 0.95, ci_level: float = 0.95
) -> BlandAltman:
    """Bland-Altman statistics for ``method - reference``."""
    a, b = _paired(method, reference)
    difference, mean = a - b, 0.5 * (a + b)
    n = difference.size
    bias, sd = float(np.mean(difference)), float(np.std(difference, ddof=1))
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    t = float(stats.t.ppf(0.5 + ci_level / 2.0, n - 1))
    low, high = bias - z * sd, bias + z * sd
    se_bias = sd / np.sqrt(n)
    se_limit = sd * np.sqrt(1.0 / n + z**2 / (2.0 * (n - 1)))
    if np.ptp(mean) > 0:
        fit = stats.linregress(mean, difference)
        slope, p_value = float(fit.slope), float(fit.pvalue)
    else:
        slope, p_value = float("nan"), float("nan")
    return BlandAltman(
        n=n,
        bias=bias,
        bias_ci_low=bias - t * se_bias,
        bias_ci_high=bias + t * se_bias,
        sd=sd,
        loa_low=low,
        loa_high=high,
        loa_low_ci=(low - t * se_limit, low + t * se_limit),
        loa_high_ci=(high - t * se_limit, high + t * se_limit),
        proportional_slope=slope,
        proportional_p=p_value,
    )


@dataclass(frozen=True)
class Icc:
    kind: str  # "ICC(2,1)" or "ICC(3,1)"
    value: float
    ci_low: float
    ci_high: float
    n: int


def _anova(a: FloatArray, b: FloatArray) -> tuple[float, float, float, int]:
    """Mean squares for subjects (rows), methods (columns) and error; and n."""
    data = np.column_stack([a, b])
    n = data.shape[0]
    grand = data.mean()
    ss_rows = _K * np.sum((data.mean(axis=1) - grand) ** 2)
    ss_columns = n * np.sum((data.mean(axis=0) - grand) ** 2)
    ss_error = np.sum((data - grand) ** 2) - ss_rows - ss_columns
    return (
        float(ss_rows / (n - 1)),
        float(ss_columns / (_K - 1)),
        float(ss_error / ((n - 1) * (_K - 1))),
        n,
    )


def icc_consistency(method: ArrayLike, reference: ArrayLike, *, ci_level: float = 0.95) -> Icc:
    """ICC(3,1): two-way mixed, consistency, single measurement. Ignores a constant offset."""
    msr, _, mse, n = _anova(*_paired(method, reference))
    value = (msr - mse) / (msr + (_K - 1) * mse)
    alpha = 1.0 - ci_level
    df1, df2 = n - 1, (n - 1) * (_K - 1)
    f = msr / mse
    f_low = f / stats.f.ppf(1 - alpha / 2, df1, df2)
    f_high = f * stats.f.ppf(1 - alpha / 2, df2, df1)
    return Icc(
        "ICC(3,1)",
        float(value),
        float((f_low - 1) / (f_low + _K - 1)),
        float((f_high - 1) / (f_high + _K - 1)),
        n,
    )


def icc_agreement(method: ArrayLike, reference: ArrayLike, *, ci_level: float = 0.95) -> Icc:
    """ICC(2,1): two-way random, absolute agreement, single measurement. Penalises offsets."""
    msr, msc, mse, n = _anova(*_paired(method, reference))
    k = _K
    value = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    alpha = 1.0 - ci_level
    f_columns = msc / mse
    numerator = (
        (k - 1) * (n - 1) * (k * value * f_columns + n * (1 + (k - 1) * value) - k * value) ** 2
    )
    denominator = (n - 1) * k**2 * value**2 * f_columns**2 + (
        n * (1 + (k - 1) * value) - k * value
    ) ** 2
    v = numerator / denominator
    f1 = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f2 = stats.f.ppf(1 - alpha / 2, v, n - 1)
    low = n * (msr - f1 * mse) / (f1 * (k * msc + (k * n - k - n) * mse) + n * msr)
    high = n * (f2 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f2 * msr)
    return Icc("ICC(2,1)", float(value), float(low), float(high), n)


def agreement_summary(
    method: ArrayLike, reference: ArrayLike, config: Mapping[str, Any]
) -> dict[str, Any]:
    """All agreement statistics for one set of pairs, as a flat record."""
    ci_level = float(config["ci_level"])
    ba = bland_altman(method, reference, level=float(config["loa_level"]), ci_level=ci_level)
    absolute = icc_agreement(method, reference, ci_level=ci_level)
    consistency = icc_consistency(method, reference, ci_level=ci_level)
    record = asdict(ba)
    for limit in ("loa_low_ci", "loa_high_ci"):
        lower, upper = record.pop(limit)
        record[f"{limit}_low"], record[f"{limit}_high"] = lower, upper
    record.update(
        {
            "rmse": rmse(method, reference),
            "mae": mean_absolute_error(method, reference),
            "icc21": absolute.value,
            "icc21_ci_low": absolute.ci_low,
            "icc21_ci_high": absolute.ci_high,
            "icc31": consistency.value,
            "icc31_ci_low": consistency.ci_low,
            "icc31_ci_high": consistency.ci_high,
        }
    )
    return record


def agreement_table(
    pairs: pd.DataFrame, config: Mapping[str, Any], by: Sequence[str] = ()
) -> pd.DataFrame:
    """One row of agreement statistics per variable x task x event (x each column in ``by``).

    ``pairs`` has one row per athlete and measurement with columns ``variable, task, event,
    markerless, mocap`` plus any stratification columns (e.g. ``maturity_group``, ``sex``).
    Groups with too few complete pairs are kept in the table with the reason, not dropped.
    ``meets_threshold`` compares RMSE with the pre-specified limit for that variable's unit.
    """
    keys = ["variable", "task", "event", *by]
    absent = [c for c in (*keys, "markerless", "mocap") if c not in pairs.columns]
    if absent:
        raise ValueError(f"pairs is missing columns: {absent}")
    threshold = float(config["rmse_threshold_deg"])
    rows: list[dict[str, Any]] = []
    for group_key, group in pairs.groupby(keys, sort=True):
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        try:
            row.update(agreement_summary(group["markerless"], group["mocap"], config))
            row["meets_threshold"] = bool(row["rmse"] < threshold)
            row["note"] = ""
        except ValueError as error:
            row.update({"n": int(group[["markerless", "mocap"]].dropna().shape[0])})
            row["note"] = str(error)
        rows.append(row)
    table = pd.DataFrame(rows)
    if "synthetic" in pairs.columns:
        table["synthetic"] = bool(pairs["synthetic"].all())
    return table
