"""Assemble the evaluation tables from out-of-fold predictions.

Every table is tidy, is computed from the predictions table alone, and carries enough columns
to be traced (model, metric, counts). Confidence intervals come from the athlete-level
bootstrap. Nothing here is specific to synthetic data; the ``synthetic`` flag is carried from
the predictions.

Caveat on intervals: they are percentile intervals. ICI and ECE measure an absolute error, so
resampling noise inflates them and their bootstrap distribution sits above the point estimate;
the interval can then fail to contain the estimate. Read those intervals as a description of
resampling variability, and prefer the calibration slope and intercept for inference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.eval import bootstrap, calibration, dca, metrics, subgroups
from youthtwin.eval.predictions import risk_columns

Metric = Callable[[Any, Any], float]


def metric_functions(config: Mapping[str, Any]) -> dict[str, Metric]:
    """The configured metrics by name."""
    calibration_cfg = config["calibration"]
    available: dict[str, Metric] = {
        "time_dependent_auc_4wk": metrics.time_dependent_auc,
        "pr_auc": metrics.pr_auc,
        "brier_score": metrics.brier_score,
        "calibration_intercept": calibration.calibration_intercept,
        "calibration_slope": calibration.calibration_slope,
        "ici": lambda y, p: calibration.integrated_calibration_index(
            y, p, frac=float(calibration_cfg["loess_frac"])
        ),
        "ece": lambda y, p: calibration.expected_calibration_error(
            y, p, n_bins=int(calibration_cfg["n_bins"])
        ),
    }
    unknown = [name for name in config["metrics"] if name not in available]
    if unknown:
        raise KeyError(f"Unknown metrics in config: {unknown}")
    return {name: available[name] for name in config["metrics"]}


def _safe(metric: Metric) -> Metric:
    """A metric that returns NaN where it is undefined (e.g. a resample with no events)."""

    def wrapped(outcome: Any, risk: Any) -> float:
        try:
            return float(metric(outcome, risk))
        except (ValueError, np.linalg.LinAlgError):
            return float("nan")

    return wrapped


def performance_table(
    predictions: pd.DataFrame, config: Mapping[str, Any], *, seed: int
) -> pd.DataFrame:
    """Every metric for every model, with athlete-level bootstrap intervals."""
    settings = config["bootstrap"]
    outcome = predictions["outcome"].to_numpy()
    athletes = predictions["athlete_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    for model, column in risk_columns(predictions).items():
        risk = predictions[column].to_numpy()
        for name, metric in metric_functions(config).items():
            result = bootstrap.cluster_bootstrap(
                _safe(metric),
                outcome,
                risk,
                athletes,
                n_resamples=int(settings["n_resamples"]),
                level=float(settings["ci_level"]),
                seed=seed,
            )
            rows.append(
                {
                    "model": model,
                    "metric": name,
                    "estimate": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "ci_level": result.level,
                    "n_resamples": result.n_resamples,
                }
            )
        for threshold in config["thresholds"]:
            for name, value in metrics.sensitivity_specificity(outcome, risk, threshold).items():
                if name != "threshold":
                    rows.append(
                        {
                            "model": model,
                            "metric": f"{name}_at_{threshold:g}",
                            "estimate": value,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "ci_level": np.nan,
                            "n_resamples": 0,
                        }
                    )
    table = pd.DataFrame(rows)
    return _with_counts(table, predictions)


def difference_table(
    predictions: pd.DataFrame,
    reference: str,
    config: Mapping[str, Any],
    *,
    seed: int,
    metric_names: tuple[str, ...] = ("time_dependent_auc_4wk", "brier_score"),
) -> pd.DataFrame:
    """Each model minus ``reference``, paired on the same resampled athletes."""
    settings = config["bootstrap"]
    columns = risk_columns(predictions)
    if reference not in columns:
        raise KeyError(f"Reference model '{reference}' is not in the predictions")
    functions = metric_functions(config)
    outcome = predictions["outcome"].to_numpy()
    athletes = predictions["athlete_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    for model, column in columns.items():
        if model == reference:
            continue
        for name in metric_names:
            result = bootstrap.cluster_bootstrap_difference(
                _safe(functions[name]),
                outcome,
                predictions[column].to_numpy(),
                predictions[columns[reference]].to_numpy(),
                athletes,
                n_resamples=int(settings["n_resamples"]),
                level=float(settings["ci_level"]),
                seed=seed,
            )
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": name,
                    "difference": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "ci_level": result.level,
                    "interval_excludes_zero": bool(result.ci_low > 0 or result.ci_high < 0),
                }
            )
    return _with_counts(pd.DataFrame(rows), predictions)


def decision_curve_table(predictions: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    settings = config["decision_curve"]
    low, high = settings["threshold_range"]
    thresholds = np.linspace(float(low), float(high), int(settings["n_thresholds"]))
    risks = {m: predictions[c].to_numpy() for m, c in risk_columns(predictions).items()}
    return _with_counts(
        dca.decision_curve(predictions["outcome"].to_numpy(), risks, thresholds), predictions
    )


def calibration_table(predictions: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Binned calibration curve of every model."""
    n_bins = int(config["calibration"]["n_bins"])
    frames = []
    for model, column in risk_columns(predictions).items():
        bins = calibration.calibration_bins(predictions["outcome"], predictions[column], n_bins)
        frames.append(pd.DataFrame({"model": model, "bin": np.arange(len(bins["n"])), **bins}))
    return _with_counts(pd.concat(frames, ignore_index=True), predictions)


def subgroup_report(
    predictions: pd.DataFrame, config: Mapping[str, Any], maturity_column: str
) -> pd.DataFrame:
    """Primary metric and Brier score of every model in every subgroup."""
    settings = config["subgroups"]
    functions = metric_functions(config)
    chosen = {name: _safe(functions[name]) for name in (config["primary_metric"], "brier_score")}
    with_strata = subgroups.add_strata(predictions, settings, maturity_column)
    frames = []
    for model, column in risk_columns(predictions).items():
        table = subgroups.subgroup_table(
            with_strata, column, settings["by"], chosen, min_events=int(settings["min_events"])
        )
        table.insert(0, "model", model)
        frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    out["synthetic"] = bool(predictions["synthetic"].all())
    return out


def _with_counts(table: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["label"] = str(predictions["label"].iloc[0])
    table["n_rows"] = len(predictions)
    table["n_athletes"] = int(predictions["athlete_id"].nunique())
    table["n_events"] = int(predictions["outcome"].sum())
    table["synthetic"] = bool(predictions["synthetic"].all())
    return table
