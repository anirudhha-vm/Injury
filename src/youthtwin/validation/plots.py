"""Figures for the validation study. Save them with ``utils.plotting.save_figure`` so that
synthetic figures are watermarked and every figure gets a provenance file."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from youthtwin.validation.agreement import bland_altman

STRATUM_ORDER = ("pre_phv", "circa_phv", "post_phv")


def bland_altman_grid(pairs: pd.DataFrame, config: Mapping[str, Any], unit: str = "deg") -> Figure:
    """One Bland-Altman panel per variable: difference against pair mean, bias and limits."""
    variables = sorted(pairs["variable"].unique())
    columns = min(3, len(variables))
    rows = int(np.ceil(len(variables) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)
    for axis, variable in zip(axes.ravel(), variables, strict=False):
        part = pairs[pairs["variable"] == variable].dropna(subset=["markerless", "mocap"])
        mean = 0.5 * (part["markerless"] + part["mocap"])
        difference = part["markerless"] - part["mocap"]
        stats = bland_altman(
            part["markerless"],
            part["mocap"],
            level=float(config["loa_level"]),
            ci_level=float(config["ci_level"]),
        )
        axis.scatter(mean, difference, s=10, alpha=0.6)
        axis.axhline(stats.bias, color="black", linewidth=1)
        for limit in (stats.loa_low, stats.loa_high):
            axis.axhline(limit, color="black", linewidth=1, linestyle="--")
        axis.axhline(0.0, color="grey", linewidth=0.5)
        axis.set_title(f"{variable} (n = {stats.n})")
        axis.set_xlabel(f"Mean of methods ({unit})")
        axis.set_ylabel(f"Markerless - mocap ({unit})")
    for axis in axes.ravel()[len(variables) :]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def rmse_by_stratum(table: pd.DataFrame, config: Mapping[str, Any], unit: str = "deg") -> Figure:
    """RMSE of each variable per maturity stratum, with the pre-specified limit marked.

    ``table`` is an ``agreement_table`` computed with ``by=["maturity_group"]``.
    """
    pivot = table.pivot_table(index="variable", columns="maturity_group", values="rmse")
    pivot = pivot[[s for s in STRATUM_ORDER if s in pivot.columns]]
    figure, axis = plt.subplots(figsize=(1.6 * len(pivot) + 3.0, 4.0))
    width = 0.8 / max(1, pivot.shape[1])
    positions = np.arange(len(pivot))
    for k, stratum in enumerate(pivot.columns):
        axis.bar(positions + k * width, pivot[stratum], width=width, label=stratum)
    axis.axhline(
        float(config["rmse_threshold_deg"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="pre-specified limit",
    )
    axis.set_xticks(positions + width * (pivot.shape[1] - 1) / 2)
    axis.set_xticklabels(pivot.index, rotation=30, ha="right")
    axis.set_ylabel(f"RMSE ({unit})")
    axis.legend()
    figure.tight_layout()
    return figure


def curve_overlay(
    percent: np.ndarray[Any, Any],
    markerless: np.ndarray[Any, Any],
    mocap: np.ndarray[Any, Any],
    label: str,
    unit: str = "deg",
) -> Figure:
    """Mean and SD band of time-normalised curves from the two systems (rows = trials)."""
    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    for curves, name in ((markerless, "markerless"), (mocap, "mocap")):
        mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        axis.plot(percent, mean, label=name)
        axis.fill_between(percent, mean - sd, mean + sd, alpha=0.2)
    axis.set_xlabel("Task window (%)")
    axis.set_ylabel(f"{label} ({unit})")
    axis.legend()
    figure.tight_layout()
    return figure
