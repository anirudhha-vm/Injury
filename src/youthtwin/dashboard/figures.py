"""Plots for the research dashboard. Every figure from a synthetic bundle is watermarked."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from youthtwin.pose.skeletons import SIDES
from youthtwin.recommend.rules import OUTCOME_NAMES
from youthtwin.utils.plotting import watermark_synthetic

_CHAIN = ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")
_SIDE_COLOUR = {"left": "tab:blue", "right": "tab:orange"}


def _finish(figure: Figure, synthetic: bool) -> Figure:
    figure.tight_layout()
    return watermark_synthetic(figure) if synthetic else figure


def growth_figure(
    measurements: pd.DataFrame,
    curve: pd.DataFrame,
    aphv: pd.DataFrame,
    *,
    level: float,
    synthetic: bool,
) -> Figure:
    """Measured stature, fitted curve with posterior band, and the APHV interval."""
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.fill_between(
        curve["age"],
        curve["stature_low"],
        curve["stature_high"],
        alpha=0.25,
        label=f"{level:.0%} posterior band",
    )
    axis.plot(curve["age"], curve["stature_mean"], linewidth=1.2, label="fitted curve")
    axis.scatter(
        measurements["age"],
        measurements["stature"],
        s=18,
        color="black",
        zorder=3,
        label="measured",
    )
    if not aphv.empty:
        row = aphv.iloc[0]
        axis.axvspan(
            row["aphv_low"],
            row["aphv_high"],
            color="tab:green",
            alpha=0.15,
            label=f"age at PHV, {level:.0%} interval",
        )
        axis.axvline(row["aphv_mean"], color="tab:green", linewidth=1)
    axis.set_xlim(curve["age"].min(), curve["age"].max())
    axis.set_xlabel("Age (years)")
    axis.set_ylabel("Stature (cm)")
    axis.legend(fontsize=7, loc="lower right")
    return _finish(figure, synthetic)


def velocity_figure(curve: pd.DataFrame, *, level: float, synthetic: bool) -> Figure:
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    axis.fill_between(curve["age"], curve["velocity_low"], curve["velocity_high"], alpha=0.25)
    axis.plot(curve["age"], curve["velocity_mean"], linewidth=1.2)
    axis.set_xlabel("Age (years)")
    axis.set_ylabel(f"Height velocity (cm/year), {level:.0%} band")
    return _finish(figure, synthetic)


def risk_figure(
    risk: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    level: float,
    horizon_weeks: int,
    synthetic: bool,
) -> Figure:
    """Out-of-fold predicted risk over time per injury type, with observed injuries marked."""
    labels = list(risk["label"].unique())
    figure, axes = plt.subplots(
        len(labels), 1, figsize=(7.0, 2.2 * len(labels)), sharex=True, squeeze=False
    )
    onsets = pd.to_datetime(injuries["onset_date"]) if not injuries.empty else pd.Series([])
    for axis, label in zip(axes[:, 0], labels, strict=True):
        part = risk[risk["label"] == label].sort_values("week_start")
        weeks = pd.to_datetime(part["week_start"])
        axis.fill_between(weeks, part["risk_low"], part["risk_high"], alpha=0.25)
        axis.plot(weeks, part["risk"], linewidth=1.2)
        for onset in onsets:
            axis.axvline(onset, color="tab:red", linewidth=0.8, linestyle=":")
        axis.set_ylabel(f"{horizon_weeks}-week risk", fontsize=8)
        axis.set_title(OUTCOME_NAMES.get(label, label), fontsize=9, loc="left")
        axis.set_ylim(bottom=0)
    axes[-1, 0].set_xlabel(
        f"Week (band: {level:.0%} from growth uncertainty; dotted: recorded injury)"
    )
    figure.autofmt_xdate()
    return _finish(figure, synthetic)


def contribution_figure(contributions: pd.DataFrame, *, top: int, synthetic: bool) -> Figure:
    """Largest SHAP contributions (log-odds) to the latest prediction for one outcome."""
    order = contributions.reindex(
        contributions["shap_log_odds"].abs().sort_values(ascending=False).index
    ).head(top)[::-1]
    figure, axis = plt.subplots(figsize=(7.0, 0.4 * len(order) + 1.2))
    colours = ["tab:red" if v > 0 else "tab:blue" for v in order["shap_log_odds"]]
    axis.barh(order["input"], order["shap_log_odds"], color=colours)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Contribution to the model's log-odds (red raises, blue lowers)")
    axis.tick_params(axis="y", labelsize=7)
    return _finish(figure, synthetic)


def skeleton_figure(keypoints: pd.DataFrame, frame: int, *, synthetic: bool) -> Figure:
    """One frame of the side-view stick figure, drawn from keypoints only."""
    now = keypoints[keypoints["frame"] == frame].set_index("joint")
    figure, axis = plt.subplots(figsize=(3.6, 5.0))
    for side in SIDES:
        points = np.array(
            [[now.loc[f"{side}_{j}", "x"], now.loc[f"{side}_{j}", "y"]] for j in _CHAIN[:4]]
        )
        axis.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=4,
            color=_SIDE_COLOUR[side],
            label=side,
        )
        foot = np.array(
            [
                [now.loc[f"{side}_{j}", "x"], now.loc[f"{side}_{j}", "y"]]
                for j in ("heel", "ankle", "foot_index")
            ]
        )
        axis.plot(foot[:, 0], foot[:, 1], color=_SIDE_COLOUR[side])
    axis.set_xlim(keypoints["x"].min() - 40, keypoints["x"].max() + 40)
    axis.set_ylim(keypoints["y"].max() + 40, keypoints["y"].min() - 40)  # image y is down
    axis.set_aspect("equal")
    axis.axis("off")
    axis.legend(fontsize=7, loc="upper right")
    time_s: Any = now["time_s"].iloc[0]
    axis.set_title(f"t = {float(time_s):.2f} s", fontsize=9)
    return _finish(figure, synthetic)
