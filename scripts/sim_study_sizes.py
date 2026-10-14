"""Simulation study: performance of the tabular risk models against cohort size.

Generates ``n_replicates`` independent synthetic cohorts at each size in
``configs/eval/cohort_size_study.yaml``, evaluates every model with athlete-grouped
cross-validation, and writes to ``outputs/sim_study_sizes/<run_id>/``:

``replicates.csv`` (one row per cohort, model and metric), ``summary.csv`` (mean and spread
across cohorts), ``performance_by_size.png``, ``run.json``.

SYNTHETIC: this shows how much a cohort of a given size can establish under the simulation's
assumed effects. It is not a finding about real athletes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer
from matplotlib.figure import Figure

from youthtwin.eval import cohort_size
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata

PANELS = {
    "auc": "AUC (out-of-fold)",
    "calibration_slope": "Calibration slope (ideal: 1)",
    "correlation_with_true_risk": "Correlation with true risk",
}


def _figure(summary: pd.DataFrame, level: float) -> Figure:
    figure, axes = plt.subplots(1, len(PANELS), figsize=(5.0 * len(PANELS), 4.2))
    events = summary[summary["metric"] == "athletes_with_event"].set_index("n_athletes")["mean"]
    for axis, (metric, title) in zip(axes, PANELS.items(), strict=True):
        part = summary[summary["metric"] == metric]
        for model, rows in part.groupby("model", sort=False):
            rows = rows.sort_values("n_athletes")
            style = "--" if model == cohort_size.TRUE_RISK else "-"
            axis.plot(rows["n_athletes"], rows["mean"], style, marker="o", label=model)
            axis.fill_between(rows["n_athletes"], rows["low"], rows["high"], alpha=0.12)
        axis.set_xscale("log")
        axis.minorticks_off()
        axis.set_xticks(list(events.index))
        axis.set_xticklabels([f"{n}\n({events[n]:.0f})" for n in events.index])
        axis.set_xlabel("athletes (mean athletes with an injury)")
        axis.set_title(title)
    axes[0].axhline(0.5, color="grey", linewidth=0.8, linestyle=":")
    axes[1].axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=7)
    figure.suptitle(f"Mean and {level:.0%} range across replicate cohorts", fontsize=10)
    figure.tight_layout()
    return figure


def main(
    n_replicates: int | None = None,
    sizes: str | None = None,
    out_dir: Path | None = None,
) -> None:
    study = load_config("eval/cohort_size_study.yaml")
    if n_replicates is not None:
        study["n_replicates"] = n_replicates
    if sizes is not None:
        study["cohort_sizes"] = [int(s) for s in sizes.split(",")]
    cohort_config = load_config("sim/cohort.yaml")
    configs = {
        "person_period": load_config("features/person_period.yaml"),
        "hazard": load_config("models/hazard.yaml"),
        "baselines": load_config("models/baselines.yaml"),
        "mech": load_config("features/mech.yaml"),
    }
    run_config = {"cohort": cohort_config, "study": study, **configs}
    metadata = run_metadata(seed=int(cohort_config["seed"]), config=run_config, synthetic=True)
    run_id = f"simsizes-{config_hash(run_config)}"
    target = out_dir or repo_root() / "outputs" / "sim_study_sizes" / run_id
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()

    def progress(size: int, replicate: int) -> None:
        typer.echo(f"  {size} athletes, cohort {replicate + 1} [{time.time() - started:.0f} s]")

    replicates = cohort_size.run_study(cohort_config, study, configs, on_replicate=progress)
    summary = cohort_size.summarise(replicates, float(study["interval_level"]))
    replicates.to_csv(target / "replicates.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    record = {"run_id": run_id, **metadata, "seconds": round(time.time() - started, 1)}
    save_figure(
        _figure(summary, float(study["interval_level"])),
        target / "performance_by_size.png",
        record,
    )
    (target / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"\nSYNTHETIC cohort-size study {run_id} -> {target}")
    shown = summary[summary["metric"].isin([*PANELS, "athletes_with_event", "n_predictors"])]
    wide = shown.pivot_table(index=["metric", "model"], columns="n_athletes", values="mean")
    typer.echo(wide.round(3).to_string())


if __name__ == "__main__":
    typer.run(main)
