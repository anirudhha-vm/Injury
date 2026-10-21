"""Regenerate every figure and table for the paper from run outputs (``make paper-figs``).

Reads one evaluation run (``scripts/evaluate.py``) and, if present, one ablation run
(``scripts/ablation.py``) and one growth-twin simulation study, and writes figures to
``paper/figures`` and tables to ``paper/tables``. Every file gets a ``.provenance.json`` sidecar
naming the run it came from; figures from synthetic runs are watermarked SYNTHETIC. By default
the most recent run of each kind under ``outputs/`` is used.

Nothing is typed by hand: a number appears in ``paper/`` only because a run produced it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import PROVENANCE_SUFFIX, save_figure

REFERENCES = ("treat_all", "treat_none")
# Further runs copied as they are: (outputs/<kind>, producing script, tables, figures).
# Real-data runs (Berkeley growth, runners' injuries) are not watermarked; synthetic ones were
# watermarked when their figures were saved.
EXTRA_RUNS: tuple[tuple[str, str, dict[str, str], dict[str, str]], ...] = (
    (
        "real_growth_berkeley",
        "scripts/real_growth_berkeley.py",
        {"summary.csv": "real_growth_twin.csv", "reference.csv": "real_growth_reference.csv"},
        {"aphv_real.png": "real_growth_twin_aphv.png"},
    ),
    (
        "real_risk_runners",
        "scripts/real_risk_runners.py",
        {
            "performance.csv": "real_risk_performance.csv",
            "differences.csv": "real_risk_differences.csv",
            "leakage_inflation.csv": "real_risk_leakage_inflation.csv",
        },
        {"calibration.png": "real_risk_calibration.png"},
    ),
    (
        "sim_study_validation",
        "scripts/sim_study_validation.py",
        {
            "agreement_overall.csv": "validity_agreement.csv",
            "agreement_by_maturity_group.csv": "validity_agreement_by_maturity.csv",
            "precision.csv": "validity_precision.csv",
        },
        {
            "bland_altman.png": "validity_bland_altman.png",
            "rmse_by_maturity.png": "validity_rmse_by_maturity.png",
        },
    ),
)


def _latest(kind: str) -> Path | None:
    root = repo_root() / "outputs" / kind
    runs = [p for p in root.glob("*") if (p / "run.json").is_file()] if root.is_dir() else []
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def _record(run_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {k: record.get(k) for k in ("run_id", "synthetic", "seed", "config_hash", "git_commit")}


def _write_table(table: pd.DataFrame, path: Path, provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    sidecar = path.with_name(path.name + PROVENANCE_SUFFIX)
    sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _calibration_figure(calibration: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    limit = float(max(calibration["predicted"].max(), calibration["observed"].max())) * 1.1
    axis.plot([0, limit], [0, limit], color="black", linewidth=1, label="ideal")
    for model, part in calibration.groupby("model"):
        axis.plot(part["predicted"], part["observed"], marker="o", markersize=3, label=model)
    axis.set_xlabel("Predicted 4-week risk (bin mean)")
    axis.set_ylabel("Observed proportion")
    axis.set_title("Calibration (out-of-fold, equal-size bins)")
    axis.legend(fontsize=7)
    return figure


def _decision_curve_figure(curve: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.0, 4.5))
    for strategy, part in curve.groupby("strategy"):
        style = {"color": "black", "linestyle": "--"} if strategy in REFERENCES else {}
        axis.plot(part["threshold"], part["net_benefit"], label=strategy, linewidth=1.2, **style)
    top = float(curve.loc[~curve["strategy"].isin(REFERENCES), "net_benefit"].max())
    axis.set_ylim(min(-0.005, -abs(top)), max(0.01, top * 1.3))
    axis.set_xlabel("Threshold probability")
    axis.set_ylabel("Net benefit")
    axis.set_title("Decision curves (out-of-fold)")
    axis.legend(fontsize=7)
    return figure


def _performance_figure(performance: pd.DataFrame, metric: str) -> plt.Figure:
    part = performance[performance["metric"] == metric].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(6.0, 0.6 * len(part) + 1.5))
    errors = [part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]]
    axis.errorbar(part["estimate"], range(len(part)), xerr=errors, fmt="o", capsize=3)
    axis.set_yticks(range(len(part)))
    axis.set_yticklabels(part["model"])
    if "auc" in metric:
        axis.axvline(0.5, color="grey", linewidth=0.8, linestyle=":")
    axis.set_xlabel(f"{metric} (athlete-level bootstrap interval)")
    figure.tight_layout()
    return figure


def main(
    evaluation_dir: Path | None = None,
    ablation_dir: Path | None = None,
    growth_dir: Path | None = None,
    paper_dir: Path | None = None,
) -> None:
    paper = paper_dir or repo_root() / "paper"
    figures, tables = paper / "figures", paper / "tables"
    primary = str(load_config("eval/metrics.yaml")["primary_metric"])
    written: list[str] = []

    evaluation = evaluation_dir or _latest("evaluation")
    if evaluation is None:
        raise typer.BadParameter("No evaluation run found; run scripts/evaluate.py first")
    provenance = {"source": "scripts/evaluate.py", **_record(evaluation)}
    loaded = {
        name: pd.read_csv(evaluation / f"{name}.csv")
        for name in (
            "performance",
            "differences",
            "decision_curve",
            "calibration",
            "subgroups",
            "shap_importance",
        )
    }
    for name, table in loaded.items():
        _write_table(table, tables / f"{name}.csv", provenance)
        written.append(f"tables/{name}.csv")
    plots = {
        "calibration.png": _calibration_figure(loaded["calibration"]),
        "decision_curve.png": _decision_curve_figure(loaded["decision_curve"]),
        "discrimination.png": _performance_figure(loaded["performance"], primary),
    }
    for filename, figure in plots.items():
        save_figure(figure, figures / filename, provenance)
        written.append(f"figures/{filename}")

    ablation = ablation_dir or _latest("ablation")
    if ablation is not None:
        source = {"source": "scripts/ablation.py", **_record(ablation)}
        _write_table(pd.read_csv(ablation / "results.csv"), tables / "ablation.csv", source)
        written.append("tables/ablation.csv")
        for name in ("performance", "differences"):  # absent in runs made before D-109
            if (ablation / f"{name}.csv").is_file():
                table = pd.read_csv(ablation / f"{name}.csv")
                _write_table(table, tables / f"ablation_{name}.csv", source)
                written.append(f"tables/ablation_{name}.csv")

    growth = growth_dir or _latest("sim_study_growth")
    if growth is not None:
        source = {"source": "scripts/sim_study_growth.py", **_record(growth)}
        _write_table(pd.read_csv(growth / "summary.csv"), tables / "growth_twin.csv", source)
        shutil.copyfile(growth / "aphv_recovery.png", figures / "growth_twin_aphv.png")
        sidecar = figures / f"growth_twin_aphv.png{PROVENANCE_SUFFIX}"
        sidecar.write_text(json.dumps(source, indent=2), encoding="utf-8")
        written += ["tables/growth_twin.csv", "figures/growth_twin_aphv.png"]

    sizes = _latest("sim_study_sizes")
    if sizes is not None:
        source = {"source": "scripts/sim_study_sizes.py", **_record(sizes)}
        _write_table(pd.read_csv(sizes / "summary.csv"), tables / "cohort_size.csv", source)
        shutil.copyfile(sizes / "performance_by_size.png", figures / "cohort_size.png")
        sidecar = figures / f"cohort_size.png{PROVENANCE_SUFFIX}"
        sidecar.write_text(json.dumps(source, indent=2), encoding="utf-8")
        written += ["tables/cohort_size.csv", "figures/cohort_size.png"]

    for kind, source_script, tables_map, figures_map in EXTRA_RUNS:
        run = _latest(kind)
        if run is None:
            continue
        source = {"source": source_script, **_record(run)}
        for name, target_name in tables_map.items():
            if (run / name).is_file():
                _write_table(pd.read_csv(run / name), tables / target_name, source)
                written.append(f"tables/{target_name}")
        for name, target_name in figures_map.items():
            if (run / name).is_file():
                figures.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run / name, figures / target_name)
                sidecar = figures / f"{target_name}{PROVENANCE_SUFFIX}"
                sidecar.write_text(json.dumps(source, indent=2), encoding="utf-8")
                written.append(f"figures/{target_name}")

    plt.close("all")
    label = "SYNTHETIC " if provenance.get("synthetic") else ""
    typer.echo(f"{label}paper outputs written to {paper}:")
    for item in written:
        typer.echo(f"  {item}")


if __name__ == "__main__":
    typer.run(main)
