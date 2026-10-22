"""Run the ablation grid on the synthetic cohort and write one tidy results table.

Generates the cohort from its seed, simulates and processes each session's trials, builds the
athlete-week table (with growth-twin features if ``--posterior-dir`` points at saved posteriors
from ``scripts/sim_study_growth.py`` or ``scripts/fit_growth.py``) and runs every configuration
in ``configs/eval/ablations.yaml``. Output in ``outputs/ablation/<run_id>/``: ``results.csv``,
the out-of-fold ``predictions.parquet``, and ``performance.csv`` / ``differences.csv`` with
athlete-level bootstrap intervals, plus a provenance record. ``--report-from <run dir>``
recomputes the two interval tables from saved predictions without refitting anything.
SYNTHETIC: these numbers are a pipeline check, not findings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer

from youthtwin.eval.ablation import ablation_report, run_ablation
from youthtwin.features.mech_load import add_mech_proxies
from youthtwin.features.person_period import build_person_period
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models import dataset
from youthtwin.sim.cohort import generate_cohort
from youthtwin.sim.truth import true_window_risk
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata


def _write_report(
    predictions: pd.DataFrame, target: Path, seed: int, grid: dict[str, Any] | None = None
) -> None:
    grid = grid or load_config("eval/ablations.yaml")
    metrics_config = load_config("eval/metrics.yaml")
    metrics_config["metrics"] = list(grid.get("report_metrics", metrics_config["metrics"]))
    tables = ablation_report(
        predictions,
        grid,
        metrics_config,
        seed=seed,
    )
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False)
    shown = tables["differences"][
        ["model", "reference", "metric", "difference", "ci_low", "ci_high"]
    ]
    typer.echo("\nPaired differences (model minus reference), athlete-level bootstrap:")
    typer.echo(shown.round(4).to_string(index=False))


def main(
    label: str = "y_growth_related",
    encoder: str = "tcn_film",
    posterior_dir: Path | None = None,
    n_athletes: int | None = None,
    n_folds: int = 5,
    ensemble_size: int | None = None,
    max_epochs: int | None = None,
    pretrain_epochs: int | None = None,
    only: str | None = None,
    seed: int = 0,
    threads: int = 4,
    out_dir: Path | None = None,
    report_from: Path | None = None,
) -> None:
    if report_from is not None:
        _write_report(pd.read_parquet(report_from / "predictions.parquet"), report_from, seed)
        return
    torch.set_num_threads(threads)  # small networks: more threads only add overhead
    cohort_config = load_config("sim/cohort.yaml")
    if n_athletes is not None:
        cohort_config["n_athletes"] = n_athletes
    configs = {
        "ablations": load_config("eval/ablations.yaml"),
        "baselines": load_config("models/baselines.yaml"),
        "person_period": load_config("features/person_period.yaml"),
        "neural": load_config("models/neural.yaml"),
        "encoder": load_config(f"models/{encoder}.yaml"),
    }
    if ensemble_size is not None:
        configs["neural"]["ensemble_size"] = ensemble_size
    if max_epochs is not None:
        configs["neural"]["max_epochs"] = max_epochs
    if only is not None:  # e.g. --only 8,9: a subset of the pre-registered grid
        keep = {int(k) for k in only.split(",")}
        grid = configs["ablations"]
        grid["ablations"] = [a for a in grid["ablations"] if int(a["id"]) in keep]
        names = {a["name"] for a in grid["ablations"]}
        grid["comparisons"] = [
            c for c in grid["comparisons"] if {c["model"], c["reference"]} <= names
        ]
    if pretrain_epochs is not None:
        configs["neural"]["pretraining"]["epochs"] = pretrain_epochs
    run_config = {"cohort": cohort_config, **configs, "label": label, "folds": n_folds}
    metadata = run_metadata(seed=seed, config=run_config, synthetic=True)
    run_id = f"ablation-{config_hash(run_config)}"
    target = out_dir or repo_root() / "outputs" / "ablation" / run_id
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cohort = generate_cohort(cohort_config)
    posteriors = None
    if posterior_dir is not None:
        posteriors = {
            sex: SitarPosterior.load(posterior_dir / f"posterior_{sex}")
            for sex in ("female", "male")
            if (posterior_dir / f"posterior_{sex}.npz").is_file()
        }
        known = {a for p in posteriors.values() for a in p.athlete_ids}
        if not set(cohort.tables["athletes"]["id"]) <= known:
            raise typer.BadParameter("The saved posteriors do not cover this cohort's athletes")
    sequences, excluded = dataset.cohort_sequences(
        cohort,
        cohort_config,
        filter_config=load_config("pose/filter.yaml"),
        event_config=load_config("pose/events.yaml"),
        feature_config=load_config("features/task_features.yaml"),
    )
    excluded.to_csv(target / "excluded_trials.csv", index=False)
    typer.echo(
        f"{len(sequences.session_ids)} sessions, {len(excluded)} trial(s) excluded "
        f"[{time.time() - started:.0f} s]"
    )
    period = build_person_period(
        cohort.tables,
        config=configs["person_period"],
        hazard_config=load_config("models/hazard.yaml"),
        posteriors=posteriors,
    )
    frame = add_mech_proxies(period.frame, load_config("features/mech.yaml"))
    truth = true_window_risk(cohort, frame, label, period.horizon_weeks)
    run = run_ablation(
        frame,
        sequences,
        label=label,
        configs=configs,
        posteriors=posteriors,
        n_folds=n_folds,
        seed=seed,
        true_risk=truth,
    )
    results = run.results
    results.to_csv(target / "results.csv", index=False)
    run.predictions.to_parquet(target / "predictions.parquet", index=False)
    record = {
        "run_id": run_id,
        **metadata,
        "person_period": period.summary,
        "ensemble_size": int(configs["neural"]["ensemble_size"]),
        "n_growth_draws": int(configs["neural"]["uncertainty"]["n_growth_draws"]),
        "ablation_ids": [int(a["id"]) for a in configs["ablations"]["ablations"]],
        "seconds": round(time.time() - started, 1),
    }
    (target / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"\nSYNTHETIC ablation {run_id} -> {target}")
    wide = results.pivot_table(index=["ablation_id", "name"], columns="metric", values="value")
    typer.echo(wide.round(4).to_string())
    for note in sorted(set(results["note"]) - {""}):
        typer.echo(f"note: {note}")
    _write_report(run.predictions, target, seed, configs["ablations"])


if __name__ == "__main__":
    typer.run(main)
