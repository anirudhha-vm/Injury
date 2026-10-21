"""Injury-risk evaluation on REAL outcomes: the competitive runners' dataset (M10, Part 2).

Reads ``data/external/runners/week_approach_maskedID_timeseries.csv``
(``scripts/fetch_public_data.py``) and evaluates the configured models under three schemes:
runner-grouped cross-validation (primary, N3), a temporal hold-out of the last quarter of
calendar time, and row-level cross-validation that leaks runners across folds (shown only to
measure the inflation it causes). Writes to ``outputs/real_risk_runners/<run_id>/``:

``predictions.parquet, performance.csv, differences.csv, calibration.csv, calibration.png,
leakage_inflation.csv, run.json``.

Adult runners and running injuries: a real-data test of the evaluation machinery, not evidence
about growth-related injury in youth.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from youthtwin.eval import bootstrap, metrics, report
from youthtwin.external import runners
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata

DATA = Path("data/external/runners/week_approach_maskedID_timeseries.csv")


def _calibration_figure(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5.5, 5))
    for model, part in table.groupby("model"):
        axis.plot(part["predicted"], part["observed"], marker="o", label=model)
    top = float(max(table["predicted"].max(), table["observed"].max())) * 1.05
    axis.plot([0, top], [0, top], color="black", linewidth=1)
    axis.set_xlabel("Mean predicted risk (decile)")
    axis.set_ylabel("Observed injury rate")
    axis.set_title("Runner-grouped cross-validation (real data)")
    axis.legend()
    figure.tight_layout()
    return figure


def _leakage_inflation(
    predictions: list[pd.DataFrame], models: list[str], n_resamples: int, seed: int
) -> pd.DataFrame:
    """AUC under row-level folds minus AUC under runner-grouped folds, same rows and models,
    paired on resampled runners: how much letting a runner sit on both sides inflates AUC."""
    by_scheme = {p["scheme"].iloc[0]: p for p in predictions}
    grouped, leaky = by_scheme["runner_grouped"], by_scheme["row_level_leaky"]
    same_rows = grouped["athlete_id"].equals(leaky["athlete_id"]) and grouped["week_start"].equals(
        leaky["week_start"]
    )
    if not same_rows:
        raise RuntimeError("The two cross-validation schemes do not cover the same rows in order")
    rows = []
    for model in models:
        result = bootstrap.cluster_bootstrap_difference(
            metrics.auc,
            grouped["outcome"].to_numpy(),
            leaky[f"risk_{model}"].to_numpy(),
            grouped[f"risk_{model}"].to_numpy(),
            grouped["athlete_id"].to_numpy(),
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append(
            {
                "model": model,
                "metric": "auc",
                "leaky_minus_grouped": result.estimate,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "ci_level": result.level,
                "synthetic": False,
            }
        )
    return pd.DataFrame(rows)


def main(n_resamples: int | None = None, out_dir: Path | None = None) -> None:
    config = load_config("external/runners.yaml")
    if n_resamples is not None:
        config["bootstrap_resamples"] = n_resamples
    metrics_config = copy.deepcopy(load_config("eval/metrics.yaml"))
    metrics_config["metrics"] = list(config["metrics"])
    metrics_config["bootstrap"]["n_resamples"] = int(config["bootstrap_resamples"])
    metrics_config["thresholds"] = []
    path = repo_root() / DATA
    if not path.is_file():
        raise typer.BadParameter(f"{path} not found; run scripts/fetch_public_data.py first")
    seed = int(config["seed"])
    run_config = {"runners": config, "metrics": metrics_config}
    metadata = {
        "run_id": f"realrisk-{config_hash(run_config)}",
        **run_metadata(seed=seed, config=run_config, synthetic=False),
        "dataset": "Lovdal, den Hartigh & Azzopardi 2021, doi:10.34894/UWU9PV (week approach)",
    }
    target = out_dir or repo_root() / "outputs" / "real_risk_runners" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frame = runners.tidy_week_approach(pd.read_csv(path), config)
    label = str(config["label"])
    factories = runners.model_factories(frame, config, seed)
    n_inputs = len(runners.feature_columns(frame, config))
    typer.echo(
        f"{frame['athlete_id'].nunique()} runners, {len(frame)} event days, "
        f"{int(frame[label].sum())} injury days, {n_inputs} inputs"
    )
    predictions, performance = [], []
    for scheme, folds in runners.evaluation_splits(frame, config, seed).items():
        held_out = runners.predict(frame, label, factories, folds, scheme)
        table = report.performance_table(held_out, metrics_config, seed=seed)
        table.insert(0, "scheme", scheme)
        predictions.append(held_out)
        performance.append(table)
        typer.echo(f"  {scheme}: {len(held_out)} held-out rows [{time.time() - started:.0f} s]")
        if scheme == "runner_grouped":
            differences = report.difference_table(
                held_out, str(config["reference_model"]), metrics_config, seed=seed
            )
            calibration = report.calibration_table(held_out, metrics_config)
    pd.concat(predictions, ignore_index=True).to_parquet(
        target / "predictions.parquet", index=False
    )
    performance_table = pd.concat(performance, ignore_index=True)
    performance_table.to_csv(target / "performance.csv", index=False)
    differences.to_csv(target / "differences.csv", index=False)
    calibration.to_csv(target / "calibration.csv", index=False)
    inflation = _leakage_inflation(
        predictions, list(factories), int(config["bootstrap_resamples"]), seed
    )
    inflation.to_csv(target / "leakage_inflation.csv", index=False)
    record = {
        **metadata,
        "n_runners": int(frame["athlete_id"].nunique()),
        "n_rows": len(frame),
        "n_injury_days": int(frame[label].sum()),
        "seconds": round(time.time() - started, 1),
    }
    (target / "run.json").write_text(json.dumps(record, indent=2, default=str), "utf-8")
    save_figure(_calibration_figure(calibration), target / "calibration.png", record)

    typer.echo(f"\nREAL-DATA risk evaluation {metadata['run_id']} -> {target}")
    shown = performance_table[
        performance_table["metric"].isin(
            ["time_dependent_auc_4wk", "calibration_slope", "brier_score"]
        )
    ]
    typer.echo(
        shown[["scheme", "model", "metric", "estimate", "ci_low", "ci_high"]]
        .round(3)
        .to_string(index=False)
    )
    typer.echo("\nPaired differences vs the load-only model (runner-grouped):")
    typer.echo(
        differences[["model", "metric", "difference", "ci_low", "ci_high"]]
        .round(3)
        .to_string(index=False)
    )
    typer.echo("\nAUC inflation from row-level (leaky) cross-validation:")
    typer.echo(inflation.round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
