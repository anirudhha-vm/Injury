"""Evaluate the tabular risk models with athlete-grouped cross-validation.

Generates the synthetic cohort from its seed, builds the athlete-week table (with growth-twin
features if ``--posterior-dir`` is given), produces out-of-fold predictions for the three
baselines, the penalised-logistic hazard and the boosted hazard, and writes to
``outputs/evaluation/<run_id>/``:

``predictions.parquet, performance.csv, differences.csv, decision_curve.csv, calibration.csv,
subgroups.csv, shap_importance.csv, what_if.json, run.json``

SYNTHETIC: a pipeline check, not findings. Use ``scripts/make_figures.py`` to turn a run into
figures and tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import typer

from youthtwin.eval import report, splits
from youthtwin.eval.predictions import out_of_fold_predictions
from youthtwin.explain import counterfactual, shap_utils
from youthtwin.features.mech_load import add_mech_proxies
from youthtwin.features.person_period import build_person_period
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models.baselines import make_baseline
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.models.risk.hazard_xgb import make_hazard_xgb
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata

SUBGROUP_COLUMNS = ["sex", "sport", "age", "moore_maturity_offset", "twin_maturity_offset_mean"]


def main(
    label: str = "y_growth_related",
    posterior_dir: Path | None = None,
    n_athletes: int | None = None,
    n_folds: int = 5,
    n_resamples: int | None = None,
    reference: str = "maturity_growth_velocity",
    seed: int = 0,
    out_dir: Path | None = None,
) -> None:
    cohort_config = load_config("sim/cohort.yaml")
    if n_athletes is not None:
        cohort_config["n_athletes"] = n_athletes
    metrics_config = load_config("eval/metrics.yaml")
    if n_resamples is not None:
        metrics_config["bootstrap"]["n_resamples"] = n_resamples
    person_period_config = load_config("features/person_period.yaml")
    hazard_config = load_config("models/hazard.yaml")
    baseline_config = load_config("models/baselines.yaml")
    run_config = {
        "cohort": cohort_config,
        "metrics": metrics_config,
        "hazard": hazard_config,
        "baselines": baseline_config,
        "label": label,
        "folds": n_folds,
    }
    metadata = run_metadata(seed=seed, config=run_config, synthetic=True)
    run_id = f"evaluation-{config_hash(run_config)}"
    target = out_dir or repo_root() / "outputs" / "evaluation" / run_id
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
    period = build_person_period(
        cohort.tables,
        config=person_period_config,
        hazard_config=hazard_config,
        posteriors=posteriors,
    )
    frame = add_mech_proxies(period.frame, load_config("features/mech.yaml"))
    features = period.predictors + [c for c in frame.columns if c.startswith("mech_")]

    factories = {
        name: (lambda f, _n=name: make_baseline(_n, f, baseline_config, person_period_config))
        for name in baseline_config["baselines"]
    }
    factories["hazard_logit"] = lambda f: make_hazard_logit(
        f, features, hazard_config["hazard_logit"], person_period_config, seed=seed
    )
    factories["hazard_xgb"] = lambda f: make_hazard_xgb(
        f, features, hazard_config["hazard_xgb"], person_period_config, seed=seed
    )
    folds = splits.grouped_stratified_folds(
        frame["athlete_id"], frame[label], n_splits=n_folds, seed=seed
    )
    predictions = out_of_fold_predictions(frame, label, factories, folds, keep=SUBGROUP_COLUMNS)
    predictions.to_parquet(target / "predictions.parquet", index=False)
    typer.echo(
        f"out-of-fold predictions for {len(factories)} models [{time.time() - started:.0f} s]"
    )

    maturity = (
        "twin_maturity_offset_mean"
        if "twin_maturity_offset_mean" in predictions.columns
        else "moore_maturity_offset"
    )
    tables = {
        "performance": report.performance_table(predictions, metrics_config, seed=seed),
        "differences": report.difference_table(predictions, reference, metrics_config, seed=seed),
        "decision_curve": report.decision_curve_table(predictions, metrics_config),
        "calibration": report.calibration_table(predictions, metrics_config),
        "subgroups": report.subgroup_report(predictions, metrics_config, maturity),
    }

    # Explanations from models fitted on the first fold's training athletes.
    train, held_out = frame[folds[0].train], frame[folds[0].test]
    importances = []
    for name in ("hazard_logit", "hazard_xgb"):
        model = factories[name](frame).fit(train, label)
        shap_result = shap_utils.explain(model, held_out, train)
        ranking = shap_utils.importance(shap_result, model.features)
        importances.append(
            pd.DataFrame(
                {"model": name, "input": ranking.index, "mean_abs_shap": ranking.to_numpy()}
            )
        )
        if name == "hazard_logit":
            settings = metrics_config["counterfactual"]
            changed = counterfactual.scale_columns(
                held_out, settings["exposure_columns"], float(settings["exposure_scale"])
            )
            what_if = counterfactual.summarise_what_if(
                counterfactual.what_if(model, held_out, changed)
            )
            (target / "what_if.json").write_text(json.dumps(what_if, indent=2), encoding="utf-8")
    tables["shap_importance"] = pd.concat(importances, ignore_index=True).assign(
        label=label, synthetic=True, note="log-odds units; fold 0 training athletes"
    )
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False)
    record = {
        "run_id": run_id,
        **metadata,
        "label": label,
        "reference_model": reference,
        "person_period": period.summary,
        "growth_twin_features": posteriors is not None,
        "seconds": round(time.time() - started, 1),
    }
    (target / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    typer.echo(f"\nSYNTHETIC evaluation {run_id} -> {target}")
    primary = tables["performance"][
        tables["performance"]["metric"].isin(
            [metrics_config["primary_metric"], "calibration_slope"]
        )
    ]
    typer.echo(
        primary[["model", "metric", "estimate", "ci_low", "ci_high"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    typer.run(main)
