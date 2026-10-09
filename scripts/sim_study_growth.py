"""Simulation study for the growth twin (C1, H1): posterior APHV vs truth vs Moore-2015.

Generates the synthetic cohort from its seed, fits the model per sex (both sexes at the same
time, each in its own process), and writes per-athlete scores, a summary, saved posteriors and
a SYNTHETIC-watermarked figure to ``outputs/sim_study_growth/<run_id>/``. Nothing here is a
finding about real athletes.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from youthtwin.growth import evaluation, sitar_bayes
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata

SEXES = ("female", "male")
Job = tuple[str, dict[str, Any], dict[str, Any], str, dict[str, Any]]


def _figure(scores: pd.DataFrame, level: float) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    low, high = scores["true_aphv"].min() - 0.5, scores["true_aphv"].max() + 0.5
    panels = (
        ("Growth twin (posterior mean, interval)", "posterior_mean", True),
        ("Moore 2015 point estimate", "point_estimate", False),
    )
    for axis, (title, column, with_interval) in zip(axes, panels, strict=True):
        for sex, marker in zip(SEXES, ("o", "s"), strict=True):
            part = scores[scores["sex"] == sex]
            if with_interval:
                errors = [
                    part[column] - part["interval_low"],
                    part["interval_high"] - part[column],
                ]
                axis.errorbar(
                    part["true_aphv"],
                    part[column],
                    yerr=errors,
                    fmt=marker,
                    markersize=3,
                    linewidth=0.5,
                    alpha=0.6,
                    label=sex,
                )
            else:
                axis.plot(
                    part["true_aphv"], part[column], marker, markersize=3, alpha=0.6, label=sex
                )
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("True age at PHV (years)")
        axis.legend()
    axes[0].set_ylabel(f"Estimated age at PHV (years; {level:.0%} interval)")
    return figure


def _fit_one_sex(job: Job) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit and score one sex. Runs in its own process."""
    sex, cohort_config, model_config, target, metadata = job
    seed = int(cohort_config["seed"])
    cohort = generate_cohort(cohort_config)
    truth = cohort.truth["athlete_truth"].set_index("athlete_id")
    data = sitar_bayes.prepare_growth_data(
        cohort.tables["athletes"],
        cohort.tables["anthropometrics"],
        sex,  # type: ignore[arg-type]
    )
    started = time.time()
    idata = sitar_bayes.fit(data, model_config, seed=seed)
    posterior = sitar_bayes.extract_posterior(idata, data, model_config, seed=seed)
    info: dict[str, Any] = {
        "n_athletes": data.n_athletes,
        "n_observations": data.n_observations,
        "seconds": round(time.time() - started, 1),
        "share_draws_with_interior_velocity_peak": float(posterior.has_interior_peak().mean()),
    }
    if model_config["inference"]["method"] == "nuts":
        info.update(sitar_bayes.diagnostics(idata))
    posterior.save(Path(target) / f"posterior_{sex}", metadata)
    scores = evaluation.score_aphv(
        posterior,
        truth["true_aphv"].to_dict(),
        dict(zip(data.athlete_ids, data.prior_aphv, strict=True)),
        level=float(model_config["acceptance"]["interval_level"]),
    )
    scores["baseline_age"] = truth.loc[scores["athlete_id"], "baseline_age"].to_numpy()
    return scores, info


def main(
    n_athletes: int | None = None,
    method: str | None = None,
    draws: int | None = None,
    tune: int | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> None:
    cohort_config = load_config("sim/cohort.yaml")
    model_config = load_config("growth/sitar.yaml")
    if n_athletes is not None:
        cohort_config["n_athletes"] = n_athletes
    if seed is not None:
        cohort_config["seed"] = seed
    for key, value in (("method", method), ("draws", draws), ("tune", tune)):
        if value is not None:
            model_config["inference"][key] = value
    run_config: dict[str, Any] = {"cohort": cohort_config, "model": model_config}
    metadata = run_metadata(seed=int(cohort_config["seed"]), config=run_config, synthetic=True)
    run_id = f"simgrowth-{config_hash(run_config)}"
    metadata = {"run_id": run_id, **metadata}
    target = out_dir or repo_root() / "outputs" / "sim_study_growth" / run_id
    target.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = [(sex, cohort_config, model_config, str(target), metadata) for sex in SEXES]
    with ProcessPoolExecutor(max_workers=len(SEXES), mp_context=get_context("spawn")) as pool:
        results = list(pool.map(_fit_one_sex, jobs))
    fits = {sex: info for sex, (_, info) in zip(SEXES, results, strict=True)}
    for sex, info in fits.items():
        typer.echo(f"{sex}: {info}")

    level = float(model_config["acceptance"]["interval_level"])
    coverage_range = model_config["acceptance"]["coverage_range"]
    scores = pd.concat([part for part, _ in results], ignore_index=True)
    mid_follow_up = scores["baseline_age"] + cohort_config["follow_up_weeks"] / 52.0 / 2.0
    scores["stratum"] = evaluation.maturity_stratum(
        (mid_follow_up - scores["true_aphv"]).to_numpy(), band_years=1.0
    )
    scores.to_parquet(target / "per_athlete.parquet", index=False)
    summary = pd.concat(
        [
            evaluation.summarise_scores(scores, level=level, coverage_range=coverage_range, by=by)
            for by in (None, "sex", "stratum")
        ],
        ignore_index=True,
    )
    summary.to_csv(target / "summary.csv", index=False)
    (target / "run.json").write_text(json.dumps({**metadata, "fits": fits}, indent=2), "utf-8")
    save_figure(_figure(scores, level), target / "aphv_recovery.png", metadata)

    typer.echo(f"\nSYNTHETIC simulation study {run_id} -> {target}")
    typer.echo(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
