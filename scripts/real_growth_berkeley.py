"""Growth twin on REAL data: the Berkeley Child Guidance Study (M10, Part 1).

Reads ``data/external/sitar/sitar/data/berkeley.rda`` (``scripts/fetch_public_data.py``),
fits a reference age at PHV to each child's full record, then gives the twin, Moore 2015 and
Mirwald 2002 (boys) only one season of each child's measurements. The twin is fitted per sex on
every child's window; scoring uses the children with a trustworthy reference. Replicates draw
new windows. Writes to ``outputs/real_growth_berkeley/<run_id>/``:

``reference.csv, per_athlete.csv, summary.csv, aphv_real.png, run.json``.

Real data (``synthetic=False``), but a historical cohort (born 1928-29, Berkeley, California):
results bear on the method, not on present-day Indian athletes.
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
import numpy as np
import pandas as pd
import typer

from youthtwin.external import berkeley, reference_aphv
from youthtwin.growth import evaluation, sitar_bayes
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata

SEXES = ("female", "male")
DATA = Path("data/external/sitar/sitar/data/berkeley.rda")
BANDS = (-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf)
BAND_LABELS = (
    "more than 3 before",
    "3 to 1 before",
    "within 1",
    "1 to 3 after",
    "more than 3 after",
)
Job = tuple[str, int, str, dict[str, Any], dict[str, Any]]


def _fit(job: Job) -> tuple[str, int, pd.DataFrame, dict[str, Any]]:
    """Fit one sex on one replicate's windows (own process). Returns per-athlete draws summary."""
    sex, replicate, windows_path, model_config, reference = job
    windows = pd.read_parquet(windows_path)
    part = windows[windows["sex"] == sex]
    data = sitar_bayes.growth_data_from_measurements(part, sex, synthetic=False)  # type: ignore[arg-type]
    started = time.time()
    seed = 1000 * replicate + SEXES.index(sex)
    idata = sitar_bayes.fit(data, model_config, seed=seed)
    posterior = sitar_bayes.extract_posterior(idata, data, model_config, seed=seed)
    info: dict[str, Any] = {
        "n_athletes": data.n_athletes,
        "n_observations": data.n_observations,
        "seconds": round(time.time() - started, 1),
    }
    if model_config["inference"]["method"] == "nuts":
        info.update(sitar_bayes.diagnostics(idata))
    scored = [a for a in posterior.athlete_ids if a in reference]
    moore = dict(zip(data.athlete_ids, data.prior_aphv, strict=True))
    scores = evaluation.score_aphv(
        posterior,
        {a: reference.get(a, np.nan) for a in posterior.athlete_ids},
        moore,
        level=float(model_config["acceptance"]["interval_level"]),
    )
    scores = scores[scores["athlete_id"].isin(scored)]
    return sex, replicate, scores, info


def _figure(scores: pd.DataFrame, level: float) -> plt.Figure:
    first = scores[scores["replicate"] == 0]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    low, high = first["true_aphv"].min() - 0.5, first["true_aphv"].max() + 0.5
    for axis, (title, column) in zip(
        axes,
        (
            ("Growth twin (one season)", "posterior_mean"),
            ("Moore 2015 (same season)", "point_estimate"),
        ),
        strict=True,
    ):
        for sex, marker in zip(SEXES, ("o", "s"), strict=True):
            part = first[first["sex"] == sex]
            if column == "posterior_mean":
                axis.errorbar(
                    part["true_aphv"],
                    part[column],
                    yerr=[
                        part[column] - part["interval_low"],
                        part["interval_high"] - part[column],
                    ],
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
        axis.set_xlabel("Reference age at PHV from full record (years)")
        axis.legend()
    axes[0].set_ylabel(f"Estimated age at PHV (years; {level:.0%} interval)")
    figure.suptitle("Berkeley Child Guidance Study (real data), replicate 0", fontsize=10)
    return figure


def _summary(scores: pd.DataFrame, level: float, coverage_range: list[float]) -> pd.DataFrame:
    frames = []
    for subset, part in (("all", scores), ("strict reference", scores[scores["strict"]])):
        for by in (None, "sex", "band"):
            table = evaluation.summarise_scores(
                part, level=level, coverage_range=coverage_range, by=by
            )
            table.insert(0, "grouping", by or "all")
            table.insert(0, "reference_subset", subset)
            mirwald = (
                part.groupby(by)["mirwald_abs_error"].mean()
                if by
                else pd.Series({"all": part["mirwald_abs_error"].mean()})
            )
            table["mirwald_mae_years"] = table["group"].map({str(k): v for k, v in mirwald.items()})
            frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    out["synthetic"] = False
    return out


def main(
    n_replicates: int | None = None, draws: int | None = None, out_dir: Path | None = None
) -> None:
    config = load_config("external/berkeley.yaml")
    model_config = load_config("growth/sitar.yaml")
    if n_replicates is not None:
        config["n_replicates"] = n_replicates
    if draws is not None:
        model_config["inference"]["draws"] = model_config["inference"]["tune"] = draws
    path = repo_root() / DATA
    if not path.is_file():
        raise typer.BadParameter(f"{path} not found; run scripts/fetch_public_data.py first")
    run_config = {"berkeley": config, "model": model_config}
    metadata = {
        "run_id": f"realgrowth-{config_hash(run_config)}",
        **run_metadata(seed=int(config["seed"]), config=run_config, synthetic=False),
        "dataset": "Berkeley Child Guidance Study (sitar 1.5.0)",
    }
    target = out_dir or repo_root() / "outputs" / "real_growth_berkeley" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()

    frame = berkeley.load_berkeley(path)
    reference = reference_aphv.reference_table(frame, config["reference"])
    reference.to_csv(target / "reference.csv", index=False)
    included = reference[reference["included"]]
    strict = set(
        included.loc[
            (included["reference_aphv"] - included["spline_aphv"]).abs()
            <= float(config["reference"]["strict_disagreement_years"]),
            "athlete_id",
        ]
    )
    typer.echo(
        f"reference APHV: {len(included)} of {len(reference)} children included "
        f"({len(strict)} under the strict rule)"
    )
    truth = dict(zip(included["athlete_id"], included["reference_aphv"], strict=True))

    jobs: list[Job] = []
    windows_by_replicate = {}
    for replicate in range(int(config["n_replicates"])):
        windows = berkeley.draw_windows(
            frame,
            sorted(frame["athlete_id"].unique()),
            config["window"],
            int(config["seed"]) + replicate,
        )
        windows_path = target / f"windows_{replicate}.parquet"
        windows.to_parquet(windows_path, index=False)
        windows_by_replicate[replicate] = windows
        jobs += [(sex, replicate, str(windows_path), model_config, truth) for sex in SEXES]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(_fit, jobs))

    level = float(model_config["acceptance"]["interval_level"])
    parts, fits = [], {}
    for sex, replicate, scores, info in results:
        windows = windows_by_replicate[replicate]
        equations = berkeley.equation_estimates(windows).set_index("athlete_id")
        starts = windows.groupby("athlete_id")["window_start"].first()
        scores = scores.assign(
            replicate=replicate,
            window_start=starts.loc[scores["athlete_id"]].to_numpy(),
            mirwald_aphv=equations.loc[scores["athlete_id"], "mirwald_aphv"].to_numpy(),
        )
        scores["mirwald_abs_error"] = (scores["mirwald_aphv"] - scores["true_aphv"]).abs()
        mid = scores["window_start"] + float(config["window"]["length_years"]) / 2.0
        scores["years_from_phv"] = mid - scores["true_aphv"]
        scores["band"] = pd.cut(scores["years_from_phv"], BANDS, labels=BAND_LABELS).astype(str)
        scores["strict"] = scores["athlete_id"].isin(strict)
        parts.append(scores)
        fits[f"{sex}_replicate{replicate}"] = info
        typer.echo(f"{sex} replicate {replicate}: {info}")
    scores = pd.concat(parts, ignore_index=True)
    scores.to_csv(target / "per_athlete.csv", index=False)
    summary = _summary(scores, level, model_config["acceptance"]["coverage_range"])
    summary.to_csv(target / "summary.csv", index=False)
    record = {**metadata, "fits": fits, "seconds": round(time.time() - started, 1)}
    (target / "run.json").write_text(json.dumps(record, indent=2, default=str), "utf-8")
    save_figure(_figure(scores, level), target / "aphv_real.png", record)
    typer.echo(f"\nREAL-DATA growth study {metadata['run_id']} -> {target}")
    columns = [
        "reference_subset",
        "grouping",
        "group",
        "n_athletes",
        "coverage",
        "posterior_mae_years",
        "point_mae_years",
        "mirwald_mae_years",
        "posterior_bias_years",
        "point_bias_years",
        "mean_interval_width_years",
    ]
    typer.echo(summary[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
