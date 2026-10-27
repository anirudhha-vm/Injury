"""Sequential growth-twin evaluation on REAL data: the Berkeley Child Guidance Study (M11).

For each sex and each of ``n_folds`` folds (grouped by child) the shared growth curve is fitted
by NUTS to the training children's full records, with each training child's Moore prior taken
from one random season. Held-out children then enter at a random age and their twin is updated
after every visit, scored on age at PHV (against the reference from the full record) and on its
six-month forecast. Writes to ``outputs/real_twin_sequential/<run_id>/``:

``scores.csv, summary_visits.csv, summary_band.csv, summary_overall.csv, twin_sequential.png,
run.json``.
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
from omegaconf import OmegaConf

from youthtwin.external import berkeley, reference_aphv, sequential
from youthtwin.growth import sitar_bayes
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata

DATA = Path("data/external/sitar/sitar/data/berkeley.rda")
BANDS = (-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf)
BAND_LABELS = (
    "more than 3 before",
    "3 to 1 before",
    "within 1",
    "1 to 3 after",
    "more than 3 after",
)
Job = tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any]]


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = OmegaConf.merge(load_config("growth/sitar.yaml"), config["model_overrides"])
    out: dict[str, Any] = OmegaConf.to_container(merged)  # type: ignore[assignment]
    return out


def _fold(job: Job) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one sex x fold and score its held-out children (own process)."""
    sex, fold, config, model_config, window_config = job
    frame = berkeley.load_berkeley(repo_root() / DATA)
    frame = frame[frame["sex"] == sex]
    reference = reference_aphv.reference_table(
        frame, load_config("external/berkeley.yaml")["reference"]
    )
    truth = dict(
        zip(
            reference.loc[reference["included"], "athlete_id"],
            reference.loc[reference["included"], "reference_aphv"],
            strict=True,
        )
    )
    seed = int(config["seed"]) + 100 * fold + (0 if sex == "female" else 50)
    folds = sequential.assign_folds(
        sorted(frame["athlete_id"].unique()), int(config["n_folds"]), int(config["seed"])
    )
    train = [a for a, k in folds.items() if k != fold]
    held_out = [a for a, k in folds.items() if k == fold]
    windows = berkeley.draw_windows(frame, train, window_config, seed)
    data = sequential.training_data(frame, train, sex, config, windows)  # type: ignore[arg-type]
    started = time.time()
    idata = sitar_bayes.fit(data, model_config, seed=seed)
    info: dict[str, Any] = {
        "sex": sex,
        "fold": fold,
        "n_train": data.n_athletes,
        "n_train_obs": data.n_observations,
        "seconds": round(time.time() - started, 1),
        "method": model_config["inference"]["method"],
    }
    if model_config["inference"]["method"] == "nuts":
        info.update(sitar_bayes.diagnostics(idata))
    if "sigma_beta_slope" in idata.posterior:
        info["sigma_beta_slope_mean"] = float(idata.posterior["sigma_beta_slope"].mean())
    info["sigma_beta_mean"] = float(idata.posterior["sigma_beta"].mean())
    shared = sitar_bayes.shared_draws(idata, int(config["n_posterior_draws"]), seed, model_config)
    sequences = sequential.entry_sequences(frame, held_out, config, seed + 1)
    scores = sequential.score_sequences(
        shared, sequences, truth, sex, model_config, config, seed=seed + 2
    )  # type: ignore[arg-type]
    return scores.assign(fold=fold), info


def _figure(by_visits: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for sex, part in by_visits.groupby("sex"):
        style = "-" if sex == "female" else "--"
        axes[0].plot(part["visits"], part["twin_mae"], "o" + style, label=f"twin ({sex})")
        axes[0].plot(part["visits"], part["moore_mae"], "s" + style, label=f"Moore ({sex})")
        axes[1].plot(part["visits"], part["twin_coverage"], "o" + style, label=sex)
        axes[2].plot(part["visits"], part["twin_mae_same_rows"], "o" + style, label=f"twin ({sex})")
        axes[2].plot(
            part["visits"], part["naive_mae"], "s" + style, label=f"velocity extrapolation ({sex})"
        )
    axes[0].set_title("Age at PHV: mean absolute error (y)")
    axes[1].set_title("Twin interval coverage")
    axes[1].axhline(0.9, color="black", linewidth=0.8, linestyle=":")
    axes[2].set_title("Six-month stature forecast: MAE (cm)")
    for axis in axes:
        axis.set_xlabel("Visits seen by the twin")
        axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


def main(
    n_folds: int | None = None,
    draws: int | None = None,
    workers: int = 4,
    leg: bool = True,
    leg_lag: bool = True,
    method: str | None = None,
    out_dir: Path | None = None,
) -> None:
    config = load_config("external/sequential.yaml")
    config["use_leg"] = leg  # --no-leg: the stature-only twin, for comparison
    config["model_overrides"]["bivariate"]["enabled"] = leg
    if not leg_lag:  # legs share each athlete's timing exactly (no individual lag)
        config["model_overrides"]["bivariate"]["lag"] = {"points": 1}
    if n_folds is not None:
        config["n_folds"] = n_folds
    model_config = _model_config(config)
    if draws is not None:
        model_config["inference"]["draws"] = model_config["inference"]["tune"] = draws
    if method is not None:  # map: reference curve at its mode; nuts: full posterior
        model_config["inference"]["method"] = method
    window_config = load_config("external/berkeley.yaml")["window"]
    if not (repo_root() / DATA).is_file():
        raise typer.BadParameter("Berkeley data not found; run scripts/fetch_public_data.py first")
    run_config = {"sequential": config, "model": model_config, "window": window_config}
    metadata = {
        "run_id": f"realtwin-{config_hash(run_config)}",
        **run_metadata(seed=int(config["seed"]), config=run_config, synthetic=False),
        "dataset": "Berkeley Child Guidance Study (sitar 1.5.0)",
    }
    target = out_dir or repo_root() / "outputs" / "real_twin_sequential" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs: list[Job] = [
        (sex, fold, config, model_config, window_config)
        for fold in range(int(config["n_folds"]))
        for sex in ("female", "male")
    ]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(_fold, jobs))
    scores = pd.concat([s for s, _ in results], ignore_index=True)
    scores["band"] = pd.cut(scores["years_from_phv"], BANDS, labels=BAND_LABELS).astype(str)
    scores = sequential.conformalize(scores, float(config["interval_level"]), by=("visits",))
    scores["synthetic"] = False
    scores.to_csv(target / "scores.csv", index=False)
    by_visits = sequential.summarise(scores, ["sex", "visits"])
    by_band = sequential.summarise(scores, ["band"])
    overall = sequential.summarise(scores.assign(all="all"), ["all"])
    by_visits.to_csv(target / "summary_visits.csv", index=False)
    by_band.to_csv(target / "summary_band.csv", index=False)
    overall.to_csv(target / "summary_overall.csv", index=False)
    record = {
        **metadata,
        "fits": [info for _, info in results],
        "seconds": round(time.time() - started, 1),
    }
    (target / "run.json").write_text(json.dumps(record, indent=2, default=str), "utf-8")
    save_figure(_figure(by_visits), target / "twin_sequential.png", record)
    typer.echo(f"REAL-DATA sequential twin {metadata['run_id']} -> {target}")
    for info in record["fits"]:
        typer.echo(f"  {info}")
    typer.echo(overall.round(3).to_string(index=False))
    typer.echo(by_visits.round(3).to_string(index=False))
    typer.echo(by_band.round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
