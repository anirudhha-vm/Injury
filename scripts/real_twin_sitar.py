"""Classical SITAR benchmark on the twin's held-out visits (M12; REAL Berkeley data).

For each sex and fold of the primary sequential run (same folds, seeds and entry ages), SITAR is
refitted in R for every held-out child and visit count; see
``youthtwin.external.sitar_benchmark``. Writes ``outputs/real_twin_sitar/<run_id>/``.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from youthtwin.external import berkeley, sequential, sitar_benchmark
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata

DATA = Path("data/external/sitar/sitar/data/berkeley.rda")
Job = tuple[str, int, dict[str, Any], dict[str, Any]]


def _job(job: Job) -> pd.DataFrame:
    sex, fold, config, settings = job
    frame_all = berkeley.load_berkeley(repo_root() / DATA)
    frame = frame_all[frame_all["sex"] == sex]
    folds = sequential.assign_folds(
        sorted(frame["athlete_id"].unique()), int(config["n_folds"]), int(config["seed"])
    )
    seed = int(config["seed"]) + 100 * fold + (0 if sex == "female" else 50)  # as the twin run
    train = [a for a, k in folds.items() if k != fold]
    held_out = [a for a, k in folds.items() if k == fold]
    measure = str(settings["measure"])
    sequences = sequential.entry_sequences(frame, held_out, config, seed + 1)
    out = sitar_benchmark.refit(
        sitar_benchmark.training_rows(frame, train, config, measure),
        sitar_benchmark.task_rows(sequences, config, measure),
        settings,
        config["train_age_range_years"],
    )
    ages = pd.DataFrame(
        [
            {"athlete_id": a, "visits": k, "age": float(v["age"].iloc[k - 1])}
            for a, v in sequences.items()
            for k in range(1, min(len(v), int(config["max_visits"])) + 1)
        ]
    )
    return out.merge(ages, on=["athlete_id", "visits"], how="left").assign(sex=sex, fold=fold)


def _versions() -> dict[str, str]:
    script = 'cat(R.version.string, "|", as.character(packageVersion("sitar")))'
    text = subprocess.run(
        [sitar_benchmark.rscript(), "-e", f'.libPaths(c("~/R/library", .libPaths())); {script}'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    r_version, sitar_version = (part.strip() for part in text.split("|"))
    return {"r": r_version, "sitar": sitar_version}


def main(out_dir: Path | None = None) -> None:
    config = load_config("external/sequential.yaml")
    config["scenario"] = {"visit_step": 1, "added_noise_cm": 0.0, "sitting_height": True}
    settings = load_config("external/sitar_benchmark.yaml")
    versions = _versions()
    run_config = {"sequential": config, "sitar": settings, "versions": versions}
    metadata = {
        "run_id": f"sitar-{config_hash(run_config)}",
        **run_metadata(seed=int(config["seed"]), config=run_config, synthetic=False),
        "versions": versions,
    }
    target = out_dir or repo_root() / "outputs" / "real_twin_sitar" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = [
        (sex, fold, config, settings)
        for sex in ("female", "male")
        for fold in range(int(config["n_folds"]))
    ]
    with ThreadPoolExecutor(max_workers=int(settings["workers"])) as pool:
        predictions = pd.concat(list(pool.map(_job, jobs)), ignore_index=True)
    predictions.to_csv(target / "predictions.csv", index=False)
    status = predictions["sitar_status"].value_counts().to_dict()
    summary = {**metadata, "n": len(predictions), "status": status, "synthetic": False}
    (target / "run.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    typer.echo(f"REAL-DATA SITAR benchmark {metadata['run_id']} -> {target}")
    typer.echo(json.dumps(status))


if __name__ == "__main__":
    typer.run(main)
