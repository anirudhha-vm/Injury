"""Fit the growth twin to a cohort directory and save one posterior per sex.

Reads ``athletes.parquet`` and ``anthropometrics.parquet`` from ``cohort_dir`` (as written by
``make sim``, or by ingestion of real data) and writes ``posterior_<sex>.npz/.json`` plus a run
record to ``out_dir``. With ``--update-from`` it performs the incremental update instead of a
refit: athlete effects are redrawn against the stored shared curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, get_args

import pandas as pd
import typer

from youthtwin.growth import sitar_bayes
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.growth.maturity_equations import Sex
from youthtwin.utils.config import load_config
from youthtwin.utils.provenance import config_hash, run_metadata


def main(
    cohort_dir: Path,
    out_dir: Path,
    config: str = "growth/sitar.yaml",
    seed: int = 0,
    update_from: Path | None = None,
) -> None:
    settings = load_config(config)
    athletes = pd.read_parquet(cohort_dir / "athletes.parquet")
    visits = pd.read_parquet(cohort_dir / "anthropometrics.parquet")
    synthetic = bool(athletes["synthetic"].all())
    metadata = run_metadata(seed=seed, config=settings, synthetic=synthetic)
    run_id = f"growth-{config_hash({'model': settings, 'cohort': str(cohort_dir), 'seed': seed})}"
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"run_id": run_id, **metadata, "fits": {}}

    for sex in get_args(Sex):
        if not (athletes["sex"] == sex).any():
            continue
        data = sitar_bayes.prepare_growth_data(athletes, visits, sex)
        started = time.time()
        if update_from is not None:
            previous = SitarPosterior.load(update_from / f"posterior_{sex}")
            posterior = sitar_bayes.update(previous, data, settings, seed=seed)
            fit_info: dict[str, Any] = {"mode": "update", "updated_from": str(update_from)}
        else:
            idata = sitar_bayes.fit(data, settings, seed=seed)
            posterior = sitar_bayes.extract_posterior(idata, data, settings, seed=seed)
            fit_info = {"mode": "fit"}
            if settings["inference"]["method"] == "nuts":
                fit_info.update(sitar_bayes.diagnostics(idata))
        posterior.save(out_dir / f"posterior_{sex}", {"run_id": run_id, **metadata})
        record["fits"][sex] = {
            **fit_info,
            "n_athletes": data.n_athletes,
            "n_observations": data.n_observations,
            "seconds": round(time.time() - started, 1),
        }
        typer.echo(f"{sex}: {record['fits'][sex]}")

    (out_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    label = "SYNTHETIC " if synthetic else ""
    typer.echo(f"{label}growth posteriors written to {out_dir} (run {run_id})")


if __name__ == "__main__":
    typer.run(main)
