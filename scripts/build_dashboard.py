"""Build the data bundle for the research dashboard (``make dashboard-data``).

Generates the synthetic cohort from its seed, loads the growth-twin posteriors saved by
``scripts/sim_study_growth.py`` (or ``scripts/fit_growth.py``), computes out-of-fold risk with
uncertainty, explanations and decision-support prompts, and writes the bundle to
``outputs/dashboard/bundle``. View it with ``make dashboard``.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer

from youthtwin.dashboard.bundle import build_bundle, write_bundle
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata


def _latest_growth_run() -> Path | None:
    root = repo_root() / "outputs" / "sim_study_growth"
    runs = (
        [p for p in root.glob("*") if (p / "posterior_male.npz").is_file()] if root.is_dir() else []
    )
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def main(posterior_dir: Path | None = None, out_dir: Path | None = None, seed: int = 0) -> None:
    source = posterior_dir or _latest_growth_run()
    if source is None:
        raise typer.BadParameter(
            "No saved growth posteriors found. Run `make sim-study-growth` first (about 3 hours), "
            "or pass --posterior-dir."
        )
    cohort_config = load_config("sim/cohort.yaml")
    configs = {
        "dashboard": load_config("dashboard.yaml"),
        "hazard": load_config("models/hazard.yaml"),
        "person_period": load_config("features/person_period.yaml"),
    }
    started = time.time()
    cohort = generate_cohort(cohort_config)
    posteriors = {
        sex: SitarPosterior.load(source / f"posterior_{sex}") for sex in ("female", "male")
    }
    known = {a for p in posteriors.values() for a in p.athlete_ids}
    if not set(cohort.tables["athletes"]["id"]) <= known:
        raise typer.BadParameter("The saved posteriors do not cover this cohort's athletes")
    bundle = build_bundle(cohort, cohort_config, posteriors, configs, seed=seed)
    run_config = {"cohort": cohort_config, **configs, "posteriors": source.name}
    metadata = {
        "run_id": f"dashboard-{config_hash(run_config)}",
        "growth_posteriors": source.name,
        **run_metadata(seed=seed, config=run_config, synthetic=True),
    }
    target = write_bundle(
        bundle, out_dir or repo_root() / "outputs" / "dashboard" / "bundle", metadata
    )
    typer.echo(f"SYNTHETIC dashboard bundle written to {target} [{time.time() - started:.0f} s]")
    for name, table in bundle.tables.items():
        typer.echo(f"  {name}: {len(table)} rows")


if __name__ == "__main__":
    typer.run(main)
