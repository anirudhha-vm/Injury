"""Run the ablation grid on the synthetic cohort and write one tidy results table.

Generates the cohort from its seed, simulates and processes each session's trials, builds the
athlete-week table (with growth-twin features if ``--posterior-dir`` points at saved posteriors
from ``scripts/sim_study_growth.py`` or ``scripts/fit_growth.py``) and runs every configuration
in ``configs/eval/ablations.yaml``. Output: ``outputs/ablation/<run_id>/results.csv`` with a
provenance record. SYNTHETIC: these numbers are a pipeline check, not findings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer

from youthtwin.eval.ablation import run_ablation
from youthtwin.features.mech_load import add_mech_proxies
from youthtwin.features.person_period import build_person_period
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models import dataset
from youthtwin.sim.cohort import generate_cohort
from youthtwin.sim.truth import true_window_risk
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata


def main(
    label: str = "y_growth_related",
    encoder: str = "tcn_film",
    posterior_dir: Path | None = None,
    n_athletes: int | None = None,
    n_folds: int = 5,
    ensemble_size: int | None = None,
    max_epochs: int | None = None,
    seed: int = 0,
    threads: int = 4,
    out_dir: Path | None = None,
) -> None:
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
    results = run_ablation(
        frame,
        sequences,
        label=label,
        configs=configs,
        posteriors=posteriors,
        n_folds=n_folds,
        seed=seed,
        true_risk=truth,
    )
    results.to_csv(target / "results.csv", index=False)
    record = {
        "run_id": run_id,
        **metadata,
        "person_period": period.summary,
        "seconds": round(time.time() - started, 1),
    }
    (target / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"\nSYNTHETIC ablation {run_id} -> {target}")
    wide = results.pivot_table(index=["ablation_id", "name"], columns="metric", values="value")
    typer.echo(wide.round(4).to_string())
    for note in sorted(set(results["note"]) - {""}):
        typer.echo(f"note: {note}")


if __name__ == "__main__":
    typer.run(main)
