"""Simulation study for the markerless-vs-mocap layer (Paper A; D-121). SYNTHETIC.

1. Calibrates the markerless landmark offset so that peak knee flexion in adult-size
   participants has the published mean absolute error (``configs/sim/validation.yaml``).
2. Simulates one study of the planned size (36 participants: 12 pre-, 12 circa-, 12 post-PHV)
   through the real pose pipeline and computes the pre-specified agreement statistics.
3. Repeats the whole study ``n_replicate_studies`` times to show how precisely a study of that
   size estimates them, and how often it would conclude RMSE < the pre-specified limit.

Writes to ``outputs/sim_study_validation/<run_id>/``: ``pairs.csv, agreement_overall.csv,
agreement_by_maturity_group.csv, replicates.csv, precision.csv, bland_altman.png,
rmse_by_maturity.png, run.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import typer
from scipy.optimize import brentq

from youthtwin.sim import validation_sim
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata
from youthtwin.validation import plots
from youthtwin.validation.agreement import agreement_table


def main(n_replicates: int | None = None, out_dir: Path | None = None) -> None:
    config = load_config("sim/validation.yaml")
    agreement_config = load_config("validation/agreement.yaml")
    if n_replicates is not None:
        config["n_replicate_studies"] = n_replicates
    pipeline = {
        "filter": load_config("pose/filter.yaml"),
        "events": load_config("pose/events.yaml"),
        "features": load_config("features/task_features.yaml"),
    }
    seed = int(config["seed"])
    run_config = {"validation": config, "agreement": agreement_config, **pipeline}
    metadata: dict[str, Any] = {
        "run_id": f"simvalidation-{config_hash(run_config)}",
        **run_metadata(seed=seed, config=run_config, synthetic=True),
    }
    target = out_dir or repo_root() / "outputs" / "sim_study_validation" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()

    settings = config["calibration"]

    def error_at(offset: float) -> float:
        pairs = validation_sim.simulate_study(
            config,
            pipeline,
            seed=seed - 1,  # calibration participants are not the study participants
            offset_sd_px=offset,
            groups={"post_phv": int(settings["n_participants"])},
        )
        return validation_sim.mean_absolute_error(pairs, settings["variable"]) - float(
            settings["target_mae_deg"]
        )

    offset = float(brentq(error_at, 0.0, 20.0, xtol=0.01))
    metadata["calibrated_offset_sd_px"] = round(offset, 3)
    typer.echo(f"calibrated markerless offset SD: {offset:.2f} px at adult size")

    pairs = validation_sim.simulate_study(config, pipeline, seed=seed, offset_sd_px=offset)
    pairs.to_csv(target / "pairs.csv", index=False)
    overall = agreement_table(pairs, agreement_config)
    by_group = agreement_table(pairs, agreement_config, by=["maturity_group"])
    overall.to_csv(target / "agreement_overall.csv", index=False)
    by_group.to_csv(target / "agreement_by_maturity_group.csv", index=False)

    replicate_tables = []
    for k in range(int(config["n_replicate_studies"])):
        study = validation_sim.simulate_study(
            config, pipeline, seed=seed + 1 + k, offset_sd_px=offset
        )
        table = agreement_table(study, agreement_config, by=["maturity_group"])
        replicate_tables.append(table.assign(replicate=k))
    replicates = pd.concat(replicate_tables, ignore_index=True)
    replicates.to_csv(target / "replicates.csv", index=False)
    precision = (
        replicates.groupby(["variable", "event", "maturity_group"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_p025=("rmse", lambda v: v.quantile(0.025)),
            rmse_p975=("rmse", lambda v: v.quantile(0.975)),
            bias_mean=("bias", "mean"),
            share_studies_meeting_threshold=("meets_threshold", "mean"),
            n_studies=("rmse", "count"),
        )
        .reset_index()
    )
    precision["rmse_threshold_deg"] = float(agreement_config["rmse_threshold_deg"])
    precision["synthetic"] = True
    precision.to_csv(target / "precision.csv", index=False)

    metadata["seconds"] = round(time.time() - started, 1)
    (target / "run.json").write_text(json.dumps(metadata, indent=2, default=str), "utf-8")
    save_figure(
        plots.bland_altman_grid(pairs, agreement_config), target / "bland_altman.png", metadata
    )
    save_figure(
        plots.rmse_by_stratum(by_group, agreement_config), target / "rmse_by_maturity.png", metadata
    )
    typer.echo(f"\nSYNTHETIC validation study {metadata['run_id']} -> {target}")
    columns = [
        "variable",
        "event",
        "maturity_group",
        "n",
        "bias",
        "loa_low",
        "loa_high",
        "rmse",
        "icc21",
    ]
    typer.echo(by_group[columns].round(2).to_string(index=False))
    typer.echo("\nAcross replicate studies of the same size:")
    typer.echo(precision.round(2).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
