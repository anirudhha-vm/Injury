"""Calibrate the synthetic injury process to a published incidence (D-120).

Finds the growth-related hazard intercept at which the expected proportion of athletes with at
least one growth-related injury per season equals the target in ``configs/sim/calibration.yaml``.
The expectation is computed exactly from the generator's true weekly hazards: changing the
intercept by ``d`` shifts every weekly hazard by ``d`` on the logit scale, so one simulated
cohort and a root-finder suffice, without sampling noise. Time-loss weeks are ignored, which
overstates the proportion very slightly. Writes ``outputs/calibration/<run_id>/calibration.json``;
the value is then copied into ``configs/sim/cohort.yaml`` with that run ID.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.optimize import brentq
from scipy.special import expit, logit

from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata


def expected_proportion(weekly: pd.DataFrame, column: str, shift: float) -> float:
    """Mean over athletes of P(at least one event in follow-up) after a logit shift."""
    hazard = expit(logit(weekly[column].to_numpy()) + shift)
    log_survive = pd.Series(np.log1p(-hazard)).groupby(weekly["athlete_id"].to_numpy()).sum()
    return float(np.mean(1.0 - np.exp(log_survive.to_numpy())))


def main(out_dir: Path | None = None) -> None:
    settings = load_config("sim/calibration.yaml")
    config = load_config("sim/cohort.yaml")
    config["n_athletes"] = int(settings["n_athletes"])
    group = str(settings["outcome_group"])
    target = float(settings["target_proportion"])
    weekly = generate_cohort(config).truth["athlete_weeks_truth"]
    column = f"hazard_{group}"
    current = float(config["injury"][group]["intercept"])
    shift = brentq(lambda d: expected_proportion(weekly, column, d) - target, -10.0, 10.0)
    chosen = round(current + shift, 3)
    run_config = {"calibration": settings, "cohort": config}
    record = {
        "run_id": f"calibration-{config_hash(run_config)}",
        **run_metadata(seed=int(config["seed"]), config=run_config, synthetic=True),
        "outcome_group": group,
        "target_proportion": target,
        "intercept_before": current,
        "intercept": chosen,
        "proportion_before": expected_proportion(weekly, column, 0.0),
        "proportion_after": expected_proportion(weekly, column, chosen - current),
    }
    target_dir = out_dir or repo_root() / "outputs" / "calibration" / record["run_id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "calibration.json").write_text(json.dumps(record, indent=2), "utf-8")
    typer.echo(json.dumps({k: record[k] for k in list(record)[-5:]}, indent=2))
    typer.echo(f"{group} intercept {chosen} ({record['run_id']})")


if __name__ == "__main__":
    typer.run(main)
