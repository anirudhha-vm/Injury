"""External check of the growth twin on the Chard Growth Study (sitar ``heights``; REAL data).

12 girls measured twice a year from 8 to 16 y, with age at menarche (right-censored for some),
no sitting height. The reference curve is learned from all Berkeley girls (MAP, stature only);
each Chard girl's twin is then updated visit by visit exactly as in the Berkeley evaluation and
compared with (a) the age at PHV from her own full record, where the record spans the spurt, and
(b) her age at menarche, an independent maturity marker that typically follows PHV by about a
year. Writes ``outputs/real_twin_external_chard/<run_id>/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from omegaconf import OmegaConf

from youthtwin.external import berkeley, reference_aphv, sequential
from youthtwin.growth import sitar_bayes
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata

CHARD = Path("data/external/sitar/sitar/data/heights.rda")
BERKELEY = Path("data/external/sitar/sitar/data/berkeley.rda")


def load_chard(path: Path) -> pd.DataFrame:
    import pyreadr

    raw = next(iter(pyreadr.read_r(str(path)).values()))
    return pd.DataFrame(
        {
            "athlete_id": "chard-" + raw["id"].astype(str),
            "sex": "female",
            "age": raw["age"].astype(float),
            "stature": raw["height"].astype(float),
            "sitting_height": np.nan,  # not measured
            "mass": np.nan,
            "menarche": raw["men"].astype(float),  # negative = right-censored at |value|
            "synthetic": False,
        }
    )


def main(draws: int = 200, out_dir: Path | None = None) -> None:
    config = load_config("external/sequential.yaml")
    config["use_leg"] = False
    config["scenario"] = {"sitting_height": False}
    merged = OmegaConf.merge(
        load_config("growth/sitar.yaml"),
        config["model_overrides"],
        {"bivariate": {"enabled": False}},
    )
    model_config: dict[str, Any] = OmegaConf.to_container(merged)  # type: ignore[assignment]
    model_config["inference"].update(method="map", draws=draws)
    run_config = {"sequential": config, "model": model_config}
    metadata = {
        "run_id": f"chard-{config_hash(run_config)}",
        **run_metadata(seed=int(config["seed"]), config=run_config, synthetic=False),
        "dataset": "Chard Growth Study (sitar heights) with Berkeley girls as reference",
    }
    target = out_dir or repo_root() / "outputs" / "real_twin_external_chard" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)

    berkeley_frame = berkeley.load_berkeley(repo_root() / BERKELEY)
    girls = berkeley_frame[berkeley_frame["sex"] == "female"]
    athletes = sorted(girls["athlete_id"].unique())
    windows = berkeley.draw_windows(
        girls, athletes, load_config("external/berkeley.yaml")["window"], int(config["seed"])
    )
    low, high = (float(v) for v in config["train_age_range_years"])
    rows = sequential.complete(girls)
    rows = rows[rows["age"].between(low, high) & rows["athlete_id"].isin(windows["athlete_id"])]
    reference_data = sitar_bayes.growth_data_from_measurements(
        rows, "female", synthetic=False, prior_frame=windows, stature_only_prior=True
    )
    idata = sitar_bayes.fit(reference_data, model_config, seed=int(config["seed"]))
    shared = sitar_bayes.shared_draws(idata, int(config["n_posterior_draws"]), 0, model_config)

    chard = load_chard(repo_root() / CHARD)
    reference = reference_aphv.reference_table(
        chard, load_config("external/berkeley.yaml")["reference"]
    )
    truth = dict(
        zip(
            reference.loc[reference["included"], "athlete_id"],
            reference.loc[reference["included"], "reference_aphv"],
            strict=True,
        )
    )
    # Chard has no sitting height: sequences need only stature (``complete`` would drop all).
    sequences: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(int(config["seed"]))
    for athlete, record in chard.sort_values(["athlete_id", "age"]).groupby("athlete_id"):
        starts = record.loc[record["age"].between(8.0, 13.0), "age"].to_numpy()
        if starts.size == 0:
            continue
        entry = float(rng.choice(starts))
        visits = record[record["age"] >= entry - 1e-9].head(int(config["max_visits"]) + 1)
        if len(visits) >= 2:
            sequences[str(athlete)] = visits.reset_index(drop=True)
    scores = sequential.score_sequences(
        shared, sequences, truth, "female", model_config, config, seed=int(config["seed"])
    )
    menarche = chard.groupby("athlete_id")["menarche"].first()
    final = scores.sort_values("visits").groupby("athlete_id").tail(1).copy()
    final["menarche"] = final["athlete_id"].map(menarche)
    final["menarche_observed"] = final["menarche"] > 0
    observed = final[final["menarche_observed"]]
    summary = {
        "n_girls": int(chard["athlete_id"].nunique()),
        "n_with_reference_aphv": len(truth),
        "n_with_menarche": int(observed.shape[0]),
        "twin_mae_vs_reference_final_visit": float(
            (final["twin_aphv"] - final["reference_aphv"]).abs().mean()
        ),
        "moore_mae_vs_reference_final_visit": float(
            (final["moore_aphv"] - final["reference_aphv"]).abs().mean()
        ),
        "menarche_minus_twin_aphv_mean": float(
            (observed["menarche"] - observed["twin_aphv"]).mean()
        ),
        "menarche_minus_twin_aphv_sd": float((observed["menarche"] - observed["twin_aphv"]).std()),
        "menarche_minus_moore_aphv_sd": float(
            (observed["menarche"] - observed["moore_aphv"]).std()
        ),
        "corr_twin_menarche": float(np.corrcoef(observed["twin_aphv"], observed["menarche"])[0, 1]),
        "corr_moore_menarche": float(
            np.corrcoef(observed["moore_aphv"], observed["menarche"])[0, 1]
        ),
        "synthetic": False,
    }
    scores.to_csv(target / "scores.csv", index=False)
    final.to_csv(target / "final_visit.csv", index=False)
    reference.to_csv(target / "reference.csv", index=False)
    pd.DataFrame([summary]).to_csv(target / "summary.csv", index=False)
    (target / "run.json").write_text(json.dumps(metadata, indent=2, default=str), "utf-8")
    typer.echo(f"REAL-DATA external check {metadata['run_id']} -> {target}")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
