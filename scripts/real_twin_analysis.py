"""Secondary analyses of the primary sequential twin run (M11; REAL data).

Reads the primary run's per-visit scores (``outputs/real_twin_sequential/<run>/scores.csv``) and
adds, without refitting the twin:

1. child-level bootstrap intervals for the headline comparisons;
2. error and bias by maturity timing (early / average / late maturers);
3. calibration of the growth-spurt probability;
4. sensitivity to the reference (smoothing-spline age at PHV);
5. a gradient-boosting baseline evaluated on exactly the same held-out visits and folds, with
   the same cross-fold conformal intervals.

Writes ``outputs/real_twin_analysis/<run_id>/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from youthtwin.eval import metrics
from youthtwin.external import berkeley, ml_baseline, reference_aphv, sequential, twin_stats
from youthtwin.paper.twin_paper import MAIN, find_runs
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.provenance import config_hash, run_metadata

DATA = Path("data/external/sitar/sitar/data/berkeley.rda")


def _ml_predictions(scores: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Gradient-boosting predictions on the primary run's held-out sequences (same seeds)."""
    config = load_config("external/sequential.yaml")
    config["scenario"] = {"visit_step": 1, "added_noise_cm": 0.0, "sitting_height": True}
    frame_all = berkeley.load_berkeley(repo_root() / DATA)
    reference = reference_aphv.reference_table(
        frame_all, load_config("external/berkeley.yaml")["reference"]
    )
    truth = dict(
        zip(
            reference.loc[reference["included"], "athlete_id"],
            reference.loc[reference["included"], "reference_aphv"],
            strict=True,
        )
    )
    parts = []
    for sex in ("female", "male"):
        frame = frame_all[frame_all["sex"] == sex]
        folds = sequential.assign_folds(
            sorted(frame["athlete_id"].unique()), int(config["n_folds"]), int(config["seed"])
        )
        for fold in range(int(config["n_folds"])):
            seed = int(config["seed"]) + 100 * fold + (0 if sex == "female" else 50)
            train = [a for a, k in folds.items() if k != fold]
            held_out = [a for a, k in folds.items() if k == fold]
            training = ml_baseline.training_set(
                frame,
                train,
                truth,
                sex,  # type: ignore[arg-type]
                config,
                entries=int(settings["entries_per_child"]),
                seed=seed + 1000,
            )
            sequences = sequential.entry_sequences(frame, held_out, config, seed + 1)
            predicted = ml_baseline.fit_predict(
                training,
                sequences,
                sex,
                config,
                settings,
                seed=seed,  # type: ignore[arg-type]
            )
            parts.append(predicted)
    return pd.concat(parts, ignore_index=True)


def main(out_dir: Path | None = None) -> None:
    settings = load_config("external/twin_analysis.yaml")
    runs = find_runs(repo_root() / "outputs" / "real_twin_sequential")
    if MAIN not in runs:
        raise typer.BadParameter("No primary sequential twin run found")
    main_run = runs[MAIN]
    scores = pd.read_csv(main_run / "scores.csv")
    source_id = json.loads((main_run / "run.json").read_text(encoding="utf-8"))["run_id"]
    run_config = {"analysis": settings, "source": source_id}
    metadata = {
        "run_id": f"twinanalysis-{config_hash(run_config)}",
        **run_metadata(seed=int(settings["bootstrap"]["seed"]), config=run_config, synthetic=False),
        "source_run": source_id,
    }
    target = out_dir or repo_root() / "outputs" / "real_twin_analysis" / metadata["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    boot = settings["bootstrap"]

    # 5. ML baseline on the same rows (needed before the intervals that compare with it).
    ml = _ml_predictions(scores, settings["ml_baseline"])
    merged = scores.merge(ml, on=["athlete_id", "visits"], how="left", suffixes=("", "_ml"))
    if merged["ml_aphv"].isna().any() or not np.allclose(merged["age"], merged["age_ml"]):
        raise RuntimeError("ML baseline rows do not match the twin's held-out visits")
    level = float(load_config("external/sequential.yaml")["interval_level"])
    half = ml_baseline.conformal_halfwidth(merged, "ml_aphv", level)
    inside = (merged["reference_aphv"] - merged["ml_aphv"]).abs() <= half
    merged["ml_conf_covered"] = inside.astype(float).where(merged["reference_aphv"].notna())
    merged["ml_conf_width"] = 2 * half
    merged["ml_score"] = -(merged["age"] - merged["ml_aphv"]).abs()

    # 1. Intervals for the headline comparisons.
    intervals = twin_stats.headline_intervals(
        merged,
        n_resamples=int(boot["n_resamples"]),
        level=float(boot["level"]),
        seed=int(boot["seed"]),
        visits=sorted(merged["visits"].unique()),
    )
    extra = [
        ("twin_minus_ml_mae", twin_stats.mae_difference("twin_aphv", "ml_aphv")),
        ("ml_minus_moore_mae", twin_stats.mae_difference("ml_aphv", "moore_aphv")),
        ("twin_minus_ml_circa_auc", twin_stats.auc_difference("twin_p_circa", "ml_score")),
        ("ml_mae", twin_stats.mae("ml_aphv")),
    ]
    rows = [
        {
            "comparison": name,
            "subset": "all",
            **twin_stats.cluster_bootstrap(
                merged,
                statistic,
                n_resamples=int(boot["n_resamples"]),
                level=float(boot["level"]),
                seed=int(boot["seed"]),
            ),
        }
        for name, statistic in extra
    ]
    # 2. Maturity timing groups.
    frame_all = berkeley.load_berkeley(repo_root() / DATA)
    reference = reference_aphv.reference_table(
        frame_all, load_config("external/berkeley.yaml")["reference"]
    )
    included = reference[reference["included"]].reset_index(drop=True)
    groups = dict(
        zip(
            included["athlete_id"],
            twin_stats.timing_groups(included, float(settings["timing_group_width_sd"])),
            strict=True,
        )
    )
    # Intervals for the twin-versus-ML comparison by visit and for both comparisons by timing.
    for k in sorted(merged["visits"].unique()):
        part = merged[merged["visits"] == k]
        rows.append(
            {
                "comparison": "twin_minus_ml_mae",
                "subset": f"visit {k}",
                **twin_stats.cluster_bootstrap(
                    part,
                    twin_stats.mae_difference("twin_aphv", "ml_aphv"),
                    n_resamples=int(boot["n_resamples"]),
                    level=float(boot["level"]),
                    seed=int(boot["seed"]),
                ),
            }
        )
    grouped = merged.assign(timing=merged["athlete_id"].map(groups))
    for group in twin_stats.TIMING_GROUPS:
        part = grouped[grouped["timing"] == group]
        for name, other in (
            ("twin_minus_moore_mae", "moore_aphv"),
            ("twin_minus_ml_mae", "ml_aphv"),
        ):
            rows.append(
                {
                    "comparison": name,
                    "subset": f"{group} maturers",
                    **twin_stats.cluster_bootstrap(
                        part,
                        twin_stats.mae_difference("twin_aphv", other),
                        n_resamples=int(boot["n_resamples"]),
                        level=float(boot["level"]),
                        seed=int(boot["seed"]),
                    ),
                }
            )
    intervals = pd.concat([intervals, pd.DataFrame(rows)], ignore_index=True)
    timing = twin_stats.by_timing_group(merged, groups)
    ml_by_group = merged.assign(timing=merged["athlete_id"].map(groups)).dropna(
        subset=["timing", "reference_aphv"]
    )
    timing["ml_mae"] = timing["timing"].map(
        ml_by_group.groupby("timing").apply(
            lambda g: (g["ml_aphv"] - g["reference_aphv"]).abs().mean(), include_groups=False
        )
    )
    timing["ml_bias"] = timing["timing"].map(
        ml_by_group.groupby("timing").apply(
            lambda g: (g["ml_aphv"] - g["reference_aphv"]).mean(), include_groups=False
        )
    )

    # 3. Calibration of the growth-spurt probability.
    circa = twin_stats.circa_calibration(merged)
    labelled = merged.dropna(subset=["circa_now"])
    circa["auc_twin"] = metrics.auc(labelled["circa_now"].astype(float), labelled["twin_p_circa"])
    circa["auc_ml"] = metrics.auc(labelled["circa_now"].astype(float), labelled["ml_score"])
    circa["auc_moore"] = metrics.auc(
        labelled["circa_now"].astype(float), -labelled["moore_offset_now"].abs()
    )

    # 4. Reference sensitivity: smoothing-spline age at PHV.
    spline = dict(zip(included["athlete_id"], included["spline_aphv"], strict=True))
    with_spline = merged.assign(spline_aphv=merged["athlete_id"].map(spline)).dropna(
        subset=["spline_aphv"]
    )
    sensitivity = {
        "n": len(with_spline),
        "twin_mae_vs_spline": float(
            (with_spline["twin_aphv"] - with_spline["spline_aphv"]).abs().mean()
        ),
        "moore_mae_vs_spline": float(
            (with_spline["moore_aphv"] - with_spline["spline_aphv"]).abs().mean()
        ),
        "ml_mae_vs_spline": float(
            (with_spline["ml_aphv"] - with_spline["spline_aphv"]).abs().mean()
        ),
    }
    ml_summary = {
        "ml_mae": float((merged["ml_aphv"] - merged["reference_aphv"]).abs().mean()),
        "ml_conf_coverage": float(merged["ml_conf_covered"].dropna().astype(float).mean()),
        "ml_conf_width": float(merged["ml_conf_width"].mean()),
        "twin_conf_width": float((merged["twin_conf_high"] - merged["twin_conf_low"]).mean()),
    }
    ml_by_visits = (
        merged.dropna(subset=["reference_aphv"])
        .groupby("visits")
        .apply(
            lambda g: pd.Series(
                {
                    "twin_mae": (g["twin_aphv"] - g["reference_aphv"]).abs().mean(),
                    "ml_mae": (g["ml_aphv"] - g["reference_aphv"]).abs().mean(),
                    "moore_mae": (g["moore_aphv"] - g["reference_aphv"]).abs().mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    intervals.to_csv(target / "intervals.csv", index=False)
    timing.to_csv(target / "timing_groups.csv", index=False)
    ml_by_visits.to_csv(target / "ml_by_visits.csv", index=False)
    merged.to_csv(target / "scores_with_ml.csv", index=False)
    summary = {
        **metadata,
        "circa_calibration": circa,
        "reference_sensitivity": sensitivity,
        "ml_baseline": ml_summary,
        "synthetic": False,
    }
    (target / "run.json").write_text(json.dumps(summary, indent=2, default=float), "utf-8")
    typer.echo(f"REAL-DATA twin analysis {metadata['run_id']} -> {target}")
    typer.echo(intervals.round(3).to_string(index=False))
    typer.echo(timing.round(3).to_string(index=False))
    typer.echo(ml_by_visits.round(3).to_string(index=False))
    typer.echo(
        json.dumps(
            {"circa": circa, "sensitivity": sensitivity, "ml": ml_summary}, indent=2, default=float
        )
    )


if __name__ == "__main__":
    typer.run(main)
