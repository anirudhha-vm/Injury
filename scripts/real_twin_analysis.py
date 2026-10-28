"""Secondary analyses of the primary sequential twin run (M11; REAL data).

Reads the primary run's per-visit scores (``outputs/real_twin_sequential/<run>/scores.csv``) and
adds, without refitting the twin:

1. child-level bootstrap intervals for the headline comparisons;
2. error and bias by maturity timing (early / average / late maturers);
3. calibration of the growth-spurt probability;
4. sensitivity to the reference (smoothing-spline age at PHV);
5. a gradient-boosting baseline evaluated on exactly the same held-out visits and folds, with
   the same cross-fold conformal intervals;
6. the classical SITAR benchmark (``scripts/real_twin_sitar.py``), when a run exists, compared
   on the same rows with the same intervals.

Writes ``outputs/real_twin_analysis/<run_id>/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from scipy import stats

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


def _latest_sitar() -> Path | None:
    root = repo_root() / "outputs" / "real_twin_sitar"
    runs = [p for p in root.glob("*") if (p / "run.json").is_file()] if root.is_dir() else []
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def _with_sitar(merged: pd.DataFrame, run: Path, level: float) -> pd.DataFrame:
    """SITAR estimates on the twin's rows, a growth-spurt score and conformal intervals."""
    sitar = pd.read_csv(run / "predictions.csv")[
        ["athlete_id", "visits", "age", "sitar_aphv", "sitar_status"]
    ]
    out = merged.merge(sitar, on=["athlete_id", "visits"], how="left", suffixes=("", "_sitar"))
    if out["sitar_status"].isna().any() or not np.allclose(out["age"], out["age_sitar"]):
        raise RuntimeError("SITAR rows do not match the twin's held-out visits")
    out["sitar_score"] = -(out["age"] - out["sitar_aphv"]).abs()
    fitted = out["sitar_aphv"].notna()
    half = pd.Series(np.nan, index=out.index)
    half[fitted] = ml_baseline.conformal_halfwidth(out[fitted], "sitar_aphv", level)
    inside = (out["reference_aphv"] - out["sitar_aphv"]).abs() <= half
    out["sitar_conf_covered"] = inside.astype(float).where(out["reference_aphv"].notna() & fitted)
    out["sitar_conf_width"] = 2 * half
    return out


def main(out_dir: Path | None = None) -> None:
    settings = load_config("external/twin_analysis.yaml")
    runs = find_runs(repo_root() / "outputs" / "real_twin_sequential")
    if MAIN not in runs:
        raise typer.BadParameter("No primary sequential twin run found")
    main_run = runs[MAIN]
    scores = pd.read_csv(main_run / "scores.csv")
    source_id = json.loads((main_run / "run.json").read_text(encoding="utf-8"))["run_id"]
    sitar_run = _latest_sitar()
    sitar_id = (
        json.loads((sitar_run / "run.json").read_text(encoding="utf-8"))["run_id"]
        if sitar_run
        else None
    )
    run_config: dict[str, Any] = {"analysis": settings, "source": source_id}
    if sitar_id:
        run_config["sitar_source"] = sitar_id
    metadata = {
        "run_id": f"twinanalysis-{config_hash(run_config)}",
        **run_metadata(seed=int(settings["bootstrap"]["seed"]), config=run_config, synthetic=False),
        "source_run": source_id,
        "sitar_run": sitar_id,
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
    if sitar_run is not None:
        merged = _with_sitar(merged, sitar_run, level)

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
    if sitar_run is not None:
        extra += [
            ("twin_minus_sitar_mae", twin_stats.mae_difference("twin_aphv", "sitar_aphv")),
            ("sitar_minus_moore_mae", twin_stats.mae_difference("sitar_aphv", "moore_aphv")),
            (
                "twin_minus_sitar_circa_auc",
                twin_stats.auc_difference("twin_p_circa", "sitar_score"),
            ),
            ("sitar_mae", twin_stats.mae("sitar_aphv")),
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
    comparators = [("twin_minus_ml_mae", "ml_aphv")]
    if sitar_run is not None:
        comparators.append(("twin_minus_sitar_mae", "sitar_aphv"))
    for k in sorted(merged["visits"].unique()):
        part = merged[merged["visits"] == k]
        for name, other in comparators:
            rows.append(
                {
                    "comparison": name,
                    "subset": f"visit {k}",
                    **twin_stats.cluster_bootstrap(
                        part,
                        twin_stats.mae_difference("twin_aphv", other),
                        n_resamples=int(boot["n_resamples"]),
                        level=float(boot["level"]),
                        seed=int(boot["seed"]),
                    ),
                }
            )
    grouped = merged.assign(timing=merged["athlete_id"].map(groups))
    for group in twin_stats.TIMING_GROUPS:
        part = grouped[grouped["timing"] == group]
        for name, other in [("twin_minus_moore_mae", "moore_aphv"), *comparators]:
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
    # 7. Conditional coverage: by maturity timing and by thirds of the twin's interval width.
    grouped["width_third"] = twin_stats.width_thirds(
        grouped.assign(twin_conf_width=grouped["twin_conf_high"] - grouped["twin_conf_low"]),
        "twin_conf_width",
    )
    covered = ["twin_conf_covered", "ml_conf_covered"]
    if sitar_run is not None:
        covered.append("sitar_conf_covered")
    conditional = []
    for column, labels in (
        ("timing", twin_stats.TIMING_GROUPS),
        ("width_third", twin_stats.WIDTH_THIRDS),
    ):
        for label in labels:
            part = grouped[(grouped[column] == label) & grouped["reference_aphv"].notna()]
            conditional.append(
                {
                    "split": column,
                    "subset": label,
                    "n": len(part),
                    "twin_mae": float((part["twin_aphv"] - part["reference_aphv"]).abs().mean()),
                    "twin_conf_width": float(
                        (part["twin_conf_high"] - part["twin_conf_low"]).mean()
                    ),
                    **{c: float(part[c].astype(float).mean()) for c in covered},
                }
            )
            if sitar_run is None or (column == "width_third" and label == "middle"):
                continue
            rows.append(
                {
                    "comparison": "twin_minus_sitar_coverage",
                    "subset": f"{label} maturers" if column == "timing" else f"{label} intervals",
                    **twin_stats.cluster_bootstrap(
                        part,
                        twin_stats.coverage_difference("twin_conf_covered", "sitar_conf_covered"),
                        n_resamples=int(boot["n_resamples"]),
                        level=float(boot["level"]),
                        seed=int(boot["seed"]),
                    ),
                }
            )
    labelled_grouped = grouped.dropna(subset=["reference_aphv"])
    width_error = stats.spearmanr(
        labelled_grouped["twin_sd"],
        (labelled_grouped["twin_aphv"] - labelled_grouped["reference_aphv"]).abs(),
    )
    adaptivity = {
        "spearman_twin_sd_abs_error": float(width_error.statistic),
        "spearman_p": float(width_error.pvalue),
    }
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

    if sitar_run is not None:
        for column, statistic in (("sitar_mae", "abs"), ("sitar_bias", "signed")):
            timing[column] = timing["timing"].map(
                ml_by_group.dropna(subset=["sitar_aphv"])
                .groupby("timing")
                .apply(
                    lambda g, kind=statistic: (
                        (g["sitar_aphv"] - g["reference_aphv"]).abs().mean()
                        if kind == "abs"
                        else (g["sitar_aphv"] - g["reference_aphv"]).mean()
                    ),
                    include_groups=False,
                )
            )

    # 3. Calibration of the growth-spurt probability.
    circa = twin_stats.circa_calibration(merged)
    labelled = merged.dropna(subset=["circa_now"])
    circa["auc_twin"] = metrics.auc(labelled["circa_now"].astype(float), labelled["twin_p_circa"])
    circa["auc_ml"] = metrics.auc(labelled["circa_now"].astype(float), labelled["ml_score"])
    if sitar_run is not None:
        circa["auc_sitar"] = metrics.auc(
            labelled["circa_now"].astype(float), labelled["sitar_score"].fillna(-np.inf)
        )
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
    sitar_summary: dict[str, Any] = {}
    if sitar_run is not None:
        status = merged["sitar_status"]
        labelled_rows = merged.dropna(subset=["reference_aphv"])
        common = labelled_rows.dropna(subset=["sitar_aphv"])
        sitar_summary = {
            "n_rows": len(merged),
            "n_failed": int(merged["sitar_aphv"].isna().sum()),
            "n_edge": int((status == "edge").sum()),
            "failure_rate": float(merged["sitar_aphv"].isna().mean()),
            "sitar_mae": float((common["sitar_aphv"] - common["reference_aphv"]).abs().mean()),
            "twin_mae_same_rows": float(
                (common["twin_aphv"] - common["reference_aphv"]).abs().mean()
            ),
            "sitar_bias": float((common["sitar_aphv"] - common["reference_aphv"]).mean()),
            "sitar_conf_coverage": float(merged["sitar_conf_covered"].dropna().mean()),
            "sitar_conf_width": float(merged["sitar_conf_width"].mean()),
        }
        both = with_spline.dropna(subset=["sitar_aphv"])
        sitar_summary["sitar_mae_vs_spline"] = float(
            (both["sitar_aphv"] - both["spline_aphv"]).abs().mean()
        )
        sitar_summary["twin_mae_vs_spline_same_rows"] = float(
            (both["twin_aphv"] - both["spline_aphv"]).abs().mean()
        )
    ml_by_visits = (
        merged.dropna(subset=["reference_aphv"])
        .groupby("visits")
        .apply(
            lambda g: pd.Series(
                {
                    "twin_mae": (g["twin_aphv"] - g["reference_aphv"]).abs().mean(),
                    "ml_mae": (g["ml_aphv"] - g["reference_aphv"]).abs().mean(),
                    "moore_mae": (g["moore_aphv"] - g["reference_aphv"]).abs().mean(),
                    **(
                        {"sitar_mae": (g["sitar_aphv"] - g["reference_aphv"]).abs().mean()}
                        if "sitar_aphv" in g
                        else {}
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    intervals.to_csv(target / "intervals.csv", index=False)
    timing.to_csv(target / "timing_groups.csv", index=False)
    pd.DataFrame(conditional).to_csv(target / "conditional_coverage.csv", index=False)
    ml_by_visits.to_csv(target / "ml_by_visits.csv", index=False)
    merged.to_csv(target / "scores_with_ml.csv", index=False)
    summary = {
        **metadata,
        "circa_calibration": circa,
        "reference_sensitivity": sensitivity,
        "ml_baseline": ml_summary,
        "sitar_benchmark": sitar_summary,
        "interval_adaptivity": adaptivity,
        "synthetic": False,
    }
    (target / "run.json").write_text(json.dumps(summary, indent=2, default=float), "utf-8")
    typer.echo(f"REAL-DATA twin analysis {metadata['run_id']} -> {target}")
    typer.echo(intervals.round(3).to_string(index=False))
    typer.echo(timing.round(3).to_string(index=False))
    typer.echo(pd.DataFrame(conditional).round(3).to_string(index=False))
    typer.echo(json.dumps(adaptivity))
    typer.echo(ml_by_visits.round(3).to_string(index=False))
    typer.echo(
        json.dumps(
            {"circa": circa, "sensitivity": sensitivity, "ml": ml_summary, "sitar": sitar_summary},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    typer.run(main)
