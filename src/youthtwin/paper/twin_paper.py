"""LaTeX values and tables for the growth-twin manuscript (Path A, M11), generated from runs.

``paper/ieee_twin/main.tex`` contains prose and macros only. Runs are selected by the
``settings`` block of each ``outputs/real_twin_sequential/<run_id>/run.json``; a value whose run
does not exist prints as a red ``[pending]`` marker (``ieee.PENDING``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.external import sequential
from youthtwin.paper.ieee import PENDING, Values, _number, percent, tabular

MAIN = "main"
VARIANTS = ("main", "yearly", "noise", "nositting", "leg_shared", "leg_lag", "nuts_fold0")
VARIANT_LABEL = {
    "main": "Six-monthly visits (primary)",
    "yearly": "Yearly visits",
    "noise": "Extra measurement error (1 cm)",
    "nositting": "No sitting height",
    "leg_shared": "Leg length, shared timing",
    "leg_lag": "Leg length, individual lag",
}
BANDS = ("more than 3 before", "3 to 1 before", "within 1", "1 to 3 after", "more than 3 after")


def classify(record: Mapping[str, Any]) -> str | None:
    """Which variant a sequential-twin run is, from its recorded settings (None: unlabelled)."""
    settings = record.get("settings")
    if not settings:
        return None
    scenario = settings.get("scenario", {})
    if settings.get("method") == "nuts":
        return "nuts_fold0" if settings.get("fold") == 0 else None
    if settings.get("fold") is not None:
        return None
    if settings.get("leg"):
        return "leg_lag" if settings.get("leg_lag") else "leg_shared"
    if int(scenario.get("visit_step", 1)) == 2:
        return "yearly"
    if float(scenario.get("added_noise_cm", 0.0)) > 0:
        return "noise"
    if not scenario.get("sitting_height", True):
        return "nositting"
    return MAIN


def find_runs(root: Path) -> dict[str, Path]:
    """The latest run of each variant under ``outputs/real_twin_sequential``."""
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for run in sorted(root.glob("*/run.json"), key=lambda p: p.stat().st_mtime):
        if not (run.parent / "scores.csv").is_file():
            continue
        variant = classify(json.loads(run.read_text(encoding="utf-8")))
        if variant is not None:
            found[variant] = run.parent
    return found


def _rid(run: Path) -> str:
    return str(json.loads((run / "run.json").read_text(encoding="utf-8"))["run_id"])


def _scores(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "scores.csv")


def _overall(scores: pd.DataFrame) -> pd.Series:
    return sequential.summarise(scores.assign(all="all"), ["all"]).iloc[0]


def main_values(run: Path | None, values: Values) -> tuple[str, str]:
    names = [
        "TwChildren",
        "TwEvaluations",
        "TwFolds",
        "TwMAE",
        "TwMooreMAE",
        "TwMirwaldMAE",
        "TwMAEFull",
        "TwMooreMAEFull",
        "TwBiasPB",
        "TwBiasFull",
        "TwCovRaw",
        "TwCovConf",
        "TwWidthRaw",
        "TwWidthConf",
        "TwCircaAUC",
        "TwCircaAUCMoore",
        "TwCircaSensTwin",
        "TwCircaSpecTwin",
        "TwCircaSensMoore",
        "TwCircaSpecMoore",
        "TwCircaPrev",
        "TwForecastMAE",
        "TwNaiveMAE",
        "TwForecastCovRaw",
        "TwForecastCovConf",
        "TwForecastWidthConf",
        "TwMAEFirst",
        "TwMAELast",
        "TwMooreFirst",
        "TwMooreLast",
        "TwSlope",
        "TwSigmaBeta",
        "TwMaxVisits",
        "TwBands",
        "TwBandsBetter",
        "TwWorstBand",
        "TwWorstBandTwin",
        "TwWorstBandMoore",
        "TwForecastMAESame",
    ]
    if run is None:
        values.pending(*names)
        return (PENDING, PENDING)
    rid = _rid(run)
    values.runs["twin_main"] = rid
    scores = _scores(run)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    o = _overall(scores)
    values.number("TwChildren", int(scores["athlete_id"].nunique()), rid)
    values.number("TwEvaluations", int(o["n"]), rid)
    values.number("TwFolds", int(scores["fold"].nunique()), rid)
    values.number("TwMaxVisits", int(scores["visits"].max()), rid)
    for key, name in (
        ("twin_mae", "TwMAE"),
        ("moore_mae", "TwMooreMAE"),
        ("mirwald_mae", "TwMirwaldMAE"),
        ("twin_mae_vs_full", "TwMAEFull"),
        ("moore_mae_vs_full", "TwMooreMAEFull"),
        ("twin_bias", "TwBiasPB"),
        ("twin_bias_vs_full", "TwBiasFull"),
        ("twin_width", "TwWidthRaw"),
        ("twin_conf_width", "TwWidthConf"),
        ("forecast_mae", "TwForecastMAE"),
        ("twin_mae_same_rows", "TwForecastMAESame"),
        ("naive_mae", "TwNaiveMAE"),
        ("forecast_conf_width", "TwForecastWidthConf"),
        ("circa_auc_twin", "TwCircaAUC"),
        ("circa_auc_moore", "TwCircaAUCMoore"),
    ):
        values.number(name, float(o[key]), rid)
    for key, name in (
        ("twin_coverage", "TwCovRaw"),
        ("twin_conf_coverage", "TwCovConf"),
        ("forecast_coverage", "TwForecastCovRaw"),
        ("forecast_conf_coverage", "TwForecastCovConf"),
        ("circa_twin_sensitivity", "TwCircaSensTwin"),
        ("circa_twin_specificity", "TwCircaSpecTwin"),
        ("circa_moore_sensitivity", "TwCircaSensMoore"),
        ("circa_moore_specificity", "TwCircaSpecMoore"),
        ("circa_prevalence", "TwCircaPrev"),
    ):
        values.put(name, percent(float(o[key])), rid)
    by_visits = sequential.summarise(scores, ["visits"])
    first, last = by_visits.iloc[0], by_visits.iloc[-1]
    values.number("TwMAEFirst", float(first["twin_mae"]), rid)
    values.number("TwMAELast", float(last["twin_mae"]), rid)
    values.number("TwMooreFirst", float(first["moore_mae"]), rid)
    values.number("TwMooreLast", float(last["moore_mae"]), rid)
    fits = record.get("fits", [])
    slopes = [f["sigma_beta_slope_mean"] for f in fits if "sigma_beta_slope_mean" in f]
    values.number("TwSlope", float(np.mean(slopes)) if slopes else float("nan"), rid)
    values.number("TwSigmaBeta", float(np.mean([f["sigma_beta_mean"] for f in fits])), rid)

    per_visit = sequential.summarise(scores, ["sex", "visits"])
    rows: list[list[str]] = []
    groups: list[int] = []
    for sex, part in per_visit.groupby("sex", sort=True):
        groups.append(len(rows))
        for _, r in part.iterrows():
            rows.append(
                [
                    str(sex).capitalize() if r["visits"] == part["visits"].min() else "",
                    _number(int(r["visits"]), 0),
                    _number(int(r["n"]), 0),
                    _number(float(r["twin_mae"]), 2),
                    _number(float(r["moore_mae"]), 2),
                    _number(float(r["mirwald_mae"]), 2),
                    percent(float(r["twin_coverage"])),
                    percent(float(r["twin_conf_coverage"])),
                    _number(float(r["twin_conf_width"]), 2),
                ]
            )
    visits_table = tabular(
        "lrrrrrrrr",
        [
            "Sex",
            "Visits",
            "$n$",
            "Twin",
            "Moore",
            "Mirwald",
            "Raw cov.",
            "Conf. cov.",
            "Conf. width",
        ],
        rows,
        groups,
    )
    scores = scores.copy()
    by_band = sequential.summarise(scores, ["band"])
    known = by_band[by_band["band"].isin(BANDS)]
    values.number("TwBands", len(known), rid)
    values.number("TwBandsBetter", int((known["twin_mae"] < known["moore_mae"]).sum()), rid)
    worst = known.iloc[int((known["twin_mae"] - known["moore_mae"]).to_numpy().argmax())]
    values.put("TwWorstBand", str(worst["band"]), rid)
    values.number("TwWorstBandTwin", float(worst["twin_mae"]), rid)
    values.number("TwWorstBandMoore", float(worst["moore_mae"]), rid)
    band_rows = []
    for band in BANDS:
        match = by_band[by_band["band"] == band]
        if match.empty:
            continue
        r = match.iloc[0]
        band_rows.append(
            [
                band.replace("more than", "$>$").replace(" to ", "--").capitalize(),
                _number(int(r["n"]), 0),
                _number(float(r["twin_mae"]), 2),
                _number(float(r["moore_mae"]), 2),
                percent(float(r["twin_coverage"])),
                percent(float(r["twin_conf_coverage"])),
            ]
        )
    band_table = tabular(
        "lrrrrr", ["Years from PHV", "$n$", "Twin", "Moore", "Raw cov.", "Conf. cov."], band_rows
    )
    return visits_table, band_table


def variant_values(runs: Mapping[str, Path], values: Values) -> str:
    rows = []
    for variant, label in VARIANT_LABEL.items():
        run = runs.get(variant)
        tag = variant.title().replace("_", "")
        if run is None:
            values.pending(f"Tw{tag}MAE", f"Tw{tag}Moore", f"Tw{tag}Cov", f"Tw{tag}AUC")
            rows.append([label, PENDING, "", "", "", ""])
            continue
        rid = _rid(run)
        values.runs[f"twin_{variant}"] = rid
        o = _overall(_scores(run))
        values.number(f"Tw{tag}MAE", float(o["twin_mae"]), rid)
        values.number(f"Tw{tag}Moore", float(o["moore_mae"]), rid)
        values.put(f"Tw{tag}Cov", percent(float(o["twin_conf_coverage"])), rid)
        values.number(f"Tw{tag}AUC", float(o["circa_auc_twin"]), rid)
        rows.append(
            [
                label,
                _number(float(o["twin_mae"]), 2),
                _number(float(o["moore_mae"]), 2),
                percent(float(o["twin_coverage"])),
                percent(float(o["twin_conf_coverage"])),
                _number(float(o["circa_auc_twin"]), 2),
            ]
        )
    return tabular(
        "lrrrrr",
        ["Setting", "Twin MAE", "Moore MAE", "Raw cov.", "Conf. cov.", "Spurt AUC"],
        rows,
        groups=[4],
    )


def nuts_values(runs: Mapping[str, Path], values: Values) -> None:
    names = ["TwNutsMAE", "TwMapFoldMAE", "TwNutsCov", "TwMapFoldCov", "TwNutsRhat", "TwNutsESS"]
    nuts, main = runs.get("nuts_fold0"), runs.get(MAIN)
    if nuts is None or main is None:
        values.pending(*names)
        return
    rid = _rid(nuts)
    values.runs["twin_nuts_fold0"] = rid
    a = _overall(_scores(nuts))
    main_scores = _scores(main)
    b = _overall(main_scores[main_scores["fold"] == 0])
    values.number("TwNutsMAE", float(a["twin_mae"]), rid)
    values.number("TwMapFoldMAE", float(b["twin_mae"]), _rid(main))
    values.put("TwNutsCov", percent(float(a["twin_coverage"])), rid)
    values.put("TwMapFoldCov", percent(float(b["twin_coverage"])), _rid(main))
    fits = json.loads((nuts / "run.json").read_text(encoding="utf-8"))["fits"]
    values.number("TwNutsRhat", max(float(f["max_rhat"]) for f in fits), rid)
    values.number("TwNutsESS", int(min(float(f["min_ess_bulk"]) for f in fits)), rid)


def chard_values(run: Path | None, values: Values) -> None:
    names = [
        "ChGirls",
        "ChRef",
        "ChMen",
        "ChTwinMAE",
        "ChMooreMAE",
        "ChCorrTwin",
        "ChCorrMoore",
        "ChLagMean",
        "ChLagSD",
        "ChMooreLagSD",
    ]
    if run is None:
        values.pending(*names)
        return
    rid = _rid(run)
    values.runs["twin_chard"] = rid
    s = pd.read_csv(run / "summary.csv").iloc[0]
    values.number("ChGirls", int(s["n_girls"]), rid)
    values.number("ChRef", int(s["n_with_reference_aphv"]), rid)
    values.number("ChMen", int(s["n_with_menarche"]), rid)
    values.number("ChTwinMAE", float(s["twin_mae_vs_reference_final_visit"]), rid)
    values.number("ChMooreMAE", float(s["moore_mae_vs_reference_final_visit"]), rid)
    values.number("ChCorrTwin", float(s["corr_twin_menarche"]), rid)
    values.number("ChCorrMoore", float(s["corr_moore_menarche"]), rid)
    values.number("ChLagMean", float(s["menarche_minus_twin_aphv_mean"]), rid)
    values.number("ChLagSD", float(s["menarche_minus_twin_aphv_sd"]), rid)
    values.number("ChMooreLagSD", float(s["menarche_minus_moore_aphv_sd"]), rid)


def build(
    root: Path,
    chard: Path | None,
    configs: Mapping[str, Any],
    analysis: Path | None = None,
) -> tuple[Values, dict[str, str], dict[str, Path]]:
    """All macros and tables. ``root``: outputs/real_twin_sequential."""
    values = Values()
    seq = configs["sequential"]
    values.put(
        "SpecInterval", percent(float(seq["interval_level"])), "configs/external/sequential.yaml"
    )
    values.put(
        "SpecCircaWidth",
        _number(float(seq["circa_phv_half_width_years"]), 0),
        "configs/external/sequential.yaml",
    )
    values.put(
        "SpecHorizon",
        _number(float(seq["forecast_horizon_years"]) * 12, 0),
        "configs/external/sequential.yaml",
    )
    runs = find_runs(root)
    visits_table, band_table = main_values(runs.get(MAIN), values)
    tables = {
        "twin_tab_visits.tex": visits_table,
        "twin_tab_bands.tex": band_table,
        "twin_tab_variants.tex": variant_values(runs, values),
    }
    nuts_values(runs, values)
    chard_values(chard, values)
    tables.update(analysis_values(analysis, values))
    timing_sd = float(configs.get("analysis", {}).get("timing_group_width_sd", float("nan")))
    values.put("SpecTimingSD", _number(timing_sd, 0), "configs/external/twin_analysis.yaml")
    return values, tables, runs


def _signed(value: float, digits: int = 2) -> str:
    return _number(float(value), digits)


def analysis_values(run: Path | None, values: Values) -> dict[str, str]:
    """Intervals, maturity timing, ML baseline, calibration and reference sensitivity."""
    pairs = {
        ("twin_minus_moore_mae", "all"): "TwDiffMoore",
        ("twin_minus_moore_mae", "visit 1"): "TwDiffMooreFirst",
        ("twin_minus_moore_mae", "visit 6"): "TwDiffMooreLast",
        ("twin_minus_moore_mae_full", "all"): "TwDiffMooreFull",
        ("twin_minus_mirwald_mae", "boys"): "TwDiffMirwald",
        ("twin_minus_moore_circa_auc", "all"): "TwDiffAUC",
        ("twin_minus_ml_mae", "all"): "TwDiffML",
        ("twin_minus_ml_mae", "visit 1"): "TwDiffMLFirst",
        ("twin_minus_ml_mae", "visit 6"): "TwDiffMLLast",
        ("twin_minus_ml_circa_auc", "all"): "TwDiffMLAUC",
        ("twin_minus_moore_mae", "early maturers"): "TwEarlyDiffMoore",
        ("twin_minus_ml_mae", "early maturers"): "TwEarlyDiffML",
        ("twin_minus_moore_mae", "average maturers"): "TwAverageDiffMoore",
        ("twin_minus_ml_mae", "average maturers"): "TwAverageDiffML",
        ("twin_minus_moore_mae", "late maturers"): "TwLateDiffMoore",
        ("twin_minus_ml_mae", "late maturers"): "TwLateDiffML",
    }
    scalars = [
        "TwEarlyChildren",
        "TwAverageChildren",
        "TwLateChildren",
        "MlMAE",
        "MlCov",
        "MlWidth",
        "MlAUC",
        "CircaSlope",
        "CircaIntercept",
        "CircaBrier",
        "SplineTwin",
        "SplineMoore",
        "SplineML",
        "TwBootN",
        "TwTimingSD",
    ]
    if run is None:
        values.pending(*pairs.values(), *[f"{n}CI" for n in pairs.values()], *scalars)
        return {"twin_tab_comparisons.tex": PENDING, "twin_tab_timing.tex": PENDING}
    rid = _rid(run)
    values.runs["twin_analysis"] = rid
    intervals = pd.read_csv(run / "intervals.csv")
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    rows = []
    for (comparison, subset), name in pairs.items():
        match = intervals[(intervals["comparison"] == comparison) & (intervals["subset"] == subset)]
        if match.empty:
            values.pending(name, f"{name}CI")
            continue
        r = match.iloc[0]
        digits = 3 if "auc" in comparison else 2
        values.put(name, _signed(r["estimate"], digits), rid)
        values.put(
            f"{name}CI", f"{_signed(r['ci_low'], digits)} to {_signed(r['ci_high'], digits)}", rid
        )
        label = {
            "twin_minus_moore_mae": "Twin $-$ Moore, MAE",
            "twin_minus_moore_mae_full": "Twin $-$ Moore, MAE (full-information ref.)",
            "twin_minus_mirwald_mae": "Twin $-$ Mirwald, MAE",
            "twin_minus_moore_circa_auc": "Twin $-$ Moore, spurt AUC",
            "twin_minus_ml_mae": "Twin $-$ boosting, MAE",
            "twin_minus_ml_circa_auc": "Twin $-$ boosting, spurt AUC",
        }[comparison]
        excludes = "yes" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "no"
        rows.append(
            [
                label,
                subset,
                _signed(r["estimate"], digits),
                f"{_signed(r['ci_low'], digits)} to {_signed(r['ci_high'], digits)}",
                excludes,
            ]
        )
    values.number("TwBootN", int(intervals["n_resamples"].max()), rid)
    ml, circa, spline = (
        record["ml_baseline"],
        record["circa_calibration"],
        record["reference_sensitivity"],
    )
    values.number("MlMAE", ml["ml_mae"], rid)
    values.put("MlCov", percent(ml["ml_conf_coverage"]), rid)
    values.number("MlWidth", ml["ml_conf_width"], rid)
    values.number("MlAUC", circa["auc_ml"], rid)
    values.number("CircaSlope", circa["calibration_slope"], rid)
    values.number("CircaIntercept", circa["calibration_intercept"], rid)
    values.number("CircaBrier", circa["brier"], rid, 3)
    values.number("SplineTwin", spline["twin_mae_vs_spline"], rid)
    values.number("SplineMoore", spline["moore_mae_vs_spline"], rid)
    values.number("SplineML", spline["ml_mae_vs_spline"], rid)
    comparisons = tabular(
        "llrrc", ["Comparison", "Subset", "Difference", "95\\% CI", "Excludes 0"], rows
    )

    timing = pd.read_csv(run / "timing_groups.csv")
    timing_rows = []
    for _, r in timing.iterrows():
        group = str(r["timing"])
        values.number(f"Tw{group.capitalize()}Children", int(r["n_children"]), rid)
        timing_rows.append(
            [
                group.capitalize(),
                _number(int(r["n_children"]), 0),
                f"{_number(float(r['twin_mae']), 2)} ({_signed(r['twin_bias'])})",
                f"{_number(float(r['ml_mae']), 2)} ({_signed(r['ml_bias'])})",
                f"{_number(float(r['moore_mae']), 2)} ({_signed(r['moore_bias'])})",
                percent(float(r["twin_conf_coverage"])),
            ]
        )
    timing_table = tabular(
        "lrrrrr",
        ["Maturity timing", "Children", "Twin", "Boosting", "Moore", "Twin conf. cov."],
        timing_rows,
    )
    return {"twin_tab_comparisons.tex": comparisons, "twin_tab_timing.tex": timing_table}
