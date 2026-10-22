"""Draft the results section from run outputs (``scripts/draft_results.py``).

Every number is written as ``[[RUN:<run_id>|<value>]]`` so the results guard can trace it to
the run that produced it. Real-data sections and the simulation study are kept apart, and every
synthetic section says so in its heading. A section whose run does not exist yet is replaced by
a ``[[RESULT: ...]]`` placeholder rather than left out silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

PRIMARY = "time_dependent_auc_4wk"
Section = Callable[[Path, str], list[str]]


def run_id(run: Path) -> str:
    return str(json.loads((run / "run.json").read_text(encoding="utf-8"))["run_id"])


def tag(run: str, value: Any, digits: int = 2) -> str:
    """A traceable number (``n/a`` for missing values, which are not results)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if hasattr(value, "item"):  # numpy / pandas scalar
        value = value.item()
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return f"[[RUN:{run}|{text}]]"


def with_interval(run: str, estimate: Any, low: Any, high: Any, digits: int = 2) -> str:
    """``estimate (low to high)``; table cells arrive as pandas scalars, hence ``Any``."""
    values = [float(v) for v in (estimate, low, high)]
    return f"{tag(run, values[0], digits)} ({tag(run, values[1], digits)} to {tag(run, values[2], digits)})"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


# --------------------------------------------------------------------------- real data


def real_growth(run: Path, rid: str) -> list[str]:
    summary = pd.read_csv(run / "summary.csv")
    reference = pd.read_csv(run / "reference.csv")
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    n_replicates = len({k.rsplit("replicate", 1)[1] for k in record.get("fits", {})})
    lines = [
        f"Reference age at PHV was available for {tag(rid, int(reference['included'].sum()))} of "
        f"{tag(rid, len(reference))} children. Each estimator saw one season of three 6-monthly "
        f"measurements per child; results pool {tag(rid, n_replicates)} random window draws.",
        "",
    ]
    rows = []
    part = summary[summary["reference_subset"] == "all"]
    for _, row in part.iterrows():
        label = "All" if row["grouping"] == "all" else str(row["group"])
        rows.append(
            [
                label,
                tag(rid, int(row["n_athletes"])),
                tag(rid, row["posterior_mae_years"]),
                tag(rid, row["point_mae_years"]),
                tag(rid, row["mirwald_mae_years"]),
                tag(rid, row["posterior_bias_years"]),
                tag(rid, row["point_bias_years"]),
                tag(rid, row["coverage"]),
                tag(rid, row["mean_interval_width_years"]),
            ]
        )
    lines += table(
        [
            "Group",
            "n (child-windows)",
            "Twin MAE (y)",
            "Moore MAE (y)",
            "Mirwald MAE, boys (y)",
            "Twin bias (y)",
            "Moore bias (y)",
            "[[SPEC: 90%]] interval coverage",
            "Interval width (y)",
        ],
        rows,
    )
    strict = summary[
        (summary["reference_subset"] == "strict reference") & (summary["grouping"] == "all")
    ]
    if not strict.empty:
        row = strict.iloc[0]
        lines += [
            "",
            "Sensitivity, children whose two reference methods agree within half a year: twin MAE "
            f"{tag(rid, row['posterior_mae_years'])} y, Moore MAE {tag(rid, row['point_mae_years'])} y, "
            f"coverage {tag(rid, row['coverage'])}.",
        ]
    fits = record.get("fits", {})
    if fits:
        worst = max(float(f.get("max_rhat", float("nan"))) for f in fits.values())
        least = min(float(f.get("min_ess_bulk", float("nan"))) for f in fits.values())
        lines += [
            "",
            f"Sampler diagnostics across fits: largest R-hat {tag(rid, worst)}, smallest bulk ESS "
            f"{tag(rid, least, 0)}.",
        ]
    return lines


def real_risk(run: Path, rid: str) -> list[str]:
    performance = pd.read_csv(run / "performance.csv")
    differences = pd.read_csv(run / "differences.csv")
    inflation = pd.read_csv(run / "leakage_inflation.csv")
    first = performance.iloc[0]
    lines = [
        f"{tag(rid, int(first['n_athletes']))} runners, {tag(rid, int(first['n_rows']))} event "
        f"days, {tag(rid, int(first['n_events']))} injury days. Adult runners: this checks the "
        "evaluation pipeline on real outcomes and is not evidence about growth-related injury.",
        "",
    ]
    rows = []
    for (scheme, model), part in performance.groupby(["scheme", "model"], sort=False):
        metrics = part.set_index("metric")
        auc, slope = metrics.loc[PRIMARY], metrics.loc["calibration_slope"]
        rows.append(
            [
                str(scheme).replace("_", " "),
                str(model).replace("_", " "),
                with_interval(rid, auc["estimate"], auc["ci_low"], auc["ci_high"]),
                with_interval(rid, slope["estimate"], slope["ci_low"], slope["ci_high"]),
            ]
        )
    lines += table(
        ["Split scheme", "Model", "AUC ([[SPEC: 95%]] CI)", "Calibration slope ([[SPEC: 95%]] CI)"],
        rows,
    )
    lines += [
        "",
        "Paired differences in AUC against the load-only model (runner-grouped folds):",
        "",
    ]
    part = differences[differences["metric"] == PRIMARY]
    lines += table(
        ["Model", "Difference ([[SPEC: 95%]] CI)"],
        [
            [
                str(r["model"]).replace("_", " "),
                with_interval(rid, r["difference"], r["ci_low"], r["ci_high"]),
            ]
            for _, r in part.iterrows()
        ],
    )
    lines += ["", "AUC inflation when a runner's days are allowed on both sides of a split:", ""]
    lines += table(
        ["Model", "Row-level minus runner-grouped AUC ([[SPEC: 95%]] CI)"],
        [
            [
                str(r["model"]).replace("_", " "),
                with_interval(rid, r["leaky_minus_grouped"], r["ci_low"], r["ci_high"], 3),
            ]
            for _, r in inflation.iterrows()
        ],
    )
    return lines


# --------------------------------------------------------------------------- simulation study


def validity(run: Path, rid: str) -> list[str]:
    agreement = pd.read_csv(run / "agreement_by_maturity_group.csv")
    precision = pd.read_csv(run / "precision.csv")
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    lines = [
        "Markerless landmark error calibrated to a published smartphone knee-flexion error "
        f"(offset SD {tag(rid, float(record['calibrated_offset_sd_px']))} px at adult size). "
        "One simulated study of the planned size:",
        "",
    ]
    rows = []
    for _, r in agreement.iterrows():
        rows.append(
            [
                f"{r['variable'].replace('_', ' ')} ({r['event'].replace('_', ' ')})",
                str(r["maturity_group"]).replace("_", " "),
                tag(rid, int(r["n"])),
                tag(rid, r["bias"]),
                f"{tag(rid, r['loa_low'])} to {tag(rid, r['loa_high'])}",
                tag(rid, r["rmse"]),
                tag(rid, r["icc21"]),
            ]
        )
    lines += table(
        [
            "Variable",
            "Maturity",
            "n",
            "Bias (deg)",
            "[[SPEC: 95%]] LoA (deg)",
            "RMSE (deg)",
            "ICC (two-way, absolute agreement)",
        ],
        rows,
    )
    lines += ["", "Across repeated studies of the same size:", ""]
    lines += table(
        [
            "Variable",
            "Maturity",
            "RMSE, [[SPEC: 95%]] range across studies (deg)",
            "Share of studies with RMSE below limit",
        ],
        [
            [
                f"{r['variable'].replace('_', ' ')} ({r['event'].replace('_', ' ')})",
                str(r["maturity_group"]).replace("_", " "),
                f"{tag(rid, r['rmse_p025'])} to {tag(rid, r['rmse_p975'])}",
                tag(rid, r["share_studies_meeting_threshold"]),
            ]
            for _, r in precision.iterrows()
        ],
    )
    return lines


def sim_growth(run: Path, rid: str) -> list[str]:
    summary = pd.read_csv(run / "summary.csv")
    rows = [
        [
            str(r["group"]),
            tag(rid, int(r["n_athletes"])),
            tag(rid, r["posterior_mae_years"]),
            tag(rid, r["point_mae_years"]),
            tag(rid, r["coverage"]),
            tag(rid, r["mean_interval_width_years"]),
        ]
        for _, r in summary.iterrows()
    ]
    return table(
        [
            "Group",
            "n",
            "Twin MAE (y)",
            "Moore MAE (y)",
            "[[SPEC: 90%]] coverage",
            "Interval width (y)",
        ],
        rows,
    )


def sim_risk(run: Path, rid: str) -> list[str]:
    performance = pd.read_csv(run / "performance.csv")
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    first = performance.iloc[0]
    chosen = record.get("confirmatory_predictors") or []
    lines = [
        f"{tag(rid, int(first['n_athletes']))} simulated athletes, {tag(rid, int(first['n_events']))} "
        f"athlete-weeks followed by a growth-related injury within 4 weeks. The confirmatory model "
        f"used {tag(rid, len(chosen))} predictor(s): {', '.join(f'`{c}`' for c in chosen) or 'none'}.",
        "",
    ]
    rows = []
    for model, part in performance.groupby("model", sort=False):
        metrics = part.set_index("metric")
        auc, slope = metrics.loc[PRIMARY], metrics.loc["calibration_slope"]
        rows.append(
            [
                str(model).replace("_", " "),
                with_interval(rid, auc["estimate"], auc["ci_low"], auc["ci_high"]),
                with_interval(rid, slope["estimate"], slope["ci_low"], slope["ci_high"]),
            ]
        )
    return lines + table(
        ["Model", "AUC ([[SPEC: 95%]] CI)", "Calibration slope ([[SPEC: 95%]] CI)"], rows
    )


def ablation(run: Path, rid: str) -> list[str]:
    results = pd.read_csv(run / "results.csv")
    wide = results.pivot_table(
        index=["ablation_id", "name"], columns="metric", values="value"
    ).reset_index()
    rows = [
        [
            tag(rid, int(r["ablation_id"])),
            str(r["name"]).replace("_", " "),
            tag(rid, r["auc"]),
            tag(rid, r.get("correlation_with_true_risk", float("nan"))),
        ]
        for _, r in wide.iterrows()
    ]
    lines = table(["#", "Configuration", "AUC", "Correlation with true risk"], rows)
    if (run / "differences.csv").is_file():
        differences = pd.read_csv(run / "differences.csv")
        part = differences[differences["metric"] == PRIMARY]
        lines += ["", "Pre-specified paired comparisons (AUC difference, athlete bootstrap):", ""]
        lines += table(
            ["Model", "Reference", "Difference ([[SPEC: 95%]] CI)"],
            [
                [
                    str(r["model"]).replace("_", " "),
                    str(r["reference"]).replace("_", " "),
                    with_interval(rid, r["difference"], r["ci_low"], r["ci_high"], 3),
                ]
                for _, r in part.iterrows()
            ],
        )
    return lines


def cohort_size(run: Path, rid: str) -> list[str]:
    summary = pd.read_csv(run / "summary.csv")
    sizes = sorted(summary["n_athletes"].unique())
    events = summary[summary["metric"] == "athletes_with_event"].set_index("n_athletes")["mean"]
    auc = summary[summary["metric"] == "auc"]
    rows = [["Athletes with an injury (mean)", *[tag(rid, float(events[n]), 1) for n in sizes]]]
    for model, part in auc.groupby("model", sort=False):
        part = part.set_index("n_athletes")
        rows.append(
            [
                f"AUC, {str(model).replace('_', ' ')}",
                *[
                    f"{tag(rid, part.loc[n, 'mean'])} ({tag(rid, part.loc[n, 'low'])} to {tag(rid, part.loc[n, 'high'])})"
                    for n in sizes
                ],
            ]
        )
    return table(["", *[f"{n} athletes" for n in sizes]], rows)


SECTIONS: tuple[tuple[str, str, Section, bool], ...] = (
    (
        "real_growth_berkeley",
        "Growth twin on real growth records (Berkeley Child Guidance Study)",
        real_growth,
        False,
    ),
    (
        "real_risk_runners",
        "Injury-risk evaluation on real injury outcomes (competitive runners)",
        real_risk,
        False,
    ),
    ("sim_study_validation", "Smartphone versus motion capture", validity, True),
    ("sim_study_growth", "Growth twin in a simulated cohort", sim_growth, True),
    ("evaluation", "Risk models in a simulated cohort", sim_risk, True),
    ("ablation", "Conditioning ablation", ablation, True),
    ("sim_study_sizes", "Performance against cohort size", cohort_size, True),
)


def render_results(runs: Mapping[str, Path | None]) -> str:
    """The results draft; ``runs`` maps each output kind to its run directory (or None)."""
    lines = [
        "# Results draft",
        "",
        "<!-- Generated by scripts/draft_results.py from the latest runs. Do not edit by hand:",
        "     change the analysis, rerun, regenerate. -->",
        "",
        "Part 1 uses real public data. Part 2 is a simulation study on synthetic data calibrated to",
        "published values; its numbers describe the method under stated assumptions, not athletes.",
        "",
        "## Part 1. Real data",
    ]
    simulation_started = False
    for kind, title, render, synthetic in SECTIONS:
        if synthetic and not simulation_started:
            lines += ["", "## Part 2. Simulation study (SYNTHETIC)"]
            simulation_started = True
        heading = f"### {title}" + (" (SYNTHETIC)" if synthetic else " (REAL)")
        lines += ["", heading, ""]
        run = runs.get(kind)
        if run is None:
            lines.append(f"[[RESULT: {title} - run not available yet]]")
            continue
        rid = run_id(run)
        lines += [f"Source run: `{rid}`.", "", *render(run, rid)]
    return "\n".join(lines) + "\n"
