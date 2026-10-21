"""The generated results draft: every number traceable, missing runs shown as placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from youthtwin.paper import results
from youthtwin.utils.results_guard import scan_text


def _run(path: Path, run_id: str, tables: dict[str, pd.DataFrame], **extra: object) -> Path:
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps({"run_id": run_id, **extra}), encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(path / name, index=False)
    return path


def _performance(scheme: str | None) -> pd.DataFrame:
    rows = []
    for model in ("load_only", "penalised_logistic"):
        for metric in (results.PRIMARY, "calibration_slope"):
            row = {
                "model": model,
                "metric": metric,
                "estimate": 0.6,
                "ci_low": 0.5,
                "ci_high": 0.7,
                "n_athletes": 74,
                "n_rows": 1000,
                "n_events": 50,
            }
            if scheme:
                row["scheme"] = scheme
            rows.append(row)
    return pd.DataFrame(rows)


def test_tags() -> None:
    assert results.tag("r1", 0.123456) == "[[RUN:r1|0.12]]"
    assert results.tag("r1", np.int64(7)) == "[[RUN:r1|7]]"
    assert results.tag("r1", float("nan")) == "n/a"
    assert (
        results.with_interval("r1", 0.6, 0.5, 0.7)
        == "[[RUN:r1|0.60]] ([[RUN:r1|0.50]] to [[RUN:r1|0.70]])"
    )


def test_draft_passes_the_results_guard(tmp_path: Path) -> None:
    risk = _run(
        tmp_path / "risk",
        "realrisk-x",
        {
            "performance.csv": _performance("runner_grouped"),
            "differences.csv": pd.DataFrame(
                [
                    {
                        "model": "penalised_logistic",
                        "metric": results.PRIMARY,
                        "difference": 0.1,
                        "ci_low": 0.05,
                        "ci_high": 0.15,
                    }
                ]
            ),
            "leakage_inflation.csv": pd.DataFrame(
                [
                    {
                        "model": "load_only",
                        "leaky_minus_grouped": 0.04,
                        "ci_low": 0.02,
                        "ci_high": 0.06,
                    }
                ]
            ),
        },
    )
    evaluation = _run(
        tmp_path / "eval",
        "evaluation-y",
        {"performance.csv": _performance(None)},
        confirmatory_predictors=["obs_leg_length_velocity"],
    )
    text = results.render_results({"real_risk_runners": risk, "evaluation": evaluation})
    path = tmp_path / "results_draft.md"
    assert scan_text(text, path) == []
    assert "[[RUN:realrisk-x|0.60]]" in text and "`obs_leg_length_velocity`" in text
    # Real and synthetic parts are separated, and synthetic headings say so.
    assert text.index("## Part 1. Real data") < text.index(
        "## Part 2. Simulation study (SYNTHETIC)"
    )
    assert "### Risk models in a simulated cohort (SYNTHETIC)" in text
    assert "### Injury-risk evaluation on real injury outcomes (competitive runners) (REAL)" in text
    # Runs that do not exist yet become placeholders, not silence.
    assert "[[RESULT: Conditioning ablation - run not available yet]]" in text
