"""IEEE manuscript values: traceable macros, pending markers, no hand-typed numbers in main.tex."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from youthtwin.paper import ieee
from youthtwin.utils.config import load_config, repo_root

CONFIGS = {
    "sitar": load_config("growth/sitar.yaml"),
    "agreement": load_config("validation/agreement.yaml"),
    "cohort": load_config("sim/cohort.yaml"),
    "calibration": load_config("sim/calibration.yaml"),
    "validation": load_config("sim/validation.yaml"),
    "runners": load_config("external/runners.yaml"),
    "berkeley": load_config("external/berkeley.yaml"),
    "metrics": load_config("eval/metrics.yaml"),
}


def test_without_runs_every_result_is_pending() -> None:
    values, tables = ieee.build({}, CONFIGS)
    results = {
        n: t for n, (t, source) in values.macros.items() if not n.startswith(("Spec", "Cited"))
    }
    assert results and all(text == ieee.PENDING for text in results.values())
    assert all(ieee.PENDING in text for text in tables.values())
    assert values.macros["SpecIncidence"][0] == "3.8\\%"


def test_values_come_from_the_run(tmp_path: Path) -> None:
    run = tmp_path / "realrisk-z"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": "realrisk-z"}), encoding="utf-8")
    rows = []
    for scheme in ("runner_grouped", "temporal_holdout", "row_level_leaky"):
        for model in ("load_only", "penalised_logistic", "gradient_boosting"):
            for metric in (ieee.PRIMARY, "calibration_slope"):
                rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "metric": metric,
                        "estimate": 0.625,
                        "ci_low": 0.5,
                        "ci_high": 0.75,
                        "n_athletes": 74,
                        "n_rows": 42798,
                        "n_events": 575,
                    }
                )
    pd.DataFrame(rows).to_csv(run / "performance.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "penalised_logistic",
                "metric": ieee.PRIMARY,
                "difference": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
            }
        ]
    ).to_csv(run / "differences.csv", index=False)
    pd.DataFrame(
        [
            {"model": m, "leaky_minus_grouped": 0.04, "ci_low": 0.01, "ci_high": 0.07}
            for m in ("load_only", "penalised_logistic", "gradient_boosting")
        ]
    ).to_csv(run / "leakage_inflation.csv", index=False)
    values, tables = ieee.build({"real_risk_runners": run}, CONFIGS)
    assert values.macros["RunAucLogit"] == ("0.62", "realrisk-z")  # rounded, with its run
    assert values.macros["RunRows"][0] == "42{,}798"
    assert values.runs == {"real_risk_runners": "realrisk-z"}
    assert "Runner-grouped CV" in tables["ieee_tab_runners.tex"]
    rendered = values.render()
    assert "\\renewcommand{\\RunAucLogit}{0.62}% realrisk-z" in rendered


def test_macro_names_are_letters_only() -> None:
    with pytest.raises(ValueError, match="letters only"):
        ieee.Values().put("Run2", "x", "r")


def test_manuscript_uses_only_defined_macros() -> None:
    """Every custom macro in main.tex is produced by the generator (or defined in the preamble)."""
    text = (repo_root() / "paper" / "ieee" / "main.tex").read_text(encoding="utf-8")
    values, _ = ieee.build({}, CONFIGS)
    used = set(re.findall(r"\\((?:Berk|Run|Val|Sim|Abl|Size|Spec|Cited)[A-Za-z]*)", text))
    assert used and used <= set(values.macros)
