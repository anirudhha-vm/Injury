"""Evaluation tables and the figure generator: structure, traceability, no hand-typed numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.eval import report
from youthtwin.eval.predictions import out_of_fold_predictions
from youthtwin.eval.splits import grouped_stratified_folds
from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.baselines import make_baseline
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.results_guard import scan_paper_dir

LABEL = "y_all_non_contact"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    settings = load_config("eval/metrics.yaml")
    settings["bootstrap"]["n_resamples"] = 40
    return settings


@pytest.fixture(scope="module")
def predictions(person_period: PersonPeriod, person_period_config: dict[str, Any]) -> pd.DataFrame:
    frame = person_period.frame
    baselines = load_config("models/baselines.yaml")
    folds = grouped_stratified_folds(frame["athlete_id"], frame[LABEL], n_splits=3, seed=0)
    factories = {
        name: (lambda f, _n=name: make_baseline(_n, f, baselines, person_period_config))
        for name in ("age_only", "maturity_growth_velocity")
    }
    return out_of_fold_predictions(
        frame, LABEL, factories, folds, keep=["sex", "sport", "age", "moore_maturity_offset"]
    )


def test_performance_table(predictions: pd.DataFrame, config: dict[str, Any]) -> None:
    table = report.performance_table(predictions, config, seed=0)
    assert set(table["model"]) == {"age_only", "maturity_growth_velocity"}
    assert set(table["metric"]) == set(config["metrics"])  # no thresholds configured yet
    assert (table["ci_low"] <= table["ci_high"]).all()
    # For discrimination and Brier score the percentile interval brackets the estimate. For ICI
    # and ECE it need not: those are upward-biased under resampling (see eval/report.py).
    unbiased = table["metric"].isin(["time_dependent_auc_4wk", "brier_score"])
    inside = (table["ci_low"] <= table["estimate"]) & (table["estimate"] <= table["ci_high"])
    assert inside[unbiased].all()
    assert (table["n_resamples"] > 30).all() and (table["ci_level"] == 0.95).all()
    assert (table["n_rows"] == len(predictions)).all() and table["synthetic"].all()
    assert (table["n_athletes"] == 300).all() and (table["label"] == LABEL).all()
    auc = table[table["metric"] == "time_dependent_auc_4wk"].set_index("model")["estimate"]
    assert auc.between(0.3, 0.9).all()
    again = report.performance_table(predictions, config, seed=0)
    pd.testing.assert_frame_equal(table, again)  # reproducible for a seed


def test_thresholds_are_reported_only_when_pre_specified(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> None:
    with_threshold = {**config, "metrics": ["brier_score"], "thresholds": [0.05]}
    table = report.performance_table(predictions, with_threshold, seed=0)
    assert {"sensitivity_at_0.05", "specificity_at_0.05", "ppv_at_0.05"} <= set(table["metric"])
    with pytest.raises(KeyError, match="Unknown metrics"):
        report.metric_functions({**config, "metrics": ["accuracy"]})


def test_difference_table(predictions: pd.DataFrame, config: dict[str, Any]) -> None:
    table = report.difference_table(predictions, "age_only", config, seed=0)
    assert (
        set(table["model"]) == {"maturity_growth_velocity"}
        and (table["reference"] == "age_only").all()
    )
    assert set(table["metric"]) == {"time_dependent_auc_4wk", "brier_score"}
    assert (
        (table["ci_low"] <= table["difference"]) & (table["difference"] <= table["ci_high"])
    ).all()
    expected = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    assert (table["interval_excludes_zero"] == expected).all()
    with pytest.raises(KeyError, match="not in the predictions"):
        report.difference_table(predictions, "oracle", config, seed=0)


def test_decision_curve_calibration_and_subgroup_tables(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> None:
    curve = report.decision_curve_table(predictions, config)
    assert set(curve["strategy"]) == {
        "treat_all",
        "treat_none",
        "age_only",
        "maturity_growth_velocity",
    }
    assert curve["threshold"].nunique() == config["decision_curve"]["n_thresholds"]
    calibration = report.calibration_table(predictions, config)
    assert (calibration.groupby("model")["n"].sum() == len(predictions)).all()
    assert (
        calibration.groupby("model")["predicted"].apply(lambda s: s.is_monotonic_increasing).all()
    )
    subgroup = report.subgroup_report(predictions, config, "moore_maturity_offset")
    assert set(subgroup["subgroup"]) == {"sex", "maturity_stratum", "sport", "age_band"}
    assert {"time_dependent_auc_4wk", "brier_score", "n_events", "note"} <= set(subgroup.columns)
    assert subgroup["synthetic"].all()


def test_figures_and_tables_are_generated_with_provenance(
    predictions: pd.DataFrame, config: dict[str, Any], tmp_path: Path
) -> None:
    run = tmp_path / "evaluation-test"
    run.mkdir()
    tables = {
        "performance": report.performance_table(predictions, config, seed=0),
        "differences": report.difference_table(predictions, "age_only", config, seed=0),
        "decision_curve": report.decision_curve_table(predictions, config),
        "calibration": report.calibration_table(predictions, config),
        "subgroups": report.subgroup_report(predictions, config, "moore_maturity_offset"),
        "shap_importance": pd.DataFrame({"model": ["m"], "input": ["age"], "mean_abs_shap": [0.1]}),
    }
    for name, table in tables.items():
        table.to_csv(run / f"{name}.csv", index=False)
    record = {
        "run_id": "evaluation-test",
        "synthetic": True,
        "seed": 0,
        "config_hash": "abc",
        "git_commit": "deadbeef",
    }
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "make_figures", repo_root() / "scripts" / "make_figures.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paper = tmp_path / "paper"
    module._latest = lambda kind: None  # no other runs in this test
    module.main(evaluation_dir=run, ablation_dir=None, growth_dir=None, paper_dir=paper)

    figures = sorted(p.name for p in (paper / "figures").glob("*.png"))
    assert figures == ["calibration.png", "decision_curve.png", "discrimination.png"]
    for file in [*(paper / "figures").glob("*.png"), *(paper / "tables").glob("*.csv")]:
        sidecar = json.loads(Path(str(file) + ".provenance.json").read_text(encoding="utf-8"))
        assert sidecar["run_id"] == "evaluation-test" and sidecar["synthetic"] is True
        assert sidecar["git_commit"] == "deadbeef"
    assert scan_paper_dir(paper) == []  # nothing in paper/ lacks provenance
    written = pd.read_csv(paper / "tables" / "performance.csv")
    np.testing.assert_allclose(written["estimate"], tables["performance"]["estimate"])
    # A stray file without provenance is caught by the guard.
    (paper / "tables" / "typed_by_hand.csv").write_text("auc\n0.9\n", encoding="utf-8")
    assert [v.reason for v in scan_paper_dir(paper)] == ["missing provenance sidecar"]
