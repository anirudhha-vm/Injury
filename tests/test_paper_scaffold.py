"""Paper scaffolds: methods text matches the configs, and nothing in paper/ is fabricated."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from youthtwin.paper import methods
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.results_guard import scan_paper_dir, scan_text

PAPER = repo_root() / "paper"


def test_committed_methods_draft_matches_the_configs() -> None:
    """If a config changes, the committed draft must be regenerated (scripts/draft_methods.py)."""
    committed = (PAPER / "methods_draft.md").read_text(encoding="utf-8")
    assert committed == methods.render_methods(methods.load_all()), (
        "paper/methods_draft.md is out of date: run `uv run python scripts/draft_methods.py`"
    )
    assert committed.startswith("<!-- GENERATED")


def test_methods_text_follows_a_config_change() -> None:
    configs = copy.deepcopy(methods.load_all())
    before = methods.render_methods(configs)
    configs["filter"]["butterworth"]["cutoff_hz"] = 11.0
    configs["hazard"]["prediction_horizon_weeks"] = 6
    configs["metrics"]["thresholds"] = [0.05]
    after = methods.render_methods(configs)
    assert "cut-off [[SPEC: 11.0]] Hz" in after and "cut-off [[SPEC: 11.0]] Hz" not in before
    assert "following [[SPEC: 6]] weeks" in after
    assert "none pre-specified yet" in before and "[[SPEC: 0.05]]" in after


def test_every_number_in_the_methods_is_a_tagged_config_value() -> None:
    text = methods.render_methods(methods.load_all())
    assert scan_text(text, Path("methods_draft.md")) == []
    # The values really are the configured ones.
    assert f"[[SPEC: {load_config('pose/filter.yaml')['butterworth']['cutoff_hz']}]] Hz" in text
    sitar = load_config("growth/sitar.yaml")
    assert methods.spec(sitar["spline"]["knots_years"]) in text
    assert f"{methods.spec(load_config('eval/metrics.yaml')['bootstrap']['n_resamples'])}" in text
    for ablation in load_config("eval/ablations.yaml")["ablations"]:
        assert f"{ablation['id']}. {ablation['name'].replace('_', ' ')}" in text


def test_methods_do_not_overstate_verification() -> None:
    text = methods.render_methods(methods.load_all())
    assert "NOT yet verified against the original paper" in text  # the Mirwald gate is closed
    assert "not validated against force plates: [[SPEC: yes]]" in text
    assert "Synthetic results are reported only in this section." in text
    assert text.count("TODO(human)") >= 5


def test_spec_formatting() -> None:
    assert methods.spec(8.0) == "[[SPEC: 8.0]]"
    assert methods.spec([6, 8]) == "[[SPEC: 6, 8]]"
    assert methods.spec(True) == "[[SPEC: yes]]" and methods.spec(False) == "[[SPEC: no]]"


@pytest.mark.parametrize("name", ["paperA_outline.md", "paperB1_outline.md", "paperB_outline.md"])
def test_outlines_contain_placeholders_and_no_results(name: str) -> None:
    text = (PAPER / name).read_text(encoding="utf-8")
    assert text.count("[[RESULT:") >= 8
    # Nothing has been filled in yet: the only RUN tag is the template in the header note.
    assert re.findall(r"\[\[RUN:([^|\]]*)", text) == ["<run_id>"]
    assert scan_text(text, PAPER / name) == []
    assert "TODO(human)" in text
    for section in (
        "## Introduction",
        "## Methods",
        "## Results",
        "## Discussion",
        "## Conclusion",
    ):
        assert section in text
    # No statistic-like number has slipped in outside a tag.
    stripped = re.sub(r"\[\[[A-Z]+:[^\]]*\]\]", "", text)
    assert not re.search(r"\d+\.\d+|\d+\s?%", stripped)


def test_paper_b_follows_tripod_and_keeps_simulation_separate() -> None:
    text = (PAPER / "paperB_outline.md").read_text(encoding="utf-8")
    for first, last in ((1, 2), (3, 4), (5, 19), (20, 24), (25, 27)):
        assert f"items {first}\u2013{last}" in text  # en dash, as written in the outline
    assert "## Simulation study (synthetic data only)" in text
    results = text[text.index("## Results") : text.index("## Simulation study")]
    assert "synthetic" not in results.lower()  # synthetic results only in their own section
    assert "not diagnosis" in text and "Bahr" in text


def test_whole_paper_directory_passes_the_results_guard() -> None:
    violations = scan_paper_dir(PAPER)
    assert violations == [], "\n".join(v.format() for v in violations)


def test_preregistration_separates_confirmatory_from_exploratory() -> None:
    text = (PAPER / "preregistration.md").read_text(encoding="utf-8")
    assert scan_text(text, PAPER / "preregistration.md") == []
    for line in (
        "**H3 (B2, exploratory):**",
        "**H4 (B2, exploratory):**",
        "**H2 (B2, confirmatory):**",
    ):
        assert line in text
    assert "Risk thresholds are elicited before data lock" in text
    assert "No threshold is chosen after seeing results" in text
    hazard = load_config("models/hazard.yaml")["confirmatory"]
    assert f"[[SPEC: max_parameters = {hazard['max_parameters']}]]" in text
    assert "[[SPEC: 3 years]]" in text  # the growth-twin calibration claim is restricted


def test_methods_describe_the_confirmatory_model() -> None:
    text = methods.render_methods(methods.load_all())
    assert "Confirmatory model: logistic regression" in text
    assert "All larger models below are exploratory." in text
    first = load_config("models/hazard.yaml")["confirmatory"]["candidate_predictors_in_order"][0]
    assert first.replace("_", " ") in text
