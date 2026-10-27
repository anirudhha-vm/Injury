"""Growth-twin manuscript generator: run classification, pending markers, defined macros."""

from __future__ import annotations

import re
from pathlib import Path

from youthtwin.paper import twin_paper
from youthtwin.paper.ieee import PENDING
from youthtwin.utils.config import load_config, repo_root


def _settings(**changes: object) -> dict[str, object]:
    base = {
        "method": "map",
        "leg": False,
        "leg_lag": True,
        "fold": None,
        "scenario": {"visit_step": 1, "added_noise_cm": 0.0, "sitting_height": True},
    }
    scenario = changes.pop("scenario", {})
    base.update(changes)
    base["scenario"] = {**base["scenario"], **scenario}  # type: ignore[dict-item]
    return {"settings": base}


def test_runs_are_classified_by_their_settings() -> None:
    assert twin_paper.classify(_settings()) == "main"
    assert twin_paper.classify(_settings(scenario={"visit_step": 2})) == "yearly"
    assert twin_paper.classify(_settings(scenario={"added_noise_cm": 1.0})) == "noise"
    assert twin_paper.classify(_settings(scenario={"sitting_height": False})) == "nositting"
    assert twin_paper.classify(_settings(leg=True, leg_lag=False)) == "leg_shared"
    assert twin_paper.classify(_settings(leg=True)) == "leg_lag"
    assert twin_paper.classify(_settings(method="nuts", fold=0)) == "nuts_fold0"
    assert twin_paper.classify(_settings(fold=2)) is None
    assert twin_paper.classify({}) is None  # runs made before settings were recorded


def test_without_runs_everything_is_pending_and_the_manuscript_macros_exist(tmp_path: Path) -> None:
    configs = {"sequential": load_config("external/sequential.yaml")}
    values, tables, runs = twin_paper.build(tmp_path / "none", None, configs)
    assert runs == {}
    results = {n: t for n, (t, _) in values.macros.items() if not n.startswith("Spec")}
    assert results and all(text == PENDING for text in results.values())
    assert all(PENDING in text for text in tables.values())
    text = (repo_root() / "paper" / "ieee_twin" / "main.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\((?:Tw|Ch|Spec)[A-Za-z]*)", text))
    assert used and used <= set(values.macros), sorted(used - set(values.macros))
