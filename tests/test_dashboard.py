"""Decision-support prompts and the research dashboard (non-negotiable N9)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from youthtwin.dashboard import figures
from youthtwin.dashboard.bundle import TABLES, Bundle, build_bundle, load_bundle, write_bundle
from youthtwin.growth import splines
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.recommend import rules
from youthtwin.sim.cohort import generate_cohort
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import is_watermarked

LABELS = ("y_growth_related", "y_non_contact_soft_tissue", "y_all_non_contact")


@pytest.fixture(scope="module")
def prompt_config() -> dict[str, Any]:
    return load_config("dashboard.yaml")["prompts"]


def _state(**overrides: Any) -> rules.AthleteWeek:
    """An unremarkable athlete-week: mid-cohort risk, tight intervals, recent screening."""
    fields: dict[str, Any] = {
        "risk": dict.fromkeys(LABELS, 0.02),
        "risk_percentile": dict.fromkeys(LABELS, 50.0),
        "risk_low": dict.fromkeys(LABELS, 0.018),
        "risk_high": dict.fromkeys(LABELS, 0.022),
        "height_velocity": 5.0,
        "maturity_offset": -0.5,
        "weeks_since_screen": 3.0,
    }
    fields.update(overrides)
    return rules.AthleteWeek(**fields)


# ----------------------------------------------------------------------------- prompts


def test_unremarkable_athlete_gets_no_prompts(prompt_config: dict[str, Any]) -> None:
    assert rules.prompts_for(_state(), prompt_config) == []


def test_every_possible_prompt_is_decision_support_framed(prompt_config: dict[str, Any]) -> None:
    """Trigger every rule at once and check wording and citations of everything produced."""
    everything = _state(
        risk_percentile=dict.fromkeys(LABELS, 99.0),
        risk_low=dict.fromkeys(LABELS, 0.001),
        risk_high=dict.fromkeys(LABELS, 0.2),
        height_velocity=9.0,
        maturity_offset=-4.0,
        weeks_since_screen=20.0,
    )
    prompts = rules.prompts_for(everything, prompt_config)
    assert {p.rule for p in prompts} == {
        "elevated_model_risk",
        "elevated_risk_during_rapid_growth",
        "uncertain_estimate",
        "growth_state_low_information",
        "screening_out_of_date",
    }
    evidence = rules.load_evidence()
    for prompt in prompts:
        assert rules.check_wording(prompt, evidence) == [], prompt.text
        assert "consider" in prompt.text.lower()
        assert prompt.evidence and set(prompt.evidence) <= set(evidence)
    elevated = [p for p in prompts if p.rule == "elevated_model_risk"]
    assert len(elevated) == 3  # one per outcome type, named in plain words (N6)
    assert any("growth-related injury" in p.text for p in elevated)
    assert all("not a finding about this athlete" in p.text for p in elevated)


@pytest.mark.parametrize("maturity_offset", [-6.0, -3.5, -1.0, 0.0, 0.5, 4.0])
@pytest.mark.parametrize("height_velocity", [1.0, 6.0, 11.0])
def test_maturity_and_growth_alone_never_signal_elevated_risk(
    prompt_config: dict[str, Any], maturity_offset: float, height_velocity: float
) -> None:
    """v2 set risk to CRITICAL from a maturity score alone. Here no growth value can do that."""
    state = _state(maturity_offset=maturity_offset, height_velocity=height_velocity)
    fired = {p.rule for p in rules.prompts_for(state, prompt_config)}
    assert "elevated_model_risk" not in fired
    assert "elevated_risk_during_rapid_growth" not in fired
    assert fired <= {"growth_state_low_information"}


def test_elevated_prompt_follows_model_risk_only(prompt_config: dict[str, Any]) -> None:
    high = _state(risk_percentile={**dict.fromkeys(LABELS, 50.0), "y_growth_related": 95.0})
    fired = [p.rule for p in rules.prompts_for(high, prompt_config)]
    assert fired == ["elevated_model_risk"]
    with_growth = _state(
        risk_percentile={**dict.fromkeys(LABELS, 50.0), "y_growth_related": 95.0},
        height_velocity=8.5,
    )
    assert "elevated_risk_during_rapid_growth" in {
        p.rule for p in rules.prompts_for(with_growth, prompt_config)
    }
    # Rapid growth with elevated risk of a *different* injury type does not add the growth prompt.
    other = _state(
        risk_percentile={**dict.fromkeys(LABELS, 50.0), "y_non_contact_soft_tissue": 95.0},
        height_velocity=8.5,
    )
    assert "elevated_risk_during_rapid_growth" not in {
        p.rule for p in rules.prompts_for(other, prompt_config)
    }


def test_wording_check_catches_bad_prompts() -> None:
    evidence = rules.load_evidence()
    bad = rules.Prompt("x", "This athlete has Osgood-Schlatter and must stop training.", ("nope",))
    problems = rules.check_wording(bad, evidence)
    assert "does not use 'consider' framing" in problems
    assert any("must" in p for p in problems) and any("unknown evidence" in p for p in problems)
    assert rules.check_wording(rules.Prompt("x", "Consider a review.", ()), evidence) == [
        "cites no evidence"
    ]


def test_evidence_library_is_marked_unverified_and_complete() -> None:
    library = load_config(rules.EVIDENCE_PATH)
    assert library["verified_by_human"] is False
    for key, entry in library["entries"].items():
        assert entry["statement"].strip() and entry["citation"].strip(), key
    source = rules.EVIDENCE_PATH.read_text(encoding="utf-8")
    assert "TODO(human)" in source


# ----------------------------------------------------------------------------- bundle and app


def _posterior(athletes: list[str], sex: str, seed: int) -> SitarPosterior:
    knots = (-4.0, -2.0, 0.0, 2.0, 4.0)
    z = np.linspace(-4, 4, 200)
    design = np.column_stack([z, *splines.basis(z, knots)])
    solution, *_ = np.linalg.lstsq(design, 5 * z + 6 * np.tanh(z), rcond=None)
    n, draws = len(athletes), 30
    ones = np.ones(draws)
    rng = np.random.default_rng(seed)
    return SitarPosterior(
        athlete_ids=tuple(athletes),
        knots=knots,
        intercept=150 * ones,
        linear=solution[0] * ones,
        coefs=np.tile(solution[1:], (draws, 1)),
        alpha=np.tile(rng.normal(0, 4, n), (draws, 1)),
        beta=rng.normal(13.0, 1.0, n)[None, :] + rng.normal(0, 0.4, (draws, n)),
        gamma=np.zeros((draws, n)),
        sigma=0.2 * ones,
        sigma_alpha=5 * ones,
        sigma_gamma=0 * ones,
        delta=np.zeros((draws, n)),
        sigma_delta=0 * ones,
        timing_offset=0 * ones,
        sigma_beta=ones,
        cohort_mean=np.nan * ones,
        cohort_sd=np.nan * ones,
        mean_age=np.full(n, 13.0),
        sex=sex,
        synthetic=True,
    )


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    cohort_config = load_config("sim/cohort.yaml")
    cohort_config.update(n_athletes=60, seed=5, follow_up_weeks=30)
    cohort = generate_cohort(cohort_config)
    athletes = cohort.tables["athletes"]
    posteriors = {
        sex: _posterior(sorted(athletes.loc[athletes["sex"] == sex, "id"]), sex, k)
        for k, sex in enumerate(("female", "male"))
    }
    dashboard = copy.deepcopy(load_config("dashboard.yaml"))
    dashboard.update(n_folds=2)
    dashboard["uncertainty"]["n_growth_draws"] = 6
    hazard = copy.deepcopy(load_config("models/hazard.yaml"))
    hazard["hazard_logit"]["max_iter"] = 200
    configs = {
        "dashboard": dashboard,
        "hazard": hazard,
        "person_period": load_config("features/person_period.yaml"),
    }
    return build_bundle(cohort, cohort_config, posteriors, configs, seed=0)


def test_bundle_tables(bundle: Bundle) -> None:
    assert set(bundle.tables) == set(TABLES)
    assert bundle.meta["synthetic"] is True and bundle.meta["n_athletes"] == 60
    for name, table in bundle.tables.items():
        assert table["synthetic"].all(), name
    risk = bundle.tables["risk"]
    assert set(risk["label"]) == set(LABELS)
    assert risk[["risk", "risk_low", "risk_high"]].notna().all().all()
    assert (
        (risk["risk_low"] <= risk["risk"] + 1e-9) & (risk["risk"] <= risk["risk_high"] + 1e-9)
    ).all()
    assert (risk["risk_high"] > risk["risk_low"]).mean() > 0.9  # growth uncertainty reaches risk
    assert risk["percentile"].between(0, 100).all()
    curve = bundle.tables["growth_curve"]
    assert (curve["stature_low"] <= curve["stature_high"]).all()
    assert bundle.tables["aphv"]["athlete_id"].nunique() == 60
    joints = set(bundle.tables["keypoints"]["joint"])
    assert "left_knee" in joints and len(joints) == 12
    assert bundle.athlete_ids() == sorted(bundle.tables["athletes"]["id"])
    prompts = bundle.tables["prompts"]
    assert prompts["text"].str.lower().str.contains("consider").all()
    assert (prompts["evidence"].str.len() > 10).all()


def test_no_identifying_or_video_fields_in_the_bundle(bundle: Bundle) -> None:
    columns = {c for table in bundle.tables.values() for c in table.columns}
    assert not columns & {"name", "date_of_birth", "dob", "video", "video_path", "keypoints_uri"}
    assert bundle.tables["athletes"]["dob_month"].str.fullmatch(r"\d{4}-\d{2}").all()


def test_figures_are_watermarked(bundle: Bundle) -> None:
    athlete = bundle.athlete_ids()[0]
    get = lambda name: bundle.for_athlete(name, athlete)  # noqa: E731
    made = [
        figures.growth_figure(
            get("measurements"), get("growth_curve"), get("aphv"), level=0.9, synthetic=True
        ),
        figures.velocity_figure(get("growth_curve"), level=0.9, synthetic=True),
        figures.risk_figure(
            get("risk"), get("injuries"), level=0.9, horizon_weeks=4, synthetic=True
        ),
        figures.contribution_figure(
            get("contributions")[get("contributions")["label"] == LABELS[0]], top=6, synthetic=True
        ),
        figures.skeleton_figure(get("keypoints"), 10, synthetic=True),
    ]
    assert all(is_watermarked(figure) for figure in made)
    real = figures.velocity_figure(get("growth_curve"), level=0.9, synthetic=False)
    assert not is_watermarked(real)
    plt.close("all")


def test_bundle_round_trip_and_app_runs_headlessly(
    bundle: Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    directory = write_bundle(bundle, tmp_path / "bundle", {"run_id": "test"})
    loaded = load_bundle(directory)
    assert loaded.meta["run_id"] == "test" and len(loaded.tables["risk"]) == len(
        bundle.tables["risk"]
    )
    with pytest.raises(FileNotFoundError, match="make dashboard-data"):
        load_bundle(tmp_path / "nothing")

    monkeypatch.setenv("YOUTHTWIN_DASHBOARD_BUNDLE", str(directory))
    app = AppTest.from_file(
        str(repo_root() / "apps" / "twin_dashboard" / "app.py"), default_timeout=120
    )
    app.run()
    assert not app.exception
    assert any("SYNTHETIC" in w.value for w in app.warning)
    assert [t.label for t in app.tabs] == [
        "Growth",
        "Injury risk",
        "Screening replay",
        "Decision support",
    ]
    assert "not diagnoses" in " ".join(c.value for c in app.caption)
    # Switching athlete re-renders without errors.
    app.sidebar.selectbox[0].select(loaded.athlete_ids()[-1]).run()
    assert not app.exception
