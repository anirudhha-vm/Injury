"""Decision-support prompts from model outputs (non-negotiable N9).

Each prompt is a sentence for a qualified person to consider, with the evidence entry it rests
on. Prompts are never diagnoses, instructions, or statements about what will happen.

Design constraints, each enforced by a test:

- every prompt contains the word "consider" and no diagnostic or prescriptive wording;
- every prompt cites at least one entry of ``evidence_library.yaml``;
- **no prompt about elevated risk can be triggered by maturity or growth alone**: the
  elevated-risk prompt depends only on the model's predicted risk relative to the cohort. (v2
  had a rule that set risk to "critical" from a maturity score regardless of anything else.)

Thresholds come from ``configs/dashboard.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from youthtwin.utils.config import load_config

EVIDENCE_PATH = Path(__file__).with_name("evidence_library.yaml")
FORBIDDEN_WORDING = (
    "diagnos",
    "must",
    "should not",
    "do not train",
    "stop training",
    "prescri",
    "will be injured",
    "will get injured",
    "is injured",
    "has osgood",
    "has sever",
    "critical",
)
OUTCOME_NAMES = {
    "y_growth_related": "growth-related injury",
    "y_non_contact_soft_tissue": "non-contact soft-tissue injury",
    "y_all_non_contact": "any non-contact injury",
}


@dataclass(frozen=True)
class Prompt:
    rule: str
    text: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class AthleteWeek:
    """What the rules are allowed to look at for one athlete at one prediction time."""

    risk: Mapping[str, float]  # predicted 4-week risk per outcome label
    risk_percentile: Mapping[str, float]  # 0-100, within the cohort at that week
    risk_low: Mapping[str, float]
    risk_high: Mapping[str, float]
    height_velocity: float | None = None  # cm/year
    maturity_offset: float | None = None  # years
    weeks_since_screen: float | None = None


def load_evidence() -> dict[str, dict[str, str]]:
    library = load_config(EVIDENCE_PATH)
    return {str(k): dict(v) for k, v in library["entries"].items()}


def prompts_for(state: AthleteWeek, config: Mapping[str, Any]) -> list[Prompt]:
    """All prompts that apply to ``state``. ``config`` is the ``prompts`` block."""
    out: list[Prompt] = []
    cut = float(config["elevated_risk_percentile"])
    elevated = [label for label, pct in state.risk_percentile.items() if pct >= cut]
    for label in elevated:
        name = OUTCOME_NAMES.get(label, label)
        out.append(
            Prompt(
                "elevated_model_risk",
                f"The model's estimated 4-week risk of {name} is in the top "
                f"{100 - cut:.0f}% of the cohort this week. Consider a review by the "
                "physiotherapist. This is a model estimate with limited accuracy, not a finding "
                "about this athlete.",
                ("screening_limits", "prediction_model_reporting"),
            )
        )
    velocity_cut = float(config["rapid_growth_velocity_cm_per_year"])
    if (
        "y_growth_related" in elevated
        and state.height_velocity is not None
        and state.height_velocity >= velocity_cut
    ):
        out.append(
            Prompt(
                "elevated_risk_during_rapid_growth",
                f"Estimated height velocity is about {state.height_velocity:.0f} cm/year "
                "alongside an elevated model estimate for growth-related injury. Consider "
                "discussing training load during this growth phase with the coach and "
                "physiotherapist.",
                ("growth_load_and_injury", "maturity_and_injury_type"),
            )
        )
    ratio = float(config["wide_interval_ratio"])
    uncertain = [
        label
        for label, risk in state.risk.items()
        if risk > 0 and (state.risk_high[label] - state.risk_low[label]) > ratio * risk
    ]
    if uncertain:
        out.append(
            Prompt(
                "uncertain_estimate",
                "The risk estimate is uncertain because the timing of this athlete's growth "
                "spurt is uncertain. Consider treating the estimate as a rough indication and "
                "repeating the height and sitting-height measurements at the next visit.",
                ("maturity_estimate_uncertainty", "prediction_model_reporting"),
            )
        )
    low_information = float(config["low_information_maturity_offset"])
    if state.maturity_offset is not None and state.maturity_offset < low_information:
        out.append(
            Prompt(
                "growth_state_low_information",
                "This athlete is estimated to be several years before the growth spurt, where "
                "its timing cannot be estimated reliably. Consider the growth-related outputs "
                "as provisional until more measurements are available.",
                ("maturity_estimate_uncertainty",),
            )
        )
    stale = float(config["stale_screen_weeks"])
    if state.weeks_since_screen is not None and state.weeks_since_screen > stale:
        out.append(
            Prompt(
                "screening_out_of_date",
                f"The latest movement screening is {state.weeks_since_screen:.0f} weeks old. "
                "Consider repeating the screening so that estimates use current data.",
                ("prediction_model_reporting",),
            )
        )
    return out


def check_wording(prompt: Prompt, evidence: Mapping[str, Any]) -> list[str]:
    """Problems with a prompt's wording or citations (empty if it is acceptable)."""
    problems = []
    lowered = prompt.text.lower()
    if "consider" not in lowered:
        problems.append("does not use 'consider' framing")
    problems += [f"contains '{word}'" for word in FORBIDDEN_WORDING if word in lowered]
    if not prompt.evidence:
        problems.append("cites no evidence")
    problems += [f"cites unknown evidence '{e}'" for e in prompt.evidence if e not in evidence]
    return problems
