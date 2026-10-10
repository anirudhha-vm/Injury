"""Data bundle behind the research dashboard.

``build_bundle`` turns a cohort, its fitted growth posteriors and the risk model into a handful
of tidy tables; the Streamlit app only reads them. Building is separate from viewing so that
the app starts instantly and so that everything shown can be inspected as a file.

What the bundle holds per athlete:

- measurements and the fitted growth curve with a posterior band; APHV with its interval;
- **out-of-fold** 4-week risk per injury type for every week (each athlete is predicted by a
  model that never saw them), with an interval from pushing growth-posterior draws through
  that model;
- which inputs moved the latest prediction (SHAP, log-odds);
- decision-support prompts with their evidence;
- keypoints of one screening trial for replay. No video is involved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from youthtwin.eval import splits
from youthtwin.explain import shap_utils
from youthtwin.features.person_period import TWIN_PREFIX, build_person_period
from youthtwin.growth.growth_state import SitarPosterior, age_on
from youthtwin.models.dataset import growth_draw_arrays
from youthtwin.models.risk.hazard_logit import make_hazard_logit
from youthtwin.pose.skeletons import JOINTS, SIDES
from youthtwin.recommend import rules
from youthtwin.sim import movement_sim
from youthtwin.sim.cohort import SyntheticCohort
from youthtwin.utils.seeding import make_rng

TABLES = (
    "athletes",
    "measurements",
    "growth_curve",
    "aphv",
    "risk",
    "contributions",
    "prompts",
    "keypoints",
    "injuries",
)
META_FILE = "bundle.json"


@dataclass(frozen=True)
class Bundle:
    tables: dict[str, pd.DataFrame]
    meta: dict[str, Any]

    def athlete_ids(self) -> list[str]:
        return sorted(self.tables["athletes"]["id"])

    def for_athlete(self, name: str, athlete_id: str) -> pd.DataFrame:
        table = self.tables[name]
        key = "id" if name == "athletes" else "athlete_id"
        return table[table[key] == athlete_id]


def _measurements(cohort: SyntheticCohort) -> pd.DataFrame:
    athletes = cohort.tables["athletes"].set_index("id")
    visits = cohort.tables["anthropometrics"].sort_values(["athlete_id", "date"])
    out = visits[["athlete_id", "date"]].copy()
    out["age"] = [
        age_on(athletes.loc[a, "dob_month"], pd.Timestamp(d).date())
        for a, d in zip(visits["athlete_id"], visits["date"], strict=True)
    ]
    for measure in ("stature", "sitting_height", "mass"):
        out[measure] = visits[[f"{measure}_1", f"{measure}_2"]].mean(axis=1).to_numpy()
    return out


def _growth_tables(
    measurements: pd.DataFrame, posteriors: Mapping[str, SitarPosterior], config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_cfg, level = config["growth_curve"], float(config["uncertainty"]["interval_level"])
    tail = (1.0 - level) / 2.0
    curves, timing = [], []
    for posterior in posteriors.values():
        aphv = posterior.aphv()
        for k, athlete in enumerate(posterior.athlete_ids):
            ages = measurements.loc[measurements["athlete_id"] == athlete, "age"]
            if ages.empty:
                continue
            grid = np.arange(
                ages.min() - float(grid_cfg["years_before_first_visit"]),
                ages.max() + float(grid_cfg["years_after_last_visit"]),
                float(grid_cfg["age_step_years"]),
            )
            stature = np.stack([posterior.stature(athlete, float(a)) for a in grid], axis=1)
            velocity = posterior.draws_at(athlete, grid)["height_velocity"]
            frame = pd.DataFrame({"athlete_id": athlete, "age": grid})
            for name, draws in (("stature", stature), ("velocity", velocity)):
                frame[f"{name}_mean"] = draws.mean(axis=0)
                frame[f"{name}_low"], frame[f"{name}_high"] = np.quantile(
                    draws, [tail, 1.0 - tail], axis=0
                )
            curves.append(frame)
            low, high = np.quantile(aphv[:, k], [tail, 1.0 - tail])
            timing.append(
                {
                    "athlete_id": athlete,
                    "aphv_mean": float(aphv[:, k].mean()),
                    "aphv_low": float(low),
                    "aphv_high": float(high),
                }
            )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(timing)


def _risk_tables(
    frame: pd.DataFrame,
    features: list[str],
    posteriors: Mapping[str, SitarPosterior],
    configs: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold risk with growth-uncertainty intervals, and SHAP for each athlete's last week."""
    dashboard, hazard = configs["dashboard"], configs["hazard"]
    level = float(dashboard["uncertainty"]["interval_level"])
    tail = (1.0 - level) / 2.0
    twin_means = [c for c in features if c.startswith(TWIN_PREFIX) and c.endswith("_mean")]
    draws = growth_draw_arrays(
        frame,
        posteriors,
        twin_means,
        n_draws=int(dashboard["uncertainty"]["n_growth_draws"]),
        seed=seed,
    )
    latest = frame.groupby("athlete_id")["week_start"].transform("max") == frame["week_start"]
    risks, contributions = [], []
    for label in dashboard["labels"]:
        folds = splits.grouped_stratified_folds(
            frame["athlete_id"], frame[label], n_splits=int(dashboard["n_folds"]), seed=seed
        )
        part = frame[["athlete_id", "week_start"]].copy()
        part["label"] = label
        for column in ("risk", "risk_low", "risk_high"):
            part[column] = np.nan
        for fold in folds:
            model = make_hazard_logit(
                frame, features, hazard["hazard_logit"], configs["person_period"], seed=seed
            ).fit(frame[fold.train], label)
            test = frame[fold.test]
            rows = np.flatnonzero(fold.test)
            samples = np.stack(
                [
                    model.predict_proba(
                        test.assign(**{c: draw[rows, k] for k, c in enumerate(twin_means)})
                    )
                    for draw in draws
                ]
            )
            part.loc[fold.test, "risk"] = samples.mean(axis=0)
            low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
            part.loc[fold.test, "risk_low"], part.loc[fold.test, "risk_high"] = low, high
            last = test[latest[fold.test].to_numpy()]
            explained = shap_utils.aggregate_to_inputs(
                shap_utils.explain(model, last, frame[fold.train]), model.features
            )
            tidy = explained.assign(athlete_id=last["athlete_id"].to_numpy()).melt(
                id_vars="athlete_id", var_name="input", value_name="shap_log_odds"
            )
            contributions.append(tidy.assign(label=label))
        part["percentile"] = part.groupby("week_start")["risk"].rank(pct=True) * 100.0
        risks.append(part)
    return pd.concat(risks, ignore_index=True), pd.concat(contributions, ignore_index=True)


def _floats(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.items()}


def _prompts(frame: pd.DataFrame, risk: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    evidence = rules.load_evidence()
    last_week = frame.groupby("athlete_id")["week_start"].transform("max") == frame["week_start"]
    current = frame[last_week].set_index("athlete_id")
    latest_risk = risk.merge(
        current.reset_index()[["athlete_id", "week_start"]], on=["athlete_id", "week_start"]
    )
    rows: list[dict[str, Any]] = []
    for athlete, part in latest_risk.groupby("athlete_id"):
        by_label = part.set_index("label")
        info = current.loc[athlete]

        def optional(column: str, row: Any = info) -> float | None:
            value = row.get(column)
            return None if value is None or pd.isna(value) else float(value)

        state = rules.AthleteWeek(
            risk=_floats(by_label["risk"]),
            risk_percentile=_floats(by_label["percentile"]),
            risk_low=_floats(by_label["risk_low"]),
            risk_high=_floats(by_label["risk_high"]),
            height_velocity=optional("twin_height_velocity_mean")
            or optional("obs_height_velocity"),
            maturity_offset=optional("twin_maturity_offset_mean")
            or optional("moore_maturity_offset"),
            weeks_since_screen=optional("weeks_since_screen"),
        )
        for prompt in rules.prompts_for(state, config["prompts"]):
            problems = rules.check_wording(prompt, evidence)
            if problems:
                raise ValueError(f"Prompt '{prompt.rule}' is not acceptable: {problems}")
            rows.append(
                {
                    "athlete_id": athlete,
                    "rule": prompt.rule,
                    "text": prompt.text,
                    "evidence": "; ".join(evidence[e]["citation"] for e in prompt.evidence),
                }
            )
    return pd.DataFrame(rows, columns=["athlete_id", "rule", "text", "evidence"])


def _keypoints(cohort: SyntheticCohort, cohort_config: Mapping[str, Any]) -> pd.DataFrame:
    """Side-view keypoints of one simulated trial from each athlete's latest session."""
    sessions = cohort.tables["screening_sessions"].sort_values("date").groupby("athlete_id").tail(1)
    features = cohort.tables["task_features"]
    load = features[features["feature"] == movement_sim.LOAD_PROXY].set_index("session_id")["value"]
    rng = make_rng(int(cohort_config["seed"]), "dashboard_replay")
    one_trial = {
        **cohort_config["movement"],
        "sequences": {**cohort_config["movement"]["sequences"], "trials_per_session": 1},
    }
    frames = []
    for session_id, athlete in zip(sessions["id"], sessions["athlete_id"], strict=True):
        (trial,) = movement_sim.session_trials(rng, float(load.loc[session_id]), one_trial)
        sequence = trial.sagittal
        for side in SIDES:
            for joint in JOINTS:
                xy = sequence.joint(side, joint)
                frames.append(
                    pd.DataFrame(
                        {
                            "athlete_id": athlete,
                            "frame": np.arange(sequence.n_frames),
                            "time_s": sequence.time_s,
                            "joint": f"{side}_{joint}",
                            "x": xy[:, 0],
                            "y": xy[:, 1],
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def build_bundle(
    cohort: SyntheticCohort,
    cohort_config: Mapping[str, Any],
    posteriors: Mapping[str, SitarPosterior],
    configs: Mapping[str, Any],
    *,
    seed: int,
) -> Bundle:
    """Build every table. ``configs``: ``dashboard, hazard, person_period`` YAML blocks."""
    period = build_person_period(
        cohort.tables,
        config=configs["person_period"],
        hazard_config=configs["hazard"],
        posteriors=posteriors,
    )
    frame = period.frame
    measurements = _measurements(cohort)
    growth_curve, aphv = _growth_tables(measurements, posteriors, configs["dashboard"])
    risk, contributions = _risk_tables(frame, period.predictors, posteriors, configs, seed)
    tables = {
        "athletes": cohort.tables["athletes"][["id", "sex", "sport", "academy_id", "dob_month"]],
        "measurements": measurements,
        "growth_curve": growth_curve,
        "aphv": aphv,
        "risk": risk,
        "contributions": contributions,
        "prompts": _prompts(frame, risk, configs["dashboard"]),
        "keypoints": _keypoints(cohort, cohort_config),
        "injuries": cohort.tables["injuries"][
            ["athlete_id", "onset_date", "outcome_group", "time_loss_days"]
        ],
    }
    synthetic = bool(cohort.metadata["synthetic"])
    for table in tables.values():
        table["synthetic"] = synthetic
    meta = {
        "synthetic": synthetic,
        "n_athletes": len(tables["athletes"]),
        "horizon_weeks": period.horizon_weeks,
        "interval_level": float(configs["dashboard"]["uncertainty"]["interval_level"]),
        "risk_model": "hazard_logit, out-of-fold",
        "person_period": period.summary,
    }
    return Bundle(tables, meta)


def write_bundle(bundle: Bundle, out_dir: Path, metadata: Mapping[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in bundle.tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {**bundle.meta, **metadata}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out_dir


def load_bundle(directory: Path) -> Bundle:
    if not (directory / META_FILE).is_file():
        raise FileNotFoundError(
            f"No dashboard bundle in {directory}. Build one with `make dashboard-data`."
        )
    tables = {name: pd.read_parquet(directory / f"{name}.parquet") for name in TABLES}
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    return Bundle(tables, meta)
