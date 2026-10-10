"""YouthTwin research dashboard (local only).

Run with ``make dashboard`` after ``make dashboard-data``. Reads a bundle directory built by
``scripts/build_dashboard.py``; it never reads video and has no authentication or deployment
configuration, because it is a research viewer, not a product.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from youthtwin.dashboard import figures
from youthtwin.dashboard.bundle import Bundle, load_bundle
from youthtwin.recommend.rules import OUTCOME_NAMES
from youthtwin.utils.config import load_config, repo_root

BUNDLE_ENV = "YOUTHTWIN_DASHBOARD_BUNDLE"
DEFAULT_BUNDLE = Path("outputs") / "dashboard" / "bundle"


@st.cache_resource
def _bundle(directory: str) -> Bundle:
    return load_bundle(Path(directory))


def _show(figure: plt.Figure) -> None:
    st.pyplot(figure)
    plt.close(figure)


def main() -> None:
    st.set_page_config(page_title="YouthTwin research dashboard", layout="wide")
    directory = os.environ.get(BUNDLE_ENV, str(repo_root() / DEFAULT_BUNDLE))
    try:
        bundle = _bundle(directory)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    meta = bundle.meta
    synthetic = bool(meta["synthetic"])
    level = float(meta["interval_level"])
    settings = load_config("dashboard.yaml")

    st.title("YouthTwin research dashboard")
    if synthetic:
        st.warning(
            "SYNTHETIC DATA. Every athlete, measurement, injury and risk shown here is simulated "
            "to test the software. Nothing on this page is a finding about real athletes, and "
            "on this simulated cohort the risk models are close to chance."
        )
    st.caption(
        "Research tool for decision support. Outputs are model estimates with uncertainty, "
        "not diagnoses or instructions."
    )

    ids = bundle.athlete_ids()
    athlete = st.sidebar.selectbox("Athlete (pseudonymous ID)", ids)
    info = bundle.for_athlete("athletes", athlete).iloc[0]
    st.sidebar.markdown(
        f"**Sex:** {info['sex']}  \n**Sport:** {info['sport']}  \n**Academy:** {info['academy_id']}"
    )
    st.sidebar.caption(f"{meta['n_athletes']} athletes in this bundle")

    measurements = bundle.for_athlete("measurements", athlete)
    curve = bundle.for_athlete("growth_curve", athlete)
    aphv = bundle.for_athlete("aphv", athlete)
    risk = bundle.for_athlete("risk", athlete)
    injuries = bundle.for_athlete("injuries", athlete)

    growth_tab, risk_tab, screening_tab, prompts_tab = st.tabs(
        ["Growth", "Injury risk", "Screening replay", "Decision support"]
    )

    with growth_tab:
        if curve.empty:
            st.info("No fitted growth curve for this athlete.")
        else:
            if not aphv.empty:
                row = aphv.iloc[0]
                age_now = float(measurements["age"].max())
                left, middle, right = st.columns(3)
                left.metric("Estimated age at peak height velocity", f"{row['aphv_mean']:.1f} y")
                middle.metric(
                    f"{level:.0%} interval", f"{row['aphv_low']:.1f} to {row['aphv_high']:.1f} y"
                )
                right.metric(
                    "Years from peak height velocity now", f"{age_now - row['aphv_mean']:+.1f}"
                )
            _show(
                figures.growth_figure(measurements, curve, aphv, level=level, synthetic=synthetic)
            )
            _show(figures.velocity_figure(curve, level=level, synthetic=synthetic))
            st.caption(
                "Timing of the growth spurt is least certain for children several years before "
                "it; treat those estimates as provisional."
            )

    with risk_tab:
        if risk.empty:
            st.info("No risk estimates for this athlete.")
        else:
            st.caption(
                f"Predicted probability of an injury starting in the following "
                f"{meta['horizon_weeks']} weeks. Each athlete is predicted by a model that was "
                "fitted without their data."
            )
            _show(
                figures.risk_figure(
                    risk,
                    injuries,
                    level=level,
                    horizon_weeks=int(meta["horizon_weeks"]),
                    synthetic=synthetic,
                )
            )
            contributions = bundle.for_athlete("contributions", athlete)
            labels = list(contributions["label"].unique())
            label = st.selectbox(
                "Inputs behind the latest estimate for",
                labels,
                format_func=lambda value: OUTCOME_NAMES.get(value, value),
            )
            _show(
                figures.contribution_figure(
                    contributions[contributions["label"] == label],
                    top=int(settings["shap"]["top_inputs"]),
                    synthetic=synthetic,
                )
            )
            st.caption(
                "Contributions describe how the model used its inputs for this estimate. They "
                "are not causes of injury."
            )

    with screening_tab:
        keypoints = bundle.for_athlete("keypoints", athlete)
        if keypoints.empty:
            st.info("No screening keypoints for this athlete.")
        else:
            last = int(keypoints["frame"].max())
            frame = st.slider("Frame", 0, last, min(last, last // 3))
            _show(figures.skeleton_figure(keypoints, frame, synthetic=synthetic))
            st.caption(
                "Drawn from stored keypoints of the latest drop-jump trial. No video is stored."
            )

    with prompts_tab:
        prompts = bundle.for_athlete("prompts", athlete)
        if prompts.empty:
            st.info("No decision-support prompts for this athlete this week.")
        for row in prompts.itertuples(index=False):
            st.markdown(f"**{row.text}**")
            st.caption(f"Evidence: {row.evidence}")
        st.caption(
            "Prompts are suggestions for a qualified person to consider. The evidence statements "
            "have not yet been verified against the original papers."
        )


main()
