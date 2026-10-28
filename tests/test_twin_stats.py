"""Twin secondary analyses: cluster bootstrap, maturity timing groups, ML baseline separation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import ml_baseline, sequential, twin_stats
from youthtwin.utils.config import load_config


def _scores(n_children: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_children):
        truth = rng.normal(13.0, 1.0)
        for visit in range(1, 4):
            rows.append(
                {
                    "athlete_id": f"c{k}",
                    "sex": "male" if k % 2 else "female",
                    "visits": visit,
                    "reference_aphv": truth,
                    "twin_aphv": truth + rng.normal(0, 0.3),
                    "moore_aphv": truth + rng.normal(0, 0.8),
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_interval_contains_estimate_and_resamples_children() -> None:
    scores = _scores()
    result = twin_stats.cluster_bootstrap(
        scores,
        twin_stats.mae_difference("twin_aphv", "moore_aphv"),
        n_resamples=300,
        level=0.95,
        seed=1,
    )
    assert result["ci_low"] < result["estimate"] < result["ci_high"] < 0  # twin clearly better
    assert result["n_clusters"] == 60
    # Duplicating every row of a child does not change the child-level interval's centre much.
    doubled = pd.concat([scores, scores], ignore_index=True)
    again = twin_stats.cluster_bootstrap(
        doubled,
        twin_stats.mae_difference("twin_aphv", "moore_aphv"),
        n_resamples=300,
        level=0.95,
        seed=1,
    )
    assert again["estimate"] == pytest.approx(result["estimate"])


def test_timing_groups_are_within_sex() -> None:
    reference = pd.DataFrame(
        {
            "sex": ["female"] * 5 + ["male"] * 5,
            "reference_aphv": [10.0, 11.5, 11.5, 11.5, 13.0, 12.0, 13.7, 13.7, 13.7, 15.4],
        }
    )
    groups = twin_stats.timing_groups(reference, 1.0).tolist()
    assert groups == ["early", "average", "average", "average", "late"] * 2


def test_ml_training_uses_only_training_children() -> None:
    from youthtwin.external import berkeley  # noqa: F401  (module import check)

    ages = np.arange(8.0, 18.01, 0.5)
    rows = []
    for k in range(6):
        height = 125 + 6 * (ages - 8) + k
        rows.append(
            pd.DataFrame(
                {
                    "athlete_id": f"a{k}",
                    "sex": "male",
                    "age": ages,
                    "stature": height,
                    "sitting_height": 0.52 * height,
                    "mass": 25 + 3 * (ages - 8),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    config = load_config("external/sequential.yaml")
    reference = {f"a{k}": 13.5 + 0.1 * k for k in range(6)}
    train = ml_baseline.training_set(
        frame, ["a0", "a1", "a2"], reference, "male", config, entries=2, seed=0
    )
    assert set(train["target"].round(2)) <= {13.5, 13.6, 13.7}
    # Features after k visits use only those k visits.
    visits = sequential.entry_sequences(frame, ["a4"], config, seed=3)["a4"]
    first = ml_baseline.features(visits.head(1), "male")
    assert first["visits"] == 1 and np.isnan(first["velocity_mean"])
    two = ml_baseline.features(visits.head(2), "male")
    assert two["age_last"] == pytest.approx(visits["age"].iloc[1])


def test_width_thirds_split_each_cell_and_coverage_difference_pairs_rows() -> None:
    scores = pd.DataFrame(
        {
            "sex": ["male"] * 6 + ["female"] * 3,
            "visits": [1] * 3 + [2] * 3 + [1] * 3,
            "width": [3.0, 1.0, 2.0, 0.5, 0.7, 0.6, 9.0, 8.0, 7.0],
            "a": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, np.nan],
            "b": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    thirds = twin_stats.width_thirds(scores, "width")
    for _, cell in scores.assign(third=thirds).groupby(["sex", "visits"]):
        ordered = cell.sort_values("width")["third"].tolist()
        assert ordered == list(twin_stats.WIDTH_THIRDS)  # within a cell, not across cells
    # the row where ``a`` is missing is dropped from both sides
    assert twin_stats.coverage_difference("a", "b")(scores) == pytest.approx(6 / 8 - 2 / 8)
