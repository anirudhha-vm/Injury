"""SITAR benchmark (M12): no leakage into the refits; recovery of age at PHV (R-gated)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from youthtwin.external import sequential, sitar_benchmark
from youthtwin.sim.growth_sim import pb1_height, pb1_peak_offset
from youthtwin.utils.config import load_config

CONFIG = load_config("external/sequential.yaml")
SETTINGS = load_config("external/sitar_benchmark.yaml")
AGES = np.arange(8.0, 18.01, 0.5)


def _frame(n: int, seed: int) -> pd.DataFrame:
    """Six-monthly Preece-Baines curves with known age at PHV."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        aphv = rng.normal(13.7, 1.0)
        s0, s1 = 0.11 * np.exp(rng.normal(0, 0.05)), 1.15 * np.exp(rng.normal(0, 0.08))
        theta = aphv - float(pb1_peak_offset(np.array([s0]), np.array([s1]))[0])
        one = np.array([1.0])
        h1 = rng.normal(178, 6)
        height = pb1_height(AGES, h1 * one, (h1 - 12.5) * one, s0 * one, s1 * one, theta * one)
        rows.append(
            pd.DataFrame(
                {
                    "athlete_id": f"b{k:02d}",
                    "sex": "male",
                    "age": AGES,
                    "stature": height + rng.normal(0, 0.3, AGES.size),
                    "sitting_height": height * 0.52,
                    "mass": 25.0 + 3.5 * (AGES - 8.0),
                    "reference": aphv,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_refit_inputs_never_leak_held_out_data() -> None:
    frame = _frame(12, 0)
    train, held_out = [f"b{k:02d}" for k in range(8)], [f"b{k:02d}" for k in range(8, 12)]
    training = sitar_benchmark.training_rows(frame, train, CONFIG, "stature")
    assert set(training["id"]) == set(train)
    sequences = sequential.entry_sequences(frame, held_out, CONFIG, seed=3)
    tasks = sitar_benchmark.task_rows(sequences, CONFIG, "stature")
    for task, rows in tasks.groupby("task"):
        athlete, k = str(task).split("|")
        assert set(rows["id"]) == {athlete}
        expected = sequences[athlete].head(int(k))
        assert len(rows) == int(k) == int(rows["visits"].iloc[0])
        np.testing.assert_allclose(rows["age"], expected["age"])  # only the first k visits
    counts = tasks.groupby("id")["visits"].max()
    for athlete, visits in sequences.items():
        assert counts[athlete] == min(len(visits), int(CONFIG["max_visits"]))


def _has_sitar() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = '.libPaths(c(path.expand("~/R/library"), .libPaths())); library(sitar)'
    return subprocess.run(["Rscript", "-e", probe], capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_sitar(), reason="needs R and the CRAN package sitar")
def test_refit_recovers_age_at_phv_on_sitar_like_curves() -> None:
    frame = _frame(40, 1)
    train = [f"b{k:02d}" for k in range(36)]
    held_out = [f"b{k:02d}" for k in range(36, 40)]
    full = {a: frame[frame["athlete_id"] == a].reset_index(drop=True) for a in held_out}
    tasks = sitar_benchmark.task_rows(
        {a: v[v["age"].between(11.0, 16.0)] for a, v in full.items()},
        {**CONFIG, "max_visits": 11},
        "stature",
    )
    tasks = tasks[tasks["visits"] == tasks.groupby("id")["visits"].transform("max")]
    out = sitar_benchmark.refit(
        sitar_benchmark.training_rows(frame, train, CONFIG, "stature"),
        tasks,
        SETTINGS,
        CONFIG["train_age_range_years"],
    )
    truth = frame.groupby("athlete_id")["reference"].first()
    assert (out["sitar_status"] == "ok").all()
    error = (out["sitar_aphv"] - out["athlete_id"].map(truth)).abs()
    assert error.max() < 0.4  # five years of visits around the spurt pin the peak down
