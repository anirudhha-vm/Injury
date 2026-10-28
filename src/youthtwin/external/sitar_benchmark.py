"""Classical SITAR benchmark for the growth twin (M12).

SITAR (Cole et al. 2010) as implemented in the R package ``sitar`` (``nlme`` fit), used as a
practitioner would: for each held-out child and number of visits ``k``, refit on the fold's
training children (full records) plus the child's first ``k`` visits, and take the child's age at
PHV from the peak of its fitted velocity curve. The R side is ``scripts/r/sitar_refit.R``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from youthtwin.external import sequential
from youthtwin.utils.config import repo_root

R_SCRIPT = Path("scripts/r/sitar_refit.R")


def training_rows(
    frame: pd.DataFrame, athletes: Sequence[str], config: Mapping[str, Any], measure: str
) -> pd.DataFrame:
    """Full records (``train_age_range_years``) of the training children: id, age, y."""
    low, high = (float(v) for v in config["train_age_range_years"])
    rows = sequential.complete(frame)
    rows = rows[rows["athlete_id"].isin(athletes) & rows["age"].between(low, high)]
    return pd.DataFrame(
        {"id": rows["athlete_id"].astype(str), "age": rows["age"], "y": rows[measure]}
    ).reset_index(drop=True)


def task_rows(
    sequences: Mapping[str, pd.DataFrame], config: Mapping[str, Any], measure: str
) -> pd.DataFrame:
    """One task per held-out child and visit count, holding only that child's first k visits."""
    parts = []
    for athlete, visits in sorted(sequences.items()):
        for k in range(1, min(len(visits), int(config["max_visits"])) + 1):
            seen = visits.head(k)
            parts.append(
                pd.DataFrame(
                    {
                        "task": f"{athlete}|{k}",
                        "id": str(athlete),
                        "visits": k,
                        "age": seen["age"].to_numpy(),
                        "y": seen[measure].to_numpy(),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def rscript() -> str:
    path = shutil.which("Rscript")
    if path is None:
        raise RuntimeError("Rscript not found: install R and the CRAN package 'sitar'")
    return path


def refit(
    training: pd.DataFrame,
    tasks: pd.DataFrame,
    settings: Mapping[str, Any],
    age_range: Sequence[float],
) -> pd.DataFrame:
    """Run the R refits; returns athlete_id, visits, sitar_aphv, sitar_status."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        training.to_csv(folder / "training.csv", index=False)
        tasks.to_csv(folder / "tasks.csv", index=False)
        result = subprocess.run(
            [
                rscript(),
                str(repo_root() / R_SCRIPT),
                str(folder / "training.csv"),
                str(folder / "tasks.csv"),
                str(folder / "out.csv"),
                str(int(settings["spline_df"])),
                str(float(settings["peak_grid_step_years"])),
                str(float(age_range[0])),
                str(float(age_range[1])),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SITAR refit failed in R:\n{result.stderr[-2000:]}")
        out = pd.read_csv(folder / "out.csv", dtype={"id": str})
    return out.rename(columns={"id": "athlete_id", "status": "sitar_status"})[
        ["athlete_id", "visits", "sitar_aphv", "sitar_status"]
    ]
