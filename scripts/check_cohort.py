"""Validate a cohort directory against the schema and run anthropometry quality control.

First command to run on real data. Reads one parquet file per table (the layout written by
``make sim``), reports invalid rows, and writes QC flags and technical error of measurement per
measurer to ``<cohort_dir>/qc/``. Flags are for a person to review; nothing is changed.
"""

from __future__ import annotations

from pathlib import Path

import typer

from youthtwin.io import qc
from youthtwin.io.ingest import read_cohort_dir
from youthtwin.utils.config import load_config


def main(cohort_dir: Path) -> None:
    tables = read_cohort_dir(cohort_dir)  # raises with the first problems if anything is invalid
    synthetic = bool(tables["athletes"]["synthetic"].all())
    typer.echo(("SYNTHETIC " if synthetic else "") + f"cohort {cohort_dir}: schema OK")
    for name, table in tables.items():
        typer.echo(f"  {name}: {len(table)} rows")
    target = cohort_dir / "qc"
    target.mkdir(exist_ok=True)
    flags = qc.anthropometry_flags(tables["anthropometrics"], load_config("io/qc.yaml"))
    flags.to_csv(target / "anthropometry_flags.csv", index=False)
    qc.tem_by_measurer(tables["anthropometrics"]).to_csv(
        target / "tem_by_measurer.csv", index=False
    )
    typer.echo(f"{len(flags)} anthropometry flag(s) for review -> {target}")
    if not flags.empty:
        typer.echo(flags["flag"].value_counts().to_string())


if __name__ == "__main__":
    typer.run(main)
