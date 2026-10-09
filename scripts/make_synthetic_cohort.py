"""Generate the synthetic cohort (``make sim``). Output is flagged synthetic=True throughout."""

from __future__ import annotations

from pathlib import Path

import typer

from youthtwin.io.ingest import validate_tables
from youthtwin.sim.cohort import generate_cohort, write_cohort
from youthtwin.utils.config import load_config, repo_root


def main(
    config: str = "sim/cohort.yaml",
    out_dir: Path | None = None,
    seed: int | None = None,
) -> None:
    settings = load_config(config)
    if seed is not None:
        settings["seed"] = seed
    cohort = generate_cohort(settings)
    report = validate_tables(cohort.tables)
    if not report.ok:
        for error in report.errors[:10]:
            typer.echo(f"{error.table}[{error.row}]: {error.message}", err=True)
        raise typer.Exit(code=1)
    target = out_dir or repo_root() / "data" / "synthetic" / f"cohort_seed{settings['seed']}"
    write_cohort(cohort, target)
    typer.echo(f"SYNTHETIC cohort written to {target}")
    for name, n_rows in report.n_rows.items():
        typer.echo(f"  {name}: {n_rows} rows")


if __name__ == "__main__":
    typer.run(main)
