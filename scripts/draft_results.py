"""Write ``paper/results_draft.md`` from the latest run of each kind under ``outputs/``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from youthtwin.paper.results import SECTIONS, render_results
from youthtwin.utils.config import repo_root


def _latest(kind: str) -> Path | None:
    root = repo_root() / "outputs" / kind
    runs = [p for p in root.glob("*") if (p / "run.json").is_file()] if root.is_dir() else []
    if kind == "evaluation":  # only full-size runs with the confirmatory model belong in the paper
        runs = [
            p for p in runs if "confirmatory_predictors" in json.loads((p / "run.json").read_text())
        ]
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def main(out: Path | None = None) -> None:
    runs = {kind: _latest(kind) for kind, *_ in SECTIONS}
    target = out or repo_root() / "paper" / "results_draft.md"
    target.write_text(render_results(runs), encoding="utf-8")
    typer.echo(f"results draft written to {target}")
    for kind, run in runs.items():
        typer.echo(f"  {kind}: {run.name if run else 'not available'}")


if __name__ == "__main__":
    typer.run(main)
