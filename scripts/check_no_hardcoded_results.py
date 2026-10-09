"""Fail if ``paper/`` has numbers not tagged with a run ID (youthtwin.utils.results_guard)."""

from __future__ import annotations

from pathlib import Path

import typer

from youthtwin.utils.config import repo_root
from youthtwin.utils.results_guard import scan_paper_dir


def main(paper_dir: Path | None = None) -> None:
    target = paper_dir if paper_dir is not None else repo_root() / "paper"
    violations = scan_paper_dir(target)
    for violation in violations:
        typer.echo(violation.format(), err=True)
    if violations:
        typer.echo(
            f"\n{len(violations)} problem(s). Wrap numbers in [[RUN:<run_id>|value]], "
            "[[SPEC: ...]] or [[CITED: ...]], or use a [[RESULT: ...]] placeholder.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"paper/ results guard: OK ({target})")


if __name__ == "__main__":
    typer.run(main)
