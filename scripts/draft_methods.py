"""Regenerate ``paper/methods_draft.md`` from the configuration files."""

from __future__ import annotations

from pathlib import Path

import typer

from youthtwin.paper.methods import load_all, render_methods
from youthtwin.utils.config import repo_root


def main(out: Path | None = None) -> None:
    target = out or repo_root() / "paper" / "methods_draft.md"
    target.write_text(render_methods(load_all()), encoding="utf-8")
    typer.echo(f"methods draft written to {target}")


if __name__ == "__main__":
    typer.run(main)
