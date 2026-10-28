"""Build the growth-twin manuscript (Path A): values, tables and figure from runs, then the PDF.

Reads ``outputs/real_twin_sequential/`` (runs told apart by their ``settings``) and the latest
``outputs/real_twin_external_chard/`` run; writes ``paper/tables/twin_*`` and
``paper/figures/twin_fig_*`` with provenance sidecars; compiles ``paper/ieee_twin/main.tex`` with
Tectonic (``--tectonic``, ``$TECTONIC`` or PATH) into ``paper/ieee_twin/GrowthTwin_IEEE.pdf``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import typer

from youthtwin.paper import twin_paper
from youthtwin.paper.ieee import PENDING
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import PROVENANCE_SUFFIX


def _sidecar(path: Path, runs: dict[str, str]) -> None:
    record = {"source": "scripts/build_twin_paper.py", "runs": runs, "synthetic": False}
    path.with_name(path.name + PROVENANCE_SUFFIX).write_text(json.dumps(record, indent=2), "utf-8")


def _latest(kind: str) -> Path | None:
    root = repo_root() / "outputs" / kind
    runs = [p for p in root.glob("*") if (p / "run.json").is_file()] if root.is_dir() else []
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def main(tectonic: Path | None = None) -> None:
    configs = {
        "sequential": load_config("external/sequential.yaml"),
        "analysis": load_config("external/twin_analysis.yaml"),
        "sitar": load_config("external/sitar_benchmark.yaml"),
    }
    values, tables, runs = twin_paper.build(
        repo_root() / "outputs" / "real_twin_sequential",
        _latest("real_twin_external_chard"),
        configs,
        analysis=_latest("real_twin_analysis"),
    )
    paper = repo_root() / "paper"
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    target = paper / "tables" / "twin_values.tex"
    target.write_text(values.render(), encoding="utf-8")
    _sidecar(target, values.runs)
    for name, text in tables.items():
        (paper / "tables" / name).write_text(text, encoding="utf-8")
        _sidecar(paper / "tables" / name, values.runs)
    figure = paper / "figures" / "twin_fig_sequential.png"
    main_run = runs.get(twin_paper.MAIN)
    if main_run is not None and (main_run / "twin_sequential.png").is_file():
        shutil.copyfile(main_run / "twin_sequential.png", figure)
        _sidecar(figure, {"twin_main": values.runs.get("twin_main", "")})
    for variant in twin_paper.VARIANTS:
        typer.echo(f"  {variant}: {runs[variant].name if variant in runs else 'pending'}")
    pending = [n for n, (text, _) in values.macros.items() if text == PENDING]
    typer.echo(f"{len(values.macros)} values, {len(pending)} pending: {sorted(pending)[:12]}")

    engine = tectonic or (Path(os.environ["TECTONIC"]) if "TECTONIC" in os.environ else None)
    executable = str(engine) if engine else shutil.which("tectonic")
    if executable is None:
        typer.echo("Tectonic not found: tables and figure written, PDF not built")
        return
    build = paper / "ieee_twin" / "build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [executable, "--outdir", str(build), str(paper / "ieee_twin" / "main.tex")], check=True
    )
    pdf = paper / "ieee_twin" / "GrowthTwin_IEEE.pdf"
    shutil.copyfile(build / "main.pdf", pdf)
    typer.echo(f"PDF written to {pdf}")


if __name__ == "__main__":
    typer.run(main)
