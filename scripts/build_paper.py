"""Build the IEEE manuscript: generated values, tables and figures, then the PDF.

1. Finds the latest run of each kind under ``outputs/`` (the evaluation run must include the
   confirmatory model; the ablation run must have paired comparisons).
2. Writes ``paper/tables/ieee_values.tex`` and ``paper/tables/ieee_tab_*.tex`` and copies the
   figures to ``paper/figures/``, each with a provenance sidecar naming the runs.
3. Compiles ``paper/ieee/main.tex`` with Tectonic (``--tectonic`` or ``$TECTONIC`` or PATH) into
   ``paper/ieee/YouthTwin_IEEE.pdf``. Without Tectonic, stops after step 2.

Values from runs that do not exist yet print as a red [pending] marker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import typer

from youthtwin.paper import ieee
from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import PROVENANCE_SUFFIX

FIGURES = {  # (output kind, file in the run) -> name under paper/figures
    ("real_growth_berkeley", "aphv_real.png"): "ieee_fig_growth.png",
    ("real_risk_runners", "calibration.png"): "ieee_fig_runners_calibration.png",
    ("sim_study_validation", "rmse_by_maturity.png"): "ieee_fig_validity_rmse.png",
    ("sim_study_validation", "bland_altman.png"): "ieee_fig_validity_ba.png",
    ("sim_study_sizes", "performance_by_size.png"): "ieee_fig_sizes.png",
}


def _latest(kind: str) -> Path | None:
    root = repo_root() / "outputs" / kind
    runs = [p for p in root.glob("*") if (p / "run.json").is_file()] if root.is_dir() else []
    if kind == "evaluation":
        runs = [
            p for p in runs if "confirmatory_predictors" in json.loads((p / "run.json").read_text())
        ]
    if kind == "ablation":
        runs = [p for p in runs if (p / "differences.csv").is_file()]
    if kind == "sim_study_growth":
        runs = [p for p in runs if (p / "summary.csv").is_file()]
    return max(runs, key=lambda p: (p / "run.json").stat().st_mtime) if runs else None


def _sidecar(path: Path, runs: dict[str, str]) -> None:
    record = {"source": "scripts/build_paper.py", "runs": runs}
    path.with_name(path.name + PROVENANCE_SUFFIX).write_text(json.dumps(record, indent=2), "utf-8")


def main(tectonic: Path | None = None) -> None:
    kinds = [kind for kind, _, _ in ieee.TABLES]
    runs = {kind: _latest(kind) for kind in kinds}
    configs = {
        "sitar": load_config("growth/sitar.yaml"),
        "agreement": load_config("validation/agreement.yaml"),
        "cohort": load_config("sim/cohort.yaml"),
        "calibration": load_config("sim/calibration.yaml"),
        "validation": load_config("sim/validation.yaml"),
        "runners": load_config("external/runners.yaml"),
        "berkeley": load_config("external/berkeley.yaml"),
        "metrics": load_config("eval/metrics.yaml"),
    }
    values, tables = ieee.build(runs, configs)
    paper = repo_root() / "paper"
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    target = paper / "tables" / "ieee_values.tex"
    target.write_text(values.render(), encoding="utf-8")
    _sidecar(target, values.runs)
    for name, text in tables.items():
        (paper / "tables" / name).write_text(text, encoding="utf-8")
        _sidecar(paper / "tables" / name, values.runs)
    for (kind, source), name in FIGURES.items():
        run = runs.get(kind)
        destination = paper / "figures" / name
        if run is not None and (run / source).is_file():
            shutil.copyfile(run / source, destination)
            _sidecar(destination, {kind: ieee.run_id(run)})
        elif destination.exists():
            destination.unlink()
    for kind, run in runs.items():
        typer.echo(f"  {kind}: {run.name if run else 'pending'}")
    pending = [n for n, (text, _) in values.macros.items() if text == ieee.PENDING]
    typer.echo(f"{len(values.macros)} values, {len(pending)} pending")

    engine = tectonic or (Path(os.environ["TECTONIC"]) if "TECTONIC" in os.environ else None)
    executable = str(engine) if engine else shutil.which("tectonic")
    if executable is None:
        typer.echo("Tectonic not found: tables and figures written, PDF not built")
        return
    build = paper / "ieee" / "build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [executable, "--outdir", str(build), str(paper / "ieee" / "main.tex")], check=True
    )
    pdf = paper / "ieee" / "YouthTwin_IEEE.pdf"
    shutil.copyfile(build / "main.pdf", pdf)
    typer.echo(f"PDF written to {pdf}")


if __name__ == "__main__":
    typer.run(main)
