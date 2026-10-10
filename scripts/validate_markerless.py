"""Agreement between markerless and mocap values (Paper A).

Input: a parquet or CSV table of paired values with columns ``athlete_id, variable, task, event,
markerless, mocap`` and, for stratified tables, ``maturity_group`` and ``sex``; a boolean
``synthetic`` column is required. Output: agreement tables (overall, by maturity group, by sex)
and figures in ``outputs/validation/<run_id>/``, each with a provenance record; figures from
synthetic pairs are watermarked.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import typer

from youthtwin.utils.config import load_config, repo_root
from youthtwin.utils.plotting import save_figure
from youthtwin.utils.provenance import config_hash, run_metadata
from youthtwin.validation import plots
from youthtwin.validation.agreement import agreement_table


def main(pairs_path: Path, out_dir: Path | None = None) -> None:
    config = load_config("validation/agreement.yaml")
    pairs = (
        pd.read_parquet(pairs_path) if pairs_path.suffix == ".parquet" else pd.read_csv(pairs_path)
    )
    if "synthetic" not in pairs.columns:
        raise typer.BadParameter("pairs must have a boolean 'synthetic' column")
    synthetic = bool(pairs["synthetic"].all())
    if not synthetic and pairs["synthetic"].any():
        raise typer.BadParameter("pairs mix synthetic and real rows")
    run_id = f"validation-{config_hash({'config': config, 'pairs': pairs_path.name})}"
    metadata = {"run_id": run_id, **run_metadata(seed=0, config=config, synthetic=synthetic)}
    target = out_dir or repo_root() / "outputs" / "validation" / run_id
    target.mkdir(parents=True, exist_ok=True)

    tables = {"overall": agreement_table(pairs, config)}
    for column in ("maturity_group", "sex"):
        if column in pairs.columns:
            tables[f"by_{column}"] = agreement_table(pairs, config, by=[column])
    for name, table in tables.items():
        table.to_csv(target / f"agreement_{name}.csv", index=False)
    (target / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_figure(plots.bland_altman_grid(pairs, config), target / "bland_altman.png", metadata)
    if "by_maturity_group" in tables:
        save_figure(
            plots.rmse_by_stratum(tables["by_maturity_group"], config),
            target / "rmse_by_maturity.png",
            metadata,
        )
    label = "SYNTHETIC " if synthetic else ""
    typer.echo(f"{label}agreement tables and figures written to {target}")
    columns = ["variable", "task", "event", "n", "bias", "loa_low", "loa_high", "rmse", "icc21"]
    typer.echo(tables["overall"][columns].round(2).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
