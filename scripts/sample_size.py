"""Sample-size planning with the Riley et al. criteria (binary outcome, per athlete-season).

Prints, for each scenario in ``configs/eval/sample_size.yaml``, the athletes needed for a model
with a given number of candidate parameters, and the number of parameters a given cohort size
can support. Writes both tables to ``outputs/sample_size/``.
"""

from __future__ import annotations

import pandas as pd
import typer

from youthtwin.eval.sample_size import required_sample_size, supported_parameters
from youthtwin.utils.config import load_config, repo_root


def main() -> None:
    config = load_config("eval/sample_size.yaml")
    options = {"shrinkage": config["shrinkage"], "optimism": config["optimism"]}
    needed, supported = [], []
    for prevalence in config["prevalence_scenarios"]:
        for parameters in config["parameter_scenarios"]:
            result = required_sample_size(
                parameters=parameters, prevalence=prevalence, margin=config["margin"], **options
            )
            needed.append(
                {
                    "prevalence": prevalence,
                    "parameters": parameters,
                    "athletes_required": result.n_required,
                    "events_required": round(result.events_required),
                    "events_per_parameter": round(result.events_per_parameter, 1),
                    "assumed_r2_cox_snell": round(result.r2_cox_snell, 4),
                }
            )
        for n in config["cohort_size_scenarios"]:
            supported.append(
                {
                    "prevalence": prevalence,
                    "athletes": n,
                    "expected_events": round(n * prevalence),
                    "parameters_supported": supported_parameters(
                        n=n, prevalence=prevalence, **options
                    ),
                }
            )
    target = repo_root() / "outputs" / "sample_size"
    target.mkdir(parents=True, exist_ok=True)
    needed_table, supported_table = pd.DataFrame(needed), pd.DataFrame(supported)
    needed_table.to_csv(target / "athletes_required.csv", index=False)
    supported_table.to_csv(target / "parameters_supported.csv", index=False)
    typer.echo("Athletes required (one outcome per athlete-season):")
    typer.echo(needed_table.to_string(index=False))
    typer.echo("\nCandidate parameters supported by a cohort:")
    typer.echo(supported_table.to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
