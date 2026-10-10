# YouthTwin

Research code for an uncertainty-aware, maturity-conditioned digital twin for prospective
prediction of growth-related injury in youth athletes (8–18 y) from smartphone video.

**Status:** M0-M9 built on synthetic data; waiting on human decisions and real data (`make todo`). The pose pipeline has not been run on real video (no test clips yet). No scientific results exist. Research code, not a medical device;
outputs are decision support only.

## Quick start

```bash
make setup      # uv sync + pre-commit hooks (Python 3.11 is fetched by uv)
make test       # tests + coverage
make lint       # ruff, mypy --strict, paper/ results guard
make sim        # generate the synthetic cohort (300 athletes x 12 months)
make evaluate   # out-of-fold evaluation of the tabular risk models (synthetic)
make paper-figs # regenerate every figure and table under paper/ from the latest runs
make dashboard-data && make dashboard   # local browser dashboard (synthetic cohort)
make gate       # human-verification gates (fail by design until resolved)
make todo       # everything waiting on a human decision
```

## Where things are

| Read this | For |
|---|---|
| `docs/PROJECT_BRIEF.md` | the full specification and milestones M0–M9 |
| `CLAUDE.md` | working rules and scientific non-negotiables |
| `docs/FILE_MAP.md` | what every file is responsible for |
| `docs/CHANGELOG_REASONING.md` | what changed and why |
| `docs/plans/` | per-milestone plans and completion notes |

## When real data arrives

Before any of this: resolve the human gates (`make gate` must pass), register
`paper/preregistration.md`, and have ethics approval in place.

```bash
# 1. Put the tables in one folder as parquet files named like the schema tables
#    (athletes, anthropometrics, exposure_weekly, injuries, screening_sessions, task_features),
#    with pseudonymous IDs and synthetic = false. Then validate and run quality control:
uv run python scripts/check_cohort.py data/processed/cohort

# 2. Screening videos (stay on the local disk) -> keypoints and task features:
uv sync --extra pose
uv run python scripts/run_pose.py data/raw_video_LOCAL_ONLY/manifest.csv

# 3. Growth twin posteriors (about 1.5 hours per sex):
uv run python scripts/fit_growth.py data/processed/cohort outputs/growth/cohort

# 4. Out-of-fold evaluation of the tabular risk models, then figures and tables:
uv run python scripts/evaluate.py --cohort-dir data/processed/cohort --posterior-dir outputs/growth/cohort
make paper-figs
```

Not yet wired for real data: the neural models (`scripts/ablation.py`) and the dashboard build
both still generate the synthetic cohort themselves, and `scripts/run_pose.py` has never been
run on a real video. See `docs/plans/M*_done.md` for what each milestone did and did not do.

## Privacy

Raw video of minors never enters this repository or any cloud service. Video stays in
`data/raw_video_LOCAL_ONLY/` (git-ignored, blocked by a pre-commit hook and CI); only keypoints
and derived features are persisted. Secrets go in `.env` (git-ignored), never in code.
