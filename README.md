# YouthTwin

Research code for an uncertainty-aware, maturity-conditioned digital twin for prospective
prediction of growth-related injury in youth athletes (8–18 y) from smartphone video.

**Status:** M0-M4 built on synthetic data. The pose pipeline has not been run on real video (no test clips yet). No scientific results exist. Research code, not a medical device;
outputs are decision support only.

## Quick start

```bash
make setup      # uv sync + pre-commit hooks (Python 3.11 is fetched by uv)
make test       # tests + coverage
make lint       # ruff, mypy --strict, paper/ results guard
make sim        # generate the synthetic cohort (300 athletes x 12 months)
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

## Privacy

Raw video of minors never enters this repository or any cloud service. Video stays in
`data/raw_video_LOCAL_ONLY/` (git-ignored, blocked by a pre-commit hook and CI); only keypoints
and derived features are persisted. Secrets go in `.env` (git-ignored), never in code.
