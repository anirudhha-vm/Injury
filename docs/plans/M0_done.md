# M0 done — 2026-10-09

## Acceptance criteria
| Criterion | Result |
|---|---|
| `make test` green | ✅ 56 passed, 1 deselected (the human gate); coverage 100% of the M0 code |
| `make lint` green | ✅ ruff check, ruff format, `mypy --strict` (19 source files), paper results guard |
| CI runs | ✅ First run passed on GitHub after the repo was pushed during M1 (commit `508d555`). |
| No raw video tracked; pre-commit blocks `*.mp4`/`*.mov` | ✅ Hook verified locally by attempting to commit a dummy `.mp4` (rejected). Also blocks `.env` and Supabase secret keys. |
| Mirwald-girls gate fails by design | ✅ `make gate` fails with the `TODO(human)` message |

## Deviations from the brief
See `docs/CHANGELOG_REASONING.md` D-001 (repo root), D-003 (gate is run separately), D-005 (no
empty module stubs), D-006 (dependencies added per milestone).

## Not verified
- Mirwald and Moore coefficients against the original papers (R-001).
- TRIPOD+AI item wording (R-002).

## Next: M1
Maturity equations with reference-value tests, synthetic growth + injury generator, data schema,
ingestion and QC. Plan goes in `docs/plans/M1.md` first.
