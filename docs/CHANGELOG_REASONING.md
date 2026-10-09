# Changelog and reasoning

Living document. **Append an entry in the same commit** as every design decision, deviation from
the brief, bug fix, or research finding that shaped the code. Newest milestone at the bottom.
What each file does is in `docs/FILE_MAP.md`; this file records **why**.

Entry format: `ID · date · what changed · why · files`.

---

## M0 — repository scaffold (2026-10-09)

### Decisions

**D-001 · Repository root is the working directory, not a nested `youthtwin/` folder.**
Why: the brief's tree shows `youthtwin/` as the repo root; the project folder
(`Injury_Prevention/`) already plays that role, and nesting would add a path level with no benefit.
The Python package is still `youthtwin` (`src/youthtwin/`).

**D-002 · `uv` (not Poetry) with `hatchling`, Python pinned to 3.11.**
Why: the brief allows either; uv was already installed, fetches Python 3.11 itself (the machine
only had 3.10 and 3.12), and `uv.lock` gives the "pinned versions" the brief requires.
Files: `pyproject.toml`, `uv.lock`.

**D-003 · The Mirwald-girls gate is a separately-run test plus a runtime guard.**
The brief asks for both a test that fails until `mirwald_girls_verified: true` and for
`make test lint` to be green. Both cannot hold if the failing test is in the default suite.
Resolution: the failing test carries a `human_gate` marker, is excluded from `make test`, and
runs under `make gate` and a non-blocking CI job, so it stays visible without blocking
development. Independently, `require_verified()` raises on any real-data run and only warns on
synthetic runs, so the protection does not depend on anyone remembering to run the gate.
Files: `src/youthtwin/growth/verification.py`, `tests/gates/`, `tests/test_verification_gate.py`,
`pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`.

**D-004 · Results guard uses explicit tags rather than "any number is an error".**
Why: `paper/` legitimately contains numbers that are not results (pre-specified thresholds,
literature values, age ranges). A blanket ban would be switched off within a week. Four tags
make intent explicit — `[[RESULT:]]` placeholder, `[[RUN:<id>|value]]` generated,
`[[SPEC:]]` design value, `[[CITED:]]` literature value — and anything statistic-like outside a
tag fails. Plain integers in prose are allowed to keep false positives low; this means the guard
can miss a hand-typed integer count, which is an accepted limitation (documented in the module).
Generated figures/tables must have a provenance sidecar.
Files: `src/youthtwin/utils/results_guard.py`, `scripts/check_no_hardcoded_results.py`.

**D-005 · Module files are created at their milestone, not as empty stubs now.**
Why: the brief requires tests first for anything scientific; an empty `kinematics.py` would
look implemented in a file listing and would pass coverage trivially. M0 creates every package
directory with a docstring naming its purpose and milestone; `docs/FILE_MAP.md` lists planned
files as 🔲. Deviation from "scaffold exactly as §6.1": directories match exactly, module files
do not exist yet.

**D-006 · Dependencies are added at the milestone that first uses them.**
Why: PyMC, PyTorch and MediaPipe are large and slow to resolve (the network here timed out on a
small install), and none is needed until M2/M3/M6. M0 depends only on `omegaconf` and `typer`.

**D-007 · Coefficients live in `configs/growth/equations.yaml`, not in Python.**
Why: rule 10 (no magic numbers in code), and it puts the human-verification flag next to the
values it covers. All values are transcribed from the brief §2.1 and are marked as not yet
checked against the original PDFs.

**D-008 · Config values the brief does not fix are marked `ASSUMPTION` or `TODO`.**
Examples: spline knot count, confidence threshold for gap filling, missed-visit probability,
trial-quality threshold. Why: so that nobody later mistakes a placeholder for a justified
choice. TEM and moment-arm scaling are left `null`/TODO rather than invented.

**D-009 · `Makefile` unsets `PYTHONPATH`.**
Why: this machine has a ROS install whose `PYTHONPATH` injected foreign pytest plugins into the
project environment and crashed test collection. Always run tests through `make test` (or unset
`PYTHONPATH` yourself).

**D-010 · Supabase keys stored in a git-ignored `.env`; Supabase is not used by any code.**
Why: the brief's privacy rule keeps raw video out of every cloud service and makes the research
database local (DuckDB/SQLite, Postgres optional). Whether any participant data may sit in a
cloud database is an ethics decision (`TODO(human)`), so until then Supabase may hold synthetic
data only. A pre-commit hook blocks `.env` files and anything matching the secret-key pattern.
Note: the project URL was not supplied, and the secret key was shared in a chat message —
consider rotating it.
Files: `.env`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`,
`docs/ethics/DRAFT_data_management_plan.md`.

**D-011 · Three video safeguards, not one.**
`.gitignore` (convenience), pre-commit hook (the acceptance criterion), and a CI check plus
`tests/test_repo_hygiene.py` (catches a contributor who never installed the hooks).

### Research notes

**R-001 · Mirwald/Moore coefficients could not be independently confirmed online.**
Searched for the Mirwald (2002) girls equation and the Moore (2015) equations. The Moore paper's
bibliographic details and model structure (boys: age × sitting height; girls: age × height) were
confirmed from repository records. A search snippet from a PMC table showed the intercept and
first two coefficients of the Mirwald **boys** equation matching the brief, but the tables
themselves were behind a bot check and could not be read, so the girls coefficients remain
unverified. The human gate therefore stays closed; this is exactly the case it exists for.

**R-002 · TRIPOD+AI checklist is a paraphrased working copy.**
Item topics were written from memory of the statement's structure, not copied from the paper.
Marked `TODO(human)` to replace with the official wording and confirm numbering before submission.

### Fixes during M0
- Lint: shortened over-long package docstrings; removed an unused `noqa`.
- Test collection crash from system `PYTHONPATH` → D-009.
