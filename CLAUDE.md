# YouthTwin — CLAUDE.md

Research codebase for a Q1 journal paper: an uncertainty-aware, maturity-conditioned digital twin
for prospective prediction of growth-related injury in youth athletes (8–18 y) from smartphone
video. Reproducible research code, **not** a product. Full spec: `docs/PROJECT_BRIEF.md`.

Current milestone status: see `docs/plans/` (`Mx.md` = plan, `Mx_done.md` = completed).

## Commands

| Command | What it does |
|---|---|
| `make setup` | `uv sync` + install pre-commit hooks |
| `make test` | pytest with coverage (excludes `human_gate` tests) |
| `make lint` | ruff check + ruff format --check + `mypy --strict` on `src/` + `paper/` results guard |
| `make gate` | human-verification gates — **fails by design** until each `TODO(human)` is resolved |
| `make todo` | list every `TODO(human)` |
| `make sim` | generate the synthetic cohort into `data/synthetic/` |
| `make test-slow` | tests that run the MCMC sampler |
| `make sim-study-growth` | growth-twin simulation study on synthetic data (about 40 minutes) |
| `make evaluate` | out-of-fold evaluation of the tabular risk models on synthetic data |
| `make ablation` | the nine-configuration ablation grid (slow) |
| `make paper-figs` | regenerate every figure and table under `paper/` from the latest runs |
| `make pipeline` | `sim` → `evaluate` → `paper-figs` |

Run `make test lint` before every commit.

## Two living documents — update them in the same commit as the change

- `docs/FILE_MAP.md` — what every file is responsible for (add/update a row when a file is
  created, renamed, removed, or changes responsibility).
- `docs/CHANGELOG_REASONING.md` — what changed and **why** (append an entry for every design
  decision, deviation from the brief, bug fix, or research finding that shaped the code).

## Working rules

1. **Plan before code**: `docs/plans/Mx.md` (goal, files, tests, acceptance) before each milestone;
   `docs/plans/Mx_done.md` after.
2. **Tests first for anything scientific**: equations, angle conventions, splits, leakage checks
   get unit tests with reference values before use.
3. **Never fabricate results.** Every number in `paper/` comes from a script and is traceable to
   config + seed + commit. Tags: `[[RESULT: …]]` placeholder, `[[RUN:<run_id>|value]]` generated,
   `[[SPEC: …]]` pre-specified design value, `[[CITED: …]]` literature value. Enforced by
   `scripts/check_no_hardcoded_results.py`.
4. **Synthetic data is labelled synthetic**: `synthetic=True` in metadata, `SYNTHETIC` watermark
   on figures; synthetic results appear in the paper only under "Simulation study".
5. **Ask, don't guess, on human-only decisions** (ethics, consent, academies, sports, lab access,
   mocap system, journal): leave `TODO(human):` and continue.
6. **Privacy by design**: raw video of minors never enters the repo or any cloud bucket. Persist
   keypoints + derived features only. Athlete IDs are pseudonymous UUIDs; the ID↔identity map
   lives outside the repo. Secrets live in `.env` (git-ignored) only.
7. **Small commits**, one logical change each.
8. **Python 3.11**, type hints everywhere, `ruff` + `mypy --strict` on `src/`, ≥85% coverage on
   `src/youthtwin/growth`, `pose/kinematics`, `eval`.
9. **Config-driven**: hyperparameters, feature lists, tasks, splits live in `configs/`. No magic
   numbers in code. Values not taken from the brief or literature are marked `ASSUMPTION`.
10. Dependencies are added at the milestone that first needs them; `uv.lock` pins versions.

## Non-negotiables (scientific integrity)

| # | Rule |
|---|---|
| N1 | Outcome = real, physio-diagnosed injury from prospective surveillance. Never train on labels generated from our own thresholds, rules, or the maturity score. |
| N2 | Any feature used to define a label may not be a model input (`tests/test_no_label_leakage.py`, M5). |
| N3 | All splits grouped by athlete; overlapping windows never cross folds; plus a temporal hold-out (last ~25% of follow-up). |
| N4 | Scalers/normalisers fit on training folds only. Never per-clip, never on the full dataset. |
| N5 | Maturity estimates carry uncertainty, and it is propagated to risk. |
| N6 | Injury outcomes are type-specific and modelled separately: (a) growth-related, (b) non-contact soft-tissue, (c) all non-contact. |
| N7 | Report per TRIPOD+AI; keep `paper/tripod_ai_checklist.md` updated as code is written. |
| N8 | Pre-register hypotheses, primary outcome/metric, and ablations in `paper/preregistration.md` before touching real outcome data. |
| N9 | Decision support, not diagnosis: "elevated risk — consider review", never diagnoses or orders. |

## v2 defects that must not come back

No rule assigns high risk from maturity alone; no hand-set risk curves; no "expected" results;
knee flexion = 180 − included angle; valgus is FPPA from the frontal camera; BlazePose lower-limb
indices 23–32; velocities filtered and in deg/s; asymmetry only at task events; SHAP on fitted
models only. Full table with the guarding test for each: `docs/PROJECT_BRIEF.md` §3.

## Layout

`src/youthtwin/` package (src layout) · `configs/` YAML · `scripts/` typer CLIs · `tests/`
(`tests/gates/` = human gates) · `data/` git-ignored except README/schema · `paper/` outputs
generated only by scripts · `docs/` brief, plans, protocol, ethics drafts. Details in
`docs/FILE_MAP.md`.
