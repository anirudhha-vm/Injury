# File map — what each file is responsible for

Living document. **Update it in the same commit** as any change that creates, renames, removes,
or changes the responsibility of a file. The reasons behind changes go in
`docs/CHANGELOG_REASONING.md`.

Status: ✅ exists · 🔲 planned (milestone in brackets). Last updated: 2026-10-09 (M1).

## Root

| File | Responsible for | Status |
|---|---|---|
| `CLAUDE.md` | Working rules, commands, scientific non-negotiables N1–N9 for anyone (human or AI) editing the repo | ✅ |
| `README.md` | Entry point: status, quick start, pointers | ✅ |
| `pyproject.toml` | Package metadata, dependencies, and ruff / mypy / pytest / coverage settings | ✅ |
| `uv.lock` | Exact pinned versions of every dependency | ✅ |
| `Makefile` | The only commands people need: `setup`, `test`, `lint`, `gate`, `todo`, `sim`; later `pipeline`, `paper-figs` | ✅ |
| `.pre-commit-config.yaml` | Blocks video files, `.env` files, and Supabase secret keys from being committed; runs ruff and the paper results guard | ✅ |
| `.github/workflows/ci.yml` | CI: lint + tests (blocking), tracked-video check, human gates (visible, non-blocking) | ✅ |
| `.gitignore` | Keeps raw video, data, secrets, and run outputs out of git | ✅ |
| `.env.example` | Names of the environment variables the project may read (no values) | ✅ |
| `.env` | Local secrets (Supabase keys). Git-ignored; never committed | ✅ local only |

## `configs/` — every tunable value (rule: no magic numbers in code)

| File | Responsible for | Status |
|---|---|---|
| `growth/equations.yaml` | Moore-2015 and Mirwald-2002 coefficients, input plausibility ranges, and the `mirwald_girls_verified` human gate | ✅ |
| `growth/sitar.yaml` | Bayesian growth-twin settings: spline knots, timing prior, sampler, posterior draws, coverage acceptance range | ✅ (used M2) |
| `pose/skeleton.yaml` | BlazePose landmark indices | ✅ (used M3) |
| `pose/estimator.yaml` | Which pose estimator and model variant; fps requirements | ✅ (used M3) |
| `pose/filter.yaml` | Gap filling, Butterworth order/cutoff, velocity units | ✅ (used M3) |
| `pose/tasks.yaml` | Screening task list, trial counts, events per task | ✅ (used M3) |
| `features/task_features.yaml` | Which angles/features are extracted, angle conventions, asymmetry rule, trial-quality threshold | ✅ (used M3) |
| `features/mech.yaml` | Definitions of the mechanistic load proxies; flag that they are not force-plate validated | ✅ (used M5) |
| `models/baselines.yaml` | Predictor sets for the three baseline models | ✅ (used M5) |
| `models/hazard.yaml` | Prediction horizon, outcome groups, risk-set rules, hazard model variants, uncertainty options | ✅ (used M5) |
| `models/tcn_film.yaml`, `models/stgcn_film.yaml` | Encoder backbones and conditioning mode | ✅ (used M6) |
| `eval/splits.yaml` | Grouped / nested / temporal / leave-one-academy-out split definitions | ✅ (used M5) |
| `eval/metrics.yaml` | Primary metric, metric list, bootstrap settings, subgroups | ✅ (used M7) |
| `eval/ablations.yaml` | The nine pre-registered ablation configurations (names only, never results) | ✅ (used M6) |
| `sim/cohort.yaml` | Every parameter of the synthetic cohort: size, seed, growth curves, visits, measurement error, movement, exposure, injury hazards (all flagged as assumptions) | ✅ |
| `io/qc.yaml` | Anthropometry QC limits: duplicate disagreement, sitting ratio, stature decrease, maximum height velocity | ✅ |

## `src/youthtwin/` — the package

| File | Responsible for | Status |
|---|---|---|
| `utils/config.py` | Finding the repo root and loading a YAML config as a dict | ✅ |
| `utils/results_guard.py` | Detecting untagged numbers in `paper/` and generated figures/tables without provenance | ✅ |
| `utils/seeding.py` | Independent, reproducible random streams from one master seed | ✅ |
| `utils/provenance.py` | Run metadata: seed, config hash, git commit, synthetic flag | ✅ |
| `utils/logging.py` | Run logging | 🔲 (when first needed) |
| `growth/verification.py` | Human-verification gate: real-data runs raise, synthetic runs warn, while a coefficient set is unverified | ✅ |
| `growth/maturity_equations.py` | Moore-2015 and Mirwald-2002 maturity offset and predicted age at PHV; rejects implausible inputs; Mirwald girls gated | ✅ |
| `growth/khamis_roche.py` | % predicted adult height; raises `NotImplementedError` until a human supplies the coefficient table | ✅ (placeholder) |
| `growth/sitar_bayes.py` | Hierarchical Bayesian SITAR growth model (C1) | 🔲 (M2) |
| `growth/growth_state.py` | Posterior → per-date growth state with uncertainty and 3-month forecast | 🔲 (M2) |
| `io/schema.py` | Row models for the nine tables: types, units, ranges, privacy rules (month-only DOB, no video paths) | ✅ |
| `io/ingest.py` | Validates tables row by row and loads them into SQLite; all-or-nothing; refuses to mix synthetic and real rows | ✅ |
| `io/qc.py` | Technical error of measurement and anthropometry flags; never edits or drops data | ✅ |
| `io/deid.py` | Pseudonymous IDs, month-only DOB, identity map that can only be written outside the repo | ✅ |
| `sim/growth_sim.py` | Preece–Baines height curves with an exactly known age at PHV; sitting height, leg length, mass; visit schedule; measurement error | ✅ |
| `sim/movement_sim.py` | Session-level screening summary features with a per-athlete trait and a maturity shift | ✅ (minimal) |
| `sim/injury_sim.py` | Weekly exposure and cause-specific injuries from a known hazard process with time-loss exclusion | ✅ |
| `sim/cohort.py` | Assembles schema-shaped tables plus separate truth tables; tags everything synthetic; writes parquet | ✅ |
| `pose/` | Estimators, skeleton definitions, filtering, two-phone sync, event detection, kinematics, task features | 🔲 (M3) |
| `validation/` | Markerless vs mocap agreement statistics (Paper A) | 🔲 (M4) |
| `features/` | Mechanistic load proxies, exposure features, person-period (athlete-week) table | 🔲 (M5) |
| `models/risk/`, `models/baselines.py` | Cause-specific discrete-time hazard models, baselines, uncertainty propagation | 🔲 (M5–M6) |
| `models/encoder/`, `models/pretrain/` | TCN / ST-GCN encoders, conditioning modes incl. FiLM, self-supervised pre-training | 🔲 (M6) |
| `eval/` | Splits, metrics, calibration, decision curves, cluster bootstrap, subgroups | 🔲 (M5, M7) |
| `explain/` | SHAP, integrated gradients, counterfactual sensitivity | 🔲 (M7) |
| `recommend/` | Decision-support text rules with an evidence library | 🔲 (M8) |

Sub-packages marked 🔲 hold only an `__init__.py` whose docstring states their purpose and milestone.

## `scripts/` — command-line entry points

| File | Responsible for | Status |
|---|---|---|
| `check_no_hardcoded_results.py` | Thin CLI over `utils/results_guard.py`; run by `make lint`, pre-commit, CI | ✅ |
| `make_synthetic_cohort.py` | Generates, validates and writes the synthetic cohort (`make sim`) | ✅ |
| `sample_size.py`, `fit_growth.py`, `run_pose.py`, `train_*.py`, `evaluate.py`, `ablation.py`, `make_figures.py` | One CLI per pipeline stage | 🔲 (M1–M7) |

## `tests/`

| File | Guards against | Status |
|---|---|---|
| `test_verification_gate.py` | Unverified coefficients reaching a real-data run | ✅ |
| `gates/test_mirwald_girls_verified.py` | **Fails by design** until a human verifies the Mirwald girls coefficients (`make gate`) | ✅ |
| `test_configs.py` | Broken configs; wrong landmark indices; mixed-up equation families; "expected result" columns; integrity switches flipped | ✅ |
| `test_results_guard.py` | Hand-typed numbers in `paper/`; figures without provenance | ✅ |
| `test_repo_hygiene.py` | Video or `.env` tracked by git; sensitive paths not ignored | ✅ |
| `conftest.py` | Shared fixtures: the default synthetic cohort and configs | ✅ |
| `test_maturity_equations.py` | Wrong equation arithmetic; mixed-up families; implausible or wrong-unit inputs; ungated Mirwald girls | ✅ |
| `test_growth_sim.py` | Non-monotone height; true APHV not at the velocity peak; implausible body proportions | ✅ |
| `test_injury_sim.py` | Injuries during time-loss; hazards not following the stated effect directions | ✅ |
| `test_cohort.py` | Non-reproducible cohorts; unlabelled synthetic data; truth leaking into observable tables | ✅ |
| `test_schema.py` | DDL and models drifting apart; full DOB, non-UUID IDs, video paths, undiagnosed injuries | ✅ |
| `test_ingest_qc_deid.py` | Silent row drops; partial loads; synthetic/real mixing; QC editing data; identity map inside the repo | ✅ |
| v2 regression tests listed in brief §3 (angle conventions, FPPA, filtering, scaler scope, leakage, …) | The specific v2 defects | 🔲 (M1–M7) |

## `docs/`, `paper/`, `data/`, `apps/`

| File | Responsible for | Status |
|---|---|---|
| `docs/PROJECT_BRIEF.md` | The full specification (source of truth for scope) | ✅ |
| `docs/FILE_MAP.md` | This file | ✅ |
| `docs/CHANGELOG_REASONING.md` | Why each change and decision was made | ✅ |
| `docs/plans/Mx.md`, `Mx_done.md` | Plan before, and completion note after, each milestone | ✅ M0, M1 |
| `docs/protocol/01–04_*.md` | DRAFT SOPs: anthropometry, camera setup, screening tasks, injury surveillance | ✅ draft |
| `docs/ethics/DRAFT_*.md` | DRAFT consent, assent, data-management plan — need ethics/legal review | ✅ draft |
| `docs/data_dictionary.md` | Every field, unit and allowed value | ✅ |
| `paper/preregistration.md` | Hypotheses, primary outcome/metric, ablation set — template to register before real data | ✅ template |
| `paper/tripod_ai_checklist.md` | TRIPOD+AI item → where it is satisfied | ✅ working copy |
| `paper/paperA_outline.md`, `paperB_outline.md` | Paper scaffolds | 🔲 (M9) |
| `paper/figures/`, `paper/tables/` | Generated only by `scripts/make_figures.py`, each with a provenance sidecar | ✅ empty |
| `data/README.md`, `data/schema/README.md` | What lives where, privacy rules | ✅ |
| `data/schema/schema.sql` | SQL DDL for the nine tables | ✅ |
| `apps/twin_dashboard/` | Streamlit research dashboard | 🔲 (M8) |
