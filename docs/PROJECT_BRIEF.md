# Claude Code Build Prompt — YouthTwin
## An uncertainty-aware, maturity-conditioned digital twin for prospective prediction of growth-related injury in youth athletes (ages 8–18) from smartphone video

> Paste this whole file into Claude Code as the first message (or save it as `docs/PROJECT_BRIEF.md` and tell Claude Code: "Read docs/PROJECT_BRIEF.md and start Milestone M0").
> It replaces the v2.0 spec ("YouthAthlete_AI_DigitalTwin_v2.docx"). Do NOT reuse v2 code verbatim — it contains bugs listed in §3.

---

## 0. Your role and working rules (read first)

You are the lead research engineer on a team writing a **Q1 journal paper** (target: IEEE JBHI / IEEE TNSRE / Computers in Biology and Medicine / J Sci Med Sport / Scand J Med Sci Sports — the team will confirm current quartile in SJR/JCR). Your job is to build a **reproducible research codebase**, not a production product.

Working rules — follow all of them:

1. **Plan before code.** At the start of every milestone, write a short plan in `docs/plans/Mx.md` (goal, files, tests, acceptance criteria), then implement.
2. **Create and maintain `CLAUDE.md`** at repo root: project summary, commands (`make test`, `make lint`, `make sim`), conventions, and the non-negotiables in §1.
3. **Tests first for anything scientific.** Equations, angle conventions, data splits, and leakage checks must have unit tests with reference values before they are used.
4. **Never fabricate results.** No hard-coded "expected AUC", no invented numbers in docs or paper drafts. Every number in `paper/` must be produced by a script in this repo and traceable to a config + seed + commit hash. Placeholders are written as `[[RESULT: description]]`.
5. **Synthetic data is labelled synthetic.** We do not have the real cohort yet. You will build a synthetic data generator (§6.8) to develop and test the pipeline. Every artifact derived from synthetic data must carry `synthetic=True` in its metadata and a `SYNTHETIC` watermark on figures. Synthetic results are never reported as findings (except in the explicitly-labelled simulation study, §6.8).
6. **Ask, don't guess, on human-only decisions**: ethics, consent, which academies, which sports, lab access, gold-standard mocap system, journal choice. Leave a `TODO(human):` and continue with the rest.
7. **Privacy by design.** Raw video of minors never enters the repo or any cloud bucket. Pipelines take video paths from a local, git-ignored directory and persist **keypoints + derived features only**. Athlete IDs are pseudonymous UUIDs; the ID↔identity map lives outside the repo.
8. **Small, reviewable commits.** One logical change per commit. Run `make test lint` before each commit.
9. **Python 3.11, type hints everywhere, `ruff` + `mypy --strict` on `src/`, `pytest` with ≥85% coverage on `src/growth`, `src/pose/kinematics`, `src/eval`.**
10. **Config-driven.** All hyperparameters, feature lists, tasks, and split definitions live in `configs/` (Hydra or OmegaConf YAML). No magic numbers in code.

---

## 1. Non-negotiables (scientific integrity — put these in CLAUDE.md)

| # | Rule | Why |
|---|---|---|
| N1 | **Outcome = real, physio-diagnosed injury from prospective surveillance.** Never train the risk model on labels generated from our own thresholds, rules, or the maturity score. | v2 labels were circular (rules → labels → model relearns rules). Reviewers reject this. |
| N2 | **Any feature used to define a label may not be a model input** (enforced by `tests/test_no_label_leakage.py`). | Same as above. |
| N3 | **All splits are grouped by athlete.** No athlete appears in both train and test. Overlapping windows from one athlete never cross folds. Plus a **temporal hold-out** (last ~25% of follow-up time). | v2 had window-level leakage. |
| N4 | **Scalers/normalisers are fit on training folds only.** Never per-clip, never on the full dataset. | v2 fit RobustScaler per clip, erasing between-athlete differences and leaking. |
| N5 | **Maturity estimates carry uncertainty** and that uncertainty is propagated to risk. | Maturity-offset equations have known systematic bias (see §2). |
| N6 | **Injury outcomes are type-specific**: (a) growth-related (apophysitis — Osgood-Schlatter, Sinding-Larsen-Johansson, Sever's; apophyseal avulsion; physeal injury; bone stress), (b) non-contact soft-tissue, (c) all non-contact. Model separately (cause-specific). | Evidence shows growth-related injuries cluster pre/around PHV while soft-tissue injuries increase post-PHV — a single "risk peaks at PHV" curve is wrong. |
| N7 | **Report per TRIPOD+AI (Collins et al., BMJ 2024).** Keep `paper/tripod_ai_checklist.md` updated as code is written. | Required by top journals for prediction models. |
| N8 | **Pre-register** hypotheses, primary outcome, primary metric, and the ablation set in `paper/preregistration.md` before touching real outcome data. | Avoid garden-of-forking-paths; strengthens review. |
| N9 | **Decision support, not diagnosis.** Outputs are framed as "elevated risk — consider review", never as diagnoses or medical orders. | Clinical/ethical defensibility. |

---

## 2. Verified scientific background you must encode correctly

### 2.1 Maturity-offset equations (implement in `src/growth/maturity_equations.py`)

All inputs: age in decimal years; stature, sitting height, leg length in cm; mass in kg. Leg length = stature − sitting height. Maturity offset (MO) = years from PHV; predicted age at PHV (APHV) = age − MO.

**Moore et al. (2015), "Enhancing a somatic maturity prediction model", MSSE** — PRIMARY (simpler, less over-fitted):
- Boys: `MO = -8.128741 + 0.0070346 * (age * sitting_height)`
- Girls: `MO = -7.709133 + 0.0042232 * (age * stature)`
- (A stature-only boys variant also appears in the literature: `MO = -7.999994 + 0.0036124 * (age * stature)` — implement as `moore2015_boys_stature` for sensitivity analysis.)

**Mirwald et al. (2002), MSSE 34(4):689–694** — SECONDARY (comparison):
- Boys: `MO = -9.236 + 0.0002708*(leg_length*sitting_height) - 0.001663*(age*leg_length) + 0.007216*(age*sitting_height) + 0.02292*(mass/stature*100)`
- Girls: `MO = -9.376 + 0.0001882*(leg_length*sitting_height) + 0.0022*(age*leg_length) + 0.005841*(age*sitting_height) - 0.002658*(age*mass) + 0.07693*(mass/stature*100)`
- `TODO(human): verify Mirwald girls coefficients against the original paper PDF before any real-data run.` Add a test that fails with a clear message until `configs/growth/equations.yaml: mirwald_girls_verified: true`.

**Bug fixed from v2:** v2 labelled the Moore girls equation as "Mirwald" and mixed it with the Mirwald boys equation. Keep the two families separate and named.

**Known limitations (encode as docstrings and use to justify the Bayesian model):**
- Predictions regress toward the population mean: APHV is underestimated at younger ages and overestimated at older ages; error is larger in early and late maturers (Malina & Kozieł 2014) and grows the further the child is from PHV (Moore 2015; Mills et al. 2017).
- Derivation samples were predominantly white European ancestry → generalisation to Indian youth is unknown. **This is a stated research gap we address** (the growth twin learns cohort-specific timing).
- **Khamis–Roche % predicted adult height (PAH%)** is a complementary maturity indicator (needs mid-parent height, with self-report correction). Implement as optional input `src/growth/khamis_roche.py`; `TODO(human): obtain coefficient table from Khamis & Roche (1994) Pediatrics 94:504–507` — do not invent coefficients. If missing, the module raises `NotImplementedError`.

### 2.2 Injury epidemiology facts that shape the design
- Prospective academy study (n = 501, Hall et al., Phys Ther Sport 2022): overall, soft-tissue, muscle, hamstring injuries **higher post-PHV**; growth-related injuries **lower post-PHV than pre-PHV**.
- High-level youth football (Sports Med Int Open 2024): body-mass change and greater maturity associated with sudden-onset injury; weekly training hours associated with gradual-onset, bone and physis injuries.
- ⇒ Inputs must include **growth velocity** (height, leg length, mass), **maturity state**, **training exposure**, not just a static PHV proximity score.
- Bahr (BJSM 2016), "Why screening tests to predict injury do not work—and probably never will": single cross-sectional screens have poor predictive value. ⇒ We use **repeated screening**, **time-varying covariates**, **dynamic prediction**, and **decision-curve analysis**, and we discuss this paper explicitly.

---

## 3. v2 defects you must NOT reproduce (add regression tests)

| v2 defect | Correct behaviour | Test |
|---|---|---|
| 3-point "knee angle" ≈180° at extension compared to flexion thresholds (30–60°) | `knee_flexion = 180 - included_angle`; document convention | `test_angle_conventions.py` |
| Knee "valgus" claimed but never computed (3D included angle mixes planes) | Compute frontal-plane projection angle (FPPA) from frontal camera + sagittal flexion from side camera; or joint angles from a 3D model with defined segment coordinate systems | `test_fppa_synthetic_skeleton.py` |
| Landmark table used 11,13,15 (shoulder/elbow/wrist) for knee; ankle used 13/14 | MediaPipe BlazePose: hips 23/24, knees 25/26, ankles 27/28, heels 29/30, foot index 31/32 — centralise in `src/pose/skeletons.py` | `test_landmark_indices.py` |
| Ankle dorsiflexion listed but not computed | Implement (knee–ankle–foot_index) with convention docs | covered |
| RobustScaler fit per clip | Fit on train fold only; serialise with model | `test_scaler_fit_scope.py` |
| Velocities in deg/frame, unfiltered | Low-pass Butterworth (4th-order, zero-lag, cutoff configurable 6–12 Hz), velocities in deg/s using true timestamps | `test_filtering.py` |
| Frame-wise L/R asymmetry during non-bilateral tasks | Asymmetry computed at task **events** (initial contact, peak knee flexion, take-off) in bilateral tasks, or between left- vs right-limb trials in unilateral tasks | `test_event_asymmetry.py` |
| `bio_risk`, `phv_proximity`, `age_norm`, `offset` all derived from one number (redundant) | Growth state = posterior summaries from §6.2 (independent quantities: APHV, MO, height velocity, leg-length velocity, mass velocity, leg/trunk growth ratio, + SDs) | `test_growth_state_collinearity.py` (warn if VIF > 10) |
| Rule "bio_risk > 0.8 ⇒ CRITICAL regardless of movement" | Removed. No rule may assign high risk from maturity alone | n/a |
| Hand-set Gaussian (σ = 0.8) risk curve and age_factor step | Removed. Relationships are learned (splines/GAM terms) and reported | n/a |
| "Expected AUC" ablation column | Removed. Results only from runs | `scripts/check_no_hardcoded_results.py` in CI greps `paper/` for numbers not tagged with a run ID |
| `CalibratedClassifierCV(...).estimator` passed to SHAP (unfitted) | SHAP on the fitted booster(s) from `calibrated_classifiers_`, or calibrate separately and explain uncalibrated model; document | `test_shap_runs.py` |
| `use_label_encoder` (deprecated) and API signature mismatches | Remove; typed interfaces + integration test | `test_api_contract.py` |
| CMU MoCap used directly for transfer | Only via explicit retargeting to the chosen skeleton topology; optional | n/a |

---

## 4. Research contributions (what the paper claims)

Primary paper (Paper B) claims, in order of importance:

- **C1 — Personalised Bayesian growth twin.** A hierarchical SITAR-style growth model (shared mean curve; per-athlete size, timing, and intensity random effects) fitted to sparse repeated anthropometrics (every 6–8 weeks), with the Moore-2015 APHV estimate as an informative prior on the timing effect. Output per athlete per day: posterior of APHV, maturity offset, height/leg/mass velocity, and leg-to-trunk growth ratio — **with credible intervals** — updated each new measurement. It is a twin because it has a persistent per-athlete state, updates with new data, and forecasts (next-3-month growth).
- **C2 — Maturity-conditioned spatiotemporal movement encoder.** A skeleton-sequence encoder (temporal CNN baseline; ST-GCN variant) whose intermediate features are modulated by the growth state via **FiLM** (feature-wise linear modulation: γ(g)⊙h + β(g)). Compared against: no growth input, late fusion, input concatenation (the v2 design), and FiLM — a clean ablation of *how* growth should condition movement.
- **C3 — Dynamic, injury-type-specific risk prediction with propagated uncertainty.** Discrete-time cause-specific hazard model on athlete-week data predicting injury in the next 4 weeks, using time-varying growth state (C1), latest screening embedding/features (C2), training exposure, and injury history. Growth-state uncertainty is propagated by Monte-Carlo sampling from the C1 posterior. Evaluated with discrimination, calibration, and **decision-curve net benefit**.
- **C4 (supporting) — Mechanistic load proxies.** Growth-scaled estimates of apophyseal traction load: patellar-tendon/tibial-tuberosity load proxy (knee extensor moment proxy ÷ patellar-tendon moment arm scaled by segment length) and Achilles/calcaneal traction proxy, from kinematics + body mass + segment lengths. Interpretable features tied to Osgood-Schlatter and Sever's mechanisms. (Clearly labelled as proxies unless validated against force plates.)

Companion paper (Paper A, earlier): **validity of smartphone markerless kinematics in youth across maturity stages** vs a marker-based gold standard (C5), plus cross-sectional maturity-dependence of injury-relevant kinematics.

Hypotheses (copy to `paper/preregistration.md`):
- H1: C1 posterior APHV has lower error vs observed APHV than Moore/Mirwald point estimates (simulation + the subset of athletes who pass PHV during follow-up).
- H2: For growth-related injury, the full model (C1+C2+C3) improves time-dependent AUC and net benefit over (a) age-only, (b) maturity-offset + growth-velocity logistic model, (c) movement-only.
- H3: FiLM conditioning outperforms input concatenation and late fusion (primary ablation).
- H4: Propagating growth uncertainty improves calibration (calibration slope closer to 1, lower ECE/ICI).
- H5 (Paper A): Markerless sagittal knee/hip angles at task events agree with mocap within clinically acceptable limits (pre-specify, e.g., RMSE < 5°), with agreement not worse in pre-PHV children.

---

## 5. Study protocol the code must support (data model is designed around this)

- **Cohort:** 8–18 y, 2–4 academies (e.g., cricket, football, athletics), target ≈300 athletes; final N via Riley et al. sample-size criteria (implement `scripts/sample_size.py` wrapping `pmsampsize` logic or a documented re-implementation) once incidence is estimated. `TODO(human)`: academies, sports.
- **Follow-up:** ≥ 1 season (9–12 months).
- **Every 6–8 weeks:** stature, sitting height, mass (duplicate measures; technical error of measurement logged); screening session with standardized tasks recorded by **two smartphones (frontal + sagittal, ≥60 fps, ideally 120)**:
  - Drop vertical jump (DVJ) ×3, single-leg squat ×3/leg, tuck jump 10 s, single-leg hop for distance ×3/leg. Task list is config-driven.
- **Once:** mid-parental height (self-report + correction), sex, sport, playing position, dominant limb, injury history.
- **Weekly:** OSTRC-O2 questionnaire + exposure (training/match hours, sessions); physio-confirmed diagnoses coded with **OSIICS**; classify into §1 N6 outcome groups. Time-loss and substantial-problem definitions per consensus.
- **Gold-standard subset (Paper A):** 30–40 athletes stratified by maturity, simultaneous marker-based mocap (`TODO(human)`: Vicon/Qualisys lab access) + the same phones.
- **Ethics:** institutional ethics committee approval per ICMR (2017) National Ethical Guidelines for research involving children; verifiable parental consent + child assent consistent with India's DPDP Act 2023. Generate **draft** templates only (`docs/ethics/`), clearly marked for human/legal review.

---

## 6. System architecture and module specs

```
L0  data/ & ethics      schema, ingestion, de-identification, QC
L1  src/growth          maturity equations → Bayesian growth twin (C1)
L2  src/pose            video → keypoints → filtering → events → kinematics → task features
L2v src/validation      markerless vs mocap agreement (Paper A)
L3  src/models/encoder  maturity-conditioned skeleton encoder (C2)
L4  src/models/risk     discrete-time cause-specific hazard + uncertainty (C3)
L4m src/features/mech   mechanistic load proxies (C4)
L5  src/eval, src/explain   grouped/temporal CV, calibration, DCA, SHAP/IG
L6  apps/twin_dashboard     minimal Streamlit research dashboard (no MLOps)
SIM src/sim             synthetic cohort generator with known ground truth
```

### 6.1 Repository layout

```
youthtwin/
├── CLAUDE.md
├── README.md
├── Makefile                      # setup, test, lint, sim, pipeline, paper-figs
├── pyproject.toml                # uv or poetry; pinned versions
├── configs/
│   ├── growth/{equations.yaml, sitar.yaml}
│   ├── pose/{estimator.yaml, filter.yaml, tasks.yaml, skeleton.yaml}
│   ├── features/{task_features.yaml, mech.yaml}
│   ├── models/{tcn_film.yaml, stgcn_film.yaml, baselines.yaml, hazard.yaml}
│   ├── eval/{splits.yaml, metrics.yaml, ablations.yaml}
│   └── sim/cohort.yaml
├── data/                         # git-ignored except README + schema
│   ├── README.md
│   ├── schema/                   # SQL DDL + pydantic models + data dictionary
│   ├── raw_video_LOCAL_ONLY/     # never committed, never uploaded
│   ├── keypoints/                # parquet
│   ├── interim/ processed/
│   └── synthetic/
├── src/youthtwin/
│   ├── io/                       # schema.py, ingest.py, deid.py, qc.py
│   ├── growth/                   # maturity_equations.py, khamis_roche.py, sitar_bayes.py, growth_state.py
│   ├── pose/                     # estimators/{mediapipe.py, rtmpose.py}, skeletons.py, filtering.py,
│   │                             # sync.py, events.py, kinematics.py, task_features.py
│   ├── validation/               # mocap_io.py, agreement.py (RMSE, ICC(2,1), Bland–Altman, SPM optional)
│   ├── features/                 # mech_load.py, exposure.py, person_period.py
│   ├── models/
│   │   ├── encoder/              # tcn.py, stgcn.py, film.py, conditioning.py (none|concat|late|film)
│   │   ├── pretrain/             # ssl.py (masked-frame / contrastive on unlabeled trials)
│   │   ├── risk/                 # hazard_logit.py, hazard_xgb.py, hazard_neural.py, uncertainty.py
│   │   └── baselines.py
│   ├── eval/                     # splits.py, metrics.py, calibration.py, dca.py, bootstrap.py, subgroups.py
│   ├── explain/                  # shap_utils.py, integrated_gradients.py, counterfactual.py
│   ├── recommend/                # rules.py, evidence_library.yaml (decision-support text only)
│   ├── sim/                      # growth_sim.py, movement_sim.py, injury_sim.py, cohort.py
│   └── utils/                    # seeding.py, logging.py, provenance.py
├── scripts/                      # CLI entry points (typer): run_pose.py, fit_growth.py, train_*.py,
│                                 # evaluate.py, ablation.py, sample_size.py, make_figures.py,
│                                 # check_no_hardcoded_results.py
├── apps/twin_dashboard/          # Streamlit
├── tests/                        # unit + integration + leakage + regression tests
├── docs/
│   ├── PROJECT_BRIEF.md          # this file
│   ├── plans/                    # per-milestone plans
│   ├── protocol/                 # data collection SOPs, task instructions, camera setup
│   ├── ethics/                   # DRAFT consent/assent templates (human review required)
│   └── data_dictionary.md
└── paper/
    ├── preregistration.md
    ├── tripod_ai_checklist.md
    ├── paperA_outline.md
    ├── paperB_outline.md
    └── figures/ tables/          # generated only by scripts/make_figures.py
```

### 6.2 L1 — Growth twin (`src/youthtwin/growth/`)

- `maturity_equations.py`: pure functions per §2.1, vectorised (numpy), unit-checked inputs (raise on implausible values: stature 90–210 cm, sitting/stature ratio 0.45–0.60, age 7–20). Return dataclass with `mo`, `aphv`, `equation_name`.
- `sitar_bayes.py`: PyMC model. Height_ij = α_i + h((age_ij − β_i)·exp(γ_i)) where h is a shared natural cubic spline (B-spline basis, knots configurable), α size, β timing, γ intensity random effects (hierarchical, sex-specific population params).
  - Prior on β_i centred on Moore-2015 APHV for athlete i with SD learned/configured (default 1.0 y).
  - Fit sitting height (or leg length) and mass with parallel models or a multivariate extension (start parallel).
  - Inference: NUTS (default) with ADVI fallback for speed; save `arviz.InferenceData`.
  - Supports **incremental update**: refit with new measurements, warm-started.
- `growth_state.py`: given posterior + a date → `GrowthState` (posterior mean & SD, and S posterior draws) for: APHV, MO, height velocity (cm/y), leg-length velocity, mass velocity (kg/y), leg/trunk growth ratio, % of PHV-velocity reached. Also 3-month forecast.
- Acceptance: on synthetic cohorts (§6.8) with known APHV, posterior 90% intervals have empirical coverage 85–95%; APHV MAE lower than Moore point estimate; tests for monotone height, positive velocities near PHV.

### 6.3 L2 — Pose & kinematics (`src/youthtwin/pose/`)

- Estimator interface `PoseEstimator.process(video_path) -> KeypointSequence` (T × J × {x,y,z?,conf}, timestamps in ms from container, fps).
  - `mediapipe.py`: use the **MediaPipe Tasks `PoseLandmarker`** API (the legacy `mp.solutions.pose` is deprecated), heavy model for research.
  - `rtmpose.py`: RTMPose (via `rtmlib` or MMPose) as comparator — Paper A compares estimators.
  - Optional later: two-camera triangulation (`sync.py` + calibration via checkerboard/ChArUco) and/or OpenCap pipeline if the team adopts it.
- `skeletons.py`: single source of truth for landmark indices & segment definitions.
- `filtering.py`: confidence-gated gap filling (linear ≤ 5 frames, else NaN), 4th-order zero-lag Butterworth; cutoff from config; residual-analysis helper to justify cutoff.
- `sync.py`: two-phone time sync (audio clap or LED flash cross-correlation).
- `events.py`: per-task event detection — DVJ: box drop, initial contact (IC), peak knee flexion, take-off, second landing; single-leg squat: start, deepest point, return; hop: take-off, landing. Detectors use heel/ankle vertical position & velocity; return frame indices + confidence. Manual-override CSV supported.
- `kinematics.py`: sagittal knee flexion, hip flexion, ankle dorsiflexion (side camera); knee **FPPA**, pelvic drop, trunk lateral lean (frontal camera); knee-to-ankle separation ratio; all with explicit sign conventions and units (degrees, deg/s).
- `task_features.py`: per trial: values at events, ranges of motion, peak angular velocities, landing time-normalised curves (101 points) for SPM; per session: mean of trials + between-limb asymmetry (event-based, LSI or symmetry angle). Output tidy parquet with `athlete_id, session_id, task, trial, limb, feature, value, quality`.
- QC: per-trial quality score (mean landmark confidence, % gaps, event-detection confidence); trials below threshold are excluded and **logged**, never silently dropped.

### 6.4 L2v — Validation (Paper A) (`src/youthtwin/validation/`)

- Load C3D/TRC mocap; align time (cross-correlation on knee angle) and conventions.
- Metrics per variable × task × event: bias, RMSE, Bland–Altman (95% LoA, proportional bias), ICC(2,1)/(3,1) with CIs; stratified by maturity group (pre/circa/post-PHV via MO bands) and sex; optional SPM1D curve comparison.
- Figures: Bland–Altman grid, curve overlays, RMSE by maturity stratum.

### 6.5 L3 — Maturity-conditioned encoder (`src/youthtwin/models/encoder/`)

- Input per trial: time-normalised or event-aligned keypoint/angle sequences (T≈101 per phase or fixed window at ≥60 fps), channels = selected joint angles + angular velocities + normalised 2D coordinates (root-centred, scaled by leg length). Per-channel normalisation stats from training fold only.
- Backbones: `tcn.py` (dilated residual 1D CNN, small: ≤200k params by default) and `stgcn.py` (on the joint graph).
- `conditioning.py` modes (config switch, the primary ablation): `none | concat_input | late_fusion | film`. FiLM layers after each residual block: γ, β = MLP(g), g = growth-state vector (posterior means + SDs, standardised).
- Training objectives:
  1. **Self-supervised pre-training** on all trials (no injury labels): masked-segment reconstruction and/or contrastive (two augmentations of same trial: time-warp 0.9–1.1, Gaussian noise, small rotation, *no L/R mirroring when the limb label matters*).
  2. **Auxiliary supervised targets** available for every trial (not injury): predict task-event kinematic summaries and, if the team rates them, expert movement-quality scores (e.g., LESS-style). These are *auxiliary*, never the injury label.
  3. Encoder output = trial embedding → session embedding (attention-pooled over trials/tasks) → passed to L4.
- Keep models small; data are small. Early stopping on grouped validation folds. Seeds logged.

### 6.6 L4 — Risk model (`src/youthtwin/models/risk/`, `src/youthtwin/features/person_period.py`)

- Build **person-period (athlete-week) table**: at each week w: growth state (from L1 at that date), latest screening features/embedding (last observation carried forward with "weeks since screen"), exposure (weekly hours, sessions, EWMA load with configurable decay; ACWR **not** used as primary), prior injury indicators, sex, sport. Outcome: injury of type k in weeks (w, w+4]; athlete-weeks while injured/time-lost are excluded from the risk set.
- Models (all cause-specific):
  - `baselines.py`: age-only logistic; MO + growth-velocity logistic (clinical baseline); screening-feature-only logistic.
  - `hazard_logit.py`: penalised (elastic-net) discrete-time logistic hazard with natural splines for continuous growth terms — **likely strong; it is a fair primary competitor**.
  - `hazard_xgb.py`: gradient-boosted hazard (monotone constraints optional) with SHAP.
  - `hazard_neural.py`: small MLP head on [growth state, session embedding, exposure], optionally FiLM-conditioned.
- `uncertainty.py`: (a) propagate L1 posterior by evaluating each model over S growth-state draws → predictive distribution of risk; (b) deep-ensemble (5 seeds) for neural; (c) optional split-conformal intervals for risk.
- Class imbalance: use proper likelihood + class weights if needed; **no SMOTE on person-period data**; report prevalence.

### 6.7 L5 — Evaluation & explanation (`src/youthtwin/eval/`, `src/youthtwin/explain/`)

- `splits.py`: GroupKFold by athlete (stratified by event status), nested for tuning; temporal hold-out; leave-one-academy-out if ≥3 academies. Unit test asserts zero athlete overlap.
- `metrics.py`: time-dependent AUC (cumulative/dynamic at 4 weeks), Brier score, calibration intercept & slope, ICI/ECE, sensitivity/specificity at pre-specified thresholds, PR-AUC (rare events).
- `dca.py`: decision-curve analysis (net benefit vs threshold probability; treat-all/treat-none references).
- `bootstrap.py`: athlete-level cluster bootstrap CIs (≥1000 resamples) for all metrics and for model differences (primary comparison method; DeLong only for single-time-point AUC sanity checks).
- `subgroups.py`: by sex, maturity stratum, sport, age band — fairness/robustness table.
- `explain/`: SHAP for tabular hazard models; integrated gradients / attention maps for encoder; counterfactual "what-if" (e.g., exposure −20%) shown as **model sensitivity, explicitly not causal effect**.
- `scripts/ablation.py`: runs the full ablation grid from `configs/eval/ablations.yaml`:
  1. age only · 2. MO + growth velocity · 3. screening features only · 4. movement encoder only (no growth) · 5. late fusion · 6. concat (v2 design) · 7. **FiLM (proposed)** · 8. FiLM + mechanistic proxies (C4) · 9. #8 + uncertainty propagation (C3). Output a single tidy results table with run IDs.

### 6.8 SIM — Synthetic cohort generator (`src/youthtwin/sim/`) — build early (M1)

Purpose: develop/test everything without real data and run a **labelled simulation study** for C1/C3 method validation.

- `growth_sim.py`: simulate individual height curves via Preece–Baines model 1 with sex-specific parameter distributions (literature-plausible APHV ≈ 12 ± 1 y girls, 14 ± 1 y boys — configurable, document sources or mark as assumptions), sitting-height/leg-length partition, mass; measurement error (TEM); sparse visit schedule every 6–8 weeks with missed visits.
- `movement_sim.py`: synthetic kinematic summary features (and optionally synthetic skeleton sequences) whose distributions shift with maturity and include individual traits.
- `injury_sim.py`: weekly hazards for each injury type generated from a **known** data-generating process (e.g., growth-related hazard ↑ with leg-length velocity × landing knee-extensor load proxy; soft-tissue hazard ↑ post-PHV and with exposure spikes). Known truth enables checking the pipeline recovers effects and that metrics/CI code is correct.
- Every output flagged `synthetic=True`. Simulation-study results may appear in the paper **only** in a section titled "Simulation study".

### 6.9 L6 — Research dashboard (`apps/twin_dashboard/`)

Streamlit, local only. Per athlete: growth curve with posterior band and APHV interval; velocity timeline; screening history with skeleton replay (from keypoints, not video); 4-week risk per injury type with uncertainty band; top SHAP contributors; decision-support text from `recommend/rules.py` (each item cites `evidence_library.yaml`, framed as "consider", never prescriptive). No auth/K8s/Grafana/MLOps — out of scope for the paper.

### 6.10 Data schema (`data/schema/`)

Tables (SQLite/DuckDB for research; Postgres optional): `athletes`(id, sex, dob_month, sport, academy_id, dominant_limb, mid_parent_height), `anthropometrics`(athlete_id, date, stature×2, sitting_height×2, mass×2, measurer_id), `screening_sessions`(id, athlete_id, date, camera_setup), `trials`(id, session_id, task, limb, trial_no, keypoints_uri, quality), `task_features`(long format), `exposure_weekly`(athlete_id, week_start, hours_training, hours_match, sessions), `ostrc_weekly`(athlete_id, week_start, q1–q4, severity), `injuries`(id, athlete_id, onset_date, osiics_code, body_area, tissue, mechanism, time_loss_days, outcome_group, diagnosed_by), `growth_posteriors`(athlete_id, fit_date, uri). Pydantic models mirror the DDL; `data_dictionary.md` documents every field and unit. Store DOB at month precision only.

---

## 7. Milestones (do them in order; each ends with green tests + a short `docs/plans/Mx_done.md`)

| ID | Milestone | Key acceptance criteria |
|---|---|---|
| **M0** | Repo scaffold, `CLAUDE.md`, tooling (uv/poetry, ruff, mypy, pytest, pre-commit, Makefile, GitHub Actions CI), configs skeleton, `paper/preregistration.md` + `tripod_ai_checklist.md` templates, `docs/protocol/` SOP drafts, `docs/ethics/` DRAFT templates | `make test lint` green; CI runs; no raw-video path tracked by git (pre-commit hook blocks `*.mp4|*.mov`) |
| **M1** | `growth/maturity_equations.py` + tests; `sim/` growth + injury generator; data schema + ingestion + QC | Reference-value tests pass; Mirwald-girls verification gate in place; synthetic cohort of 300 athletes × 12 months generated reproducibly from a seed |
| **M2** | Bayesian growth twin (`sitar_bayes.py`, `growth_state.py`) + simulation study notebook-free script | Coverage 85–95% on synthetic; APHV MAE < Moore point-estimate MAE on synthetic; incremental update works; figures watermarked SYNTHETIC |
| **M3** | Pose pipeline: estimators, skeletons, filtering, sync, events, kinematics, task features, QC | Angle-convention, landmark, filtering, event-asymmetry tests pass; runs end-to-end on a short local test clip (`TODO(human)`: provide 2–3 consented or self-recorded adult test clips) |
| **M4** | Validation module (Paper A) | Agreement metrics verified on synthetic paired data with known bias/noise; Bland–Altman & ICC match reference implementations (e.g., `pingouin`) |
| **M5** | Person-period builder, exposure features, mechanistic load proxies (C4), baselines + penalised hazard + XGB hazard | Leakage tests pass; on synthetic data the hazard model recovers the simulated effect directions; split overlap = 0 |
| **M6** | Encoder (TCN + ST-GCN), conditioning modes, SSL pre-training, session pooling, neural hazard head, uncertainty propagation | Ablation grid runs end-to-end on synthetic; each conditioning mode trains; ensemble + posterior-draw uncertainty produces calibrated intervals on synthetic |
| **M7** | Evaluation suite (metrics, calibration, DCA, cluster bootstrap, subgroups), explainability, `scripts/make_figures.py`, results provenance | Every figure/table regenerated from one command; `check_no_hardcoded_results.py` passes |
| **M8** | Streamlit research dashboard, recommendation rules with evidence library | Dashboard runs on synthetic cohort; all text is decision-support framed |
| **M9** | Paper scaffolds: `paperA_outline.md`, `paperB_outline.md` (IMRaD, TRIPOD+AI mapped), methods text auto-drafted **from configs** (not from memory), results sections as `[[RESULT: …]]` placeholders | Methods match configs; zero fabricated numbers |

After M9, stop and summarise: what's done, what needs human input (`grep -r "TODO(human)"`), and the exact commands to run once real data arrives.

---

## 8. Paper outlines to scaffold

**Paper A — "Validity of smartphone markerless motion capture for injury-relevant landing and squat kinematics in youth athletes across maturity stages"**
Intro (screening needs, markerless promise, youth/maturity and population gaps) → Methods (participants, maturity estimation, tasks, cameras, estimators compared, mocap, processing, agreement stats) → Results (agreement by variable/task/event/maturity/sex) → Discussion (which variables are usable; limits) → Conclusion.

**Paper B — "An uncertainty-aware, maturity-conditioned digital twin for dynamic prediction of growth-related injury in youth athletes"**
Intro (burden of growth-related injury; maturity timing; Bahr 2016 critique; gap: individual, uncertainty-aware, dynamic models combining growth and movement) → Methods (TRIPOD+AI order: source of data, participants, outcomes N6, predictors, sample size, missing data, growth twin C1, movement pipeline & encoder C2, mechanistic proxies C4, hazard models C3, uncertainty propagation, evaluation, ablations, simulation study) → Results (cohort flow, incidence, growth-twin accuracy, discrimination/calibration/DCA, ablation, subgroups, explanations) → Discussion (clinical meaning, bio-banding & load management implications, Indian cohort generalisability, limitations: sample size, proxies not force-validated, single-region) → Conclusion.

---

## 9. Key references to cite (team must verify each against the original before submission)

- Mirwald RL et al. (2002) An assessment of maturity from anthropometric measurements. *Med Sci Sports Exerc* 34(4):689–694.
- Moore SA et al. (2015) Enhancing a somatic maturity prediction model. *Med Sci Sports Exerc* 47(8):1755–1764.
- Malina RM, Kozieł SM (2014) Validation of maturity offset in a longitudinal sample of Polish boys/girls. *J Sports Sci*.
- Khamis HJ, Roche AF (1994) Predicting adult stature without using skeletal age: the Khamis-Roche method. *Pediatrics* 94:504–507.
- Cole TJ, Donaldson MDC, Ben-Shlomo Y (2010) SITAR — a useful instrument for growth curve analysis. *Int J Epidemiol* 39:1558–1566.
- Bahr R (2016) Why screening tests to predict injury do not work—and probably never will…: a critical review. *Br J Sports Med* 50:776–780.
- Collins GS et al. (2024) TRIPOD+AI statement. *BMJ* 385:e078378.
- Hall ECR et al. (2022) Injury risk is greater in physically mature versus biologically younger male soccer players from academies in different countries. *Phys Ther Sport* 55:111–118.
- (2024) Exploring growth, maturity, and age as injury risk factors in high-level youth football. *Sports Med Int Open* 8.
- Perez E et al. (2018) FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
- Yan S et al. (2018) Spatial temporal graph convolutional networks (ST-GCN). *AAAI*.
- Uhlrich SD et al. (2023) OpenCap: human movement dynamics from smartphone videos. *PLoS Comput Biol*.
- Riley RD et al. (2020) Calculating the sample size required for developing a clinical prediction model. *BMJ* 368:m441.
- Vickers AJ, Elkin EB (2006) Decision curve analysis. *Med Decis Making* 26:565–574.
- Clarsen B et al. (2020) OSTRC-H2 / OSTRC questionnaires; Orchard JW et al. (2020) OSIICS v13. *Br J Sports Med*.

---

## 10. First action

Start **M0** now: write `docs/plans/M0.md`, scaffold the repository exactly as in §6.1, create `CLAUDE.md` with §0–§1 condensed, set up tooling and CI, and add the failing-by-design verification gate for Mirwald-girls coefficients. Then report back with the tree, the commands, and the list of `TODO(human)` items.
