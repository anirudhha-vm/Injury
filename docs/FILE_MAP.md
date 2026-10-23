# File map — what each file is responsible for

Living document. **Update it in the same commit** as any change that creates, renames, removes,
or changes the responsibility of a file. The reasons behind changes go in
`docs/CHANGELOG_REASONING.md`.

Status: ✅ exists · 🔲 planned (milestone in brackets). Last updated: 2026-10-10 (strategy decisions after M9).

## Root

| File | Responsible for | Status |
|---|---|---|
| `CLAUDE.md` | Working rules, commands, scientific non-negotiables N1–N9 for anyone (human or AI) editing the repo | ✅ |
| `README.md` | Entry point: status, quick start, pointers | ✅ |
| `pyproject.toml` | Package metadata, dependencies, and ruff / mypy / pytest / coverage settings | ✅ |
| `uv.lock` | Exact pinned versions of every dependency | ✅ |
| `Makefile` | The only commands people need: `setup`, `test`, `test-slow`, `lint`, `gate`, `todo`, `fetch-data`, `real-growth`, `real-risk`, `sim`, `sim-study-growth`, `sim-study-sizes`, `evaluate`, `ablation`, `dashboard-data`, `dashboard`, `paper-figs`, `pipeline` | ✅ |
| `.pre-commit-config.yaml` | Blocks video files, `.env` files, and Supabase secret keys from being committed; runs ruff and the paper results guard | ✅ |
| `.github/workflows/ci.yml` | CI: lint + tests (blocking), tracked-video check, human gates (visible, non-blocking) | ✅ |
| `.gitignore` | Keeps raw video, data, secrets, and run outputs out of git | ✅ |
| `.env.example` | Names of the environment variables the project may read (no values) | ✅ |
| `.env` | Local secrets (Supabase keys). Git-ignored; never committed | ✅ local only |

## `configs/` — every tunable value (rule: no magic numbers in code)

| File | Responsible for | Status |
|---|---|---|
| `growth/equations.yaml` | Moore-2015 and Mirwald-2002 coefficients, input plausibility ranges, and the `mirwald_girls_verified` human gate | ✅ |
| `growth/sitar.yaml` | Growth-twin settings: knots, timing prior, prior scales, integration grids, sampler, number of draws kept, coverage acceptance range | ✅ |
| `pose/skeleton.yaml` | BlazePose landmark indices (must equal `pose/skeletons.py`; tested) | ✅ |
| `pose/events.yaml` | Event-detection thresholds as fractions of leg length and degrees | ✅ (assumptions) |
| `pose/estimator.yaml` | Which pose estimator, model file path, detection confidences, fps requirements | ✅ |
| `pose/filter.yaml` | Gap filling, Butterworth order/cutoff, velocity units | ✅ |
| `pose/tasks.yaml` | Screening task list, trial counts, events per task | ✅ |
| `features/task_features.yaml` | Which angles/features are extracted, angle conventions, asymmetry index, trial-quality threshold | ✅ |
| `features/mech.yaml` | Proxy definitions, the athlete-week columns they are computed from, and the not-validated flag | ✅ |
| `features/person_period.yaml` | Exposure half-life, observed-velocity window, forbidden predictor names | ✅ |
| `models/baselines.yaml` | Predictor sets and settings for the three baseline models | ✅ |
| `models/hazard.yaml` | Prediction horizon, outcome groups, risk-set rules, hyperparameters of the logistic and boosted hazards, uncertainty options | ✅ |
| `models/tcn_film.yaml`, `models/stgcn_film.yaml` | Encoder sizes and parameter budget | ✅ |
| `models/neural.yaml` | Neural hazard training, growth-vector and tabular columns, uncertainty draws, pre-training | ✅ |
| `eval/splits.yaml` | Grouped / nested / temporal / leave-one-academy-out split definitions | ✅ (read by scripts from M7) |
| `eval/sample_size.yaml` | Sample-size planning scenarios (event proportions, parameter counts, cohort sizes) | ✅ |
| `eval/metrics.yaml` | Metric list, thresholds (none pre-specified yet), decision-curve range, calibration settings, bootstrap size, subgroup definitions, what-if settings | ✅ |
| `dashboard.yaml` | Prompt thresholds, uncertainty draws, growth-curve grid, labels shown | ✅ (thresholds are placeholders) |
| `eval/ablations.yaml` | The nine pre-registered ablation configurations and the paired comparisons reported between them (names and switches only, never results) | ✅ |
| `eval/cohort_size_study.yaml` | Cohort sizes, replicates and models of the cohort-size simulation study (never results) | ✅ |
| `external/berkeley.yaml` | Source (URL, checksum), reference-APHV rules and season-window design for the real Berkeley growth study | ✅ |
| `external/runners.yaml` | Source files and checksums of the real runners' injury dataset, and its evaluation settings | ✅ |
| `sim/cohort.yaml` | Every parameter of the synthetic cohort: size, seed, growth curves, visits, measurement error, movement, exposure, injury hazards (all flagged as assumptions except the cited mean age at PHV) | ✅ |
| `sim/calibration.yaml` | Target incidence for calibrating the synthetic injury process (cited) | ✅ |
| `sim/validation.yaml` | Synthetic markerless-vs-mocap study: groups, body scale, error model, calibration target (cited), replicates | ✅ |
| `sim/test_signal.yaml` | Strong-signal overrides of the cohort config, used **only** by pipeline tests | ✅ |
| `io/qc.yaml` | Anthropometry QC limits: duplicate disagreement, sitting ratio, stature decrease, maximum height velocity | ✅ |
| `validation/agreement.yaml` | Limits-of-agreement level, confidence level, provisional RMSE acceptance limit, maturity band, alignment window | ✅ |

## `src/youthtwin/` — the package

| File | Responsible for | Status |
|---|---|---|
| `utils/config.py` | Finding the repo root and loading a YAML config as a dict | ✅ |
| `utils/results_guard.py` | Detecting untagged numbers in `paper/` and generated figures/tables without provenance | ✅ |
| `utils/seeding.py` | Independent, reproducible random streams from one master seed | ✅ |
| `utils/provenance.py` | Run metadata: seed, config hash, git commit, synthetic flag | ✅ |
| `utils/plotting.py` | SYNTHETIC watermark and figure saving with a provenance sidecar | ✅ |
| `utils/logging.py` | Run logging | 🔲 (when first needed) |
| `growth/verification.py` | Human-verification gate: real-data runs raise, synthetic runs warn, while a coefficient set is unverified | ✅ |
| `growth/maturity_equations.py` | Moore-2015 and Mirwald-2002 maturity offset and predicted age at PHV; rejects implausible inputs; Mirwald girls gated | ✅ |
| `growth/splines.py` | Natural cubic spline basis and derivative, usable on NumPy arrays and PyTensor variables; orthogonalising transform for sampling | ✅ |
| `growth/evaluation.py` | Scores posterior APHV against a known truth: coverage, errors, bias, by group | ✅ |
| `growth/khamis_roche.py` | % predicted adult height; raises `NotImplementedError` until a human supplies the coefficient table | ✅ (placeholder) |
| `growth/sitar_bayes.py` | Hierarchical growth model: data preparation, marginal likelihood (athlete effects integrated out), NUTS over the shared curve, exact athlete-effect draws, incremental update, diagnostics | ✅ (interval coverage in range within 3 years of PHV, not beyond — see `docs/plans/M2_done.md`) |
| `growth/growth_state.py` | `SitarPosterior` (saved draws, plain NumPy) and `GrowthState`: age at PHV, maturity offset, height velocity, % of peak velocity, 90-day forecast, each with mean, SD and draws | ✅ (stature only) |
| `io/schema.py` | Row models for the nine tables: types, units, ranges, privacy rules (month-only DOB, no video paths) | ✅ |
| `io/ingest.py` | Validates tables row by row; reads a cohort folder of parquet tables (the real-data entry point); loads into SQLite; all-or-nothing; refuses to mix synthetic and real rows | ✅ |
| `io/qc.py` | Technical error of measurement and anthropometry flags; never edits or drops data | ✅ |
| `io/deid.py` | Pseudonymous IDs, month-only DOB, identity map that can only be written outside the repo | ✅ |
| `sim/growth_sim.py` | Preece–Baines height curves with an exactly known age at PHV; sitting height, leg length, mass; visit schedule; measurement error | ✅ |
| `sim/movement_sim.py` | Session-level screening features with a per-athlete trait and maturity shift; synthetic drop-jump trials per session driven by the load proxy | ✅ |
| `sim/injury_sim.py` | Weekly exposure (persistent within athlete, differing between athletes) and cause-specific injuries from a known hazard process with time-loss exclusion | ✅ |
| `sim/validation_sim.py` | Synthetic markerless-vs-mocap study: two noisy observations of the same simulated drop jumps through the real pose pipeline; errors scale with body size | ✅ |
| `sim/skeleton_sim.py` | Stylised synthetic skeleton sequences (drop jump, single-leg squat) with exactly known angles and event frames, for testing the pose pipeline | ✅ |
| `sim/truth.py` | True 4-week event probability per athlete-week of a synthetic cohort, for scoring only | ✅ |
| `sim/cohort.py` | Assembles schema-shaped tables plus separate truth tables; tags everything synthetic; writes parquet | ✅ |
| `pose/skeletons.py` | Landmark indices and canonical joint names for BlazePose-33 and Halpe-26; the only place raw indices appear | ✅ (Halpe-26 indices unverified) |
| `pose/keypoints.py` | `KeypointSequence`: pixels (y down), per-frame timestamps, confidence; parquet + JSON storage with no reference to the video | ✅ |
| `pose/filtering.py` | Confidence-gated gap filling, zero-lag 4th-order Butterworth, resampling of uneven frames, derivative per second, residual analysis | ✅ |
| `pose/kinematics.py` | Joint-angle definitions and sign conventions: knee/hip flexion, ankle dorsiflexion, knee FPPA, pelvic drop, trunk lean, knee-ankle separation | ✅ |
| `pose/events.py` | Event detection for drop vertical jump, single-leg squat and hop, with confidence, reported-not-guessed missing events, and manual overrides | ✅ (thresholds are assumptions) |
| `pose/sync.py` | Time offset between the two phones by cross-correlating a shared clap or flash signal | ✅ (signal extraction from video not implemented) |
| `pose/task_features.py` | Per-trial features at events, ranges of motion, peak angular velocities, 101-point curves; session means; event-based asymmetry; quality score and exclusion log | ✅ |
| `pose/estimators/base.py` | `PoseEstimator` interface, video frame reader, optional-dependency loader | ✅ (frame reader unverified) |
| `pose/estimators/mediapipe.py` | MediaPipe Tasks `PoseLandmarker` wrapper | ⚠️ written, never run |
| `pose/estimators/rtmpose.py` | RTMPose (body with feet) wrapper via rtmlib | ⚠️ written, never run |
| `validation/agreement.py` | Markerless-vs-mocap agreement: RMSE, MAE, Bland-Altman with confidence intervals and proportional bias, ICC(2,1), ICC(3,1), stratified agreement table | ✅ |
| `validation/mocap_io.py` | TRC reader/writer, optional C3D reader, time alignment of mocap and markerless curves | ✅ (C3D path never run; markers-to-angles not implemented) |
| `validation/plots.py` | Bland-Altman grid, RMSE by maturity stratum, curve overlay | ✅ |
| `features/person_period.py` | Athlete-week table: predictors known at the end of each week, 4-week cause-specific outcomes, risk set, censoring; leakage guard | ✅ |
| `features/exposure.py` | Weekly hours, sessions, exponentially weighted load, change in hours | ✅ |
| `features/mech_load.py` | Mechanistic load proxies (knee-extensor moment, patellar-tendon, Achilles), defined up to a constant; not force-plate validated | ✅ (not yet used in a fitted model) |
| `models/tabular.py` | `HazardModel` (pipeline + input list + leakage check), natural-spline transformer, preprocessing fitted on training rows only, partial effect | ✅ |
| `models/confirmatory.py` | The confirmatory risk model: leading entries of a pre-specified predictor list, as many as the achieved events support | ✅ |
| `models/baselines.py` | Age-only, maturity + growth velocity, and screening-only logistic baselines | ✅ |
| `models/risk/hazard_logit.py` | Elastic-net discrete-time logistic hazard with spline terms | ✅ |
| `models/risk/hazard_xgb.py` | Gradient-boosted hazard with optional monotone constraints | ✅ |
| `models/risk/hazard_neural.py` | Neural hazard (encoder + head), training with athlete-grouped early stopping, deep ensemble, train-only standardisation | ✅ |
| `models/risk/uncertainty.py` | Risk distribution from growth-posterior draws and ensemble members; variance decomposition | ✅ (intervals reflect growth-timing uncertainty only and do not cover the true risk — see `docs/plans/M6_done.md`) |
| `models/encoder/sequences.py` | Trial → (channels, time) array of angles and angular velocities; session stacking; channel scaler fitted on training athletes | ✅ |
| `models/encoder/film.py` | FiLM layer (identity at initialisation) | ✅ |
| `models/encoder/tcn.py` | Dilated residual 1D CNN encoder with optional FiLM | ✅ |
| `models/encoder/stgcn.py` | Graph encoder on the hip-knee-ankle joint graph with optional FiLM | ✅ |
| `models/encoder/conditioning.py` | The conditioning switch (`none`, `concat_input`, `late_fusion`, `film`) and attention pooling of trials into a session embedding | ✅ |
| `models/pretrain/ssl.py` | Masked-segment pre-training on unlabelled trials; augmentations without left/right mirroring | ✅ |
| `models/dataset.py` | Builds encoder sequences for a synthetic cohort through the real pose pipeline; athlete-week arrays; growth-posterior draws | ✅ |
| `eval/splits.py` | Athlete-grouped stratified folds, nested folds, temporal hold-out, leave-one-academy-out; overlap check | ✅ |
| `eval/sample_size.py` | Riley et al. sample-size criteria for a binary-outcome prediction model, and the number of parameters a cohort supports | ✅ |
| `eval/collinearity.py` | Variance inflation factors and a warning for redundant model inputs | ✅ |
| `eval/metrics.py` | AUC (time-dependent at the horizon), PR-AUC, Brier score, sensitivity/specificity at pre-specified thresholds | ✅ |
| `eval/ablation.py` | Runs the nine pre-registered configurations with athlete-grouped cross-validation; tidy results with run IDs plus the out-of-fold predictions; optional scoring against known true risk; `ablation_report` gives athlete-level bootstrap intervals and the configured paired differences | ✅ |
| `eval/cohort_size.py` | Cohort-size simulation study: out-of-fold metrics of the tabular models on replicate synthetic cohorts of each size, scored also against the known true risk; summary across replicates | ✅ |
| `external/reference_aphv.py` | Reference age at PHV from a dense real growth record: Preece-Baines model 1 fit, smoothing-spline cross-check, exclusion rules | ✅ |
| `external/berkeley.py` | Loads the Berkeley Child Guidance Study (real), draws one-season windows per child, computes Moore/Mirwald from the same windows | ✅ |
| `external/runners.py` | Loads the real runners' injury dataset (bounded load ratios, week suffixes), model builders, the three split schemes (runner-grouped, temporal, row-level leak demonstration) and held-out prediction | ✅ |
| `eval/calibration.py` | Calibration intercept and slope, ICI, ECE, binned calibration curve | ✅ |
| `eval/dca.py` | Decision-curve analysis: net benefit with treat-all and treat-none references | ✅ |
| `eval/bootstrap.py` | Athlete-level cluster bootstrap for metrics and for paired differences between models | ✅ |
| `eval/subgroups.py` | Age bands and maturity strata; metrics per subgroup with a minimum-events rule | ✅ |
| `eval/predictions.py` | Out-of-fold predictions table: the single input to every metric, table and figure | ✅ |
| `eval/report.py` | Builds the performance, difference, decision-curve, calibration and subgroup tables from predictions | ✅ |
| `explain/shap_utils.py` | SHAP on the fitted tabular models; additivity; values summed back to original inputs | ✅ |
| `explain/integrated_gradients.py` | Integrated-gradients attributions for networks; pooling over a session's trials | ✅ |
| `explain/counterfactual.py` | What-if changes to inputs, labelled as model sensitivity, not causal effect | ✅ |
| `recommend/rules.py` | Decision-support prompts from model outputs; wording and citation checks; no elevated-risk prompt from maturity alone | ✅ |
| `recommend/evidence_library.yaml` | Evidence statements and citations behind the prompts | ✅ (not verified against the papers) |
| `paper/methods.py` | Renders the methods draft from the configuration files, every value tagged | ✅ |
| `paper/results.py` | Renders the results draft from run tables; every number tagged with its run ID; real data and simulation study kept apart; missing runs become placeholders | ✅ |
| `paper/ieee.py` | LaTeX macros and tables for the IEEE manuscript, generated from runs; pending runs print a red marker | ✅ |
| `dashboard/bundle.py` | Builds and loads the dashboard data bundle: growth curves, out-of-fold risk with uncertainty, SHAP contributions, prompts, replay keypoints | ✅ |
| `dashboard/figures.py` | The dashboard's plots, watermarked when synthetic | ✅ |

Sub-packages marked 🔲 hold only an `__init__.py` whose docstring states their purpose and milestone.

## `scripts/` — command-line entry points

| File | Responsible for | Status |
|---|---|---|
| `check_no_hardcoded_results.py` | Thin CLI over `utils/results_guard.py`; run by `make lint`, pre-commit, CI | ✅ |
| `make_synthetic_cohort.py` | Generates, validates and writes the synthetic cohort (`make sim`) | ✅ |
| `fit_growth.py` | Fits the growth twin to a cohort directory, or updates existing posteriors with new measurements | ✅ |
| `sim_study_growth.py` | Growth-twin simulation study on the synthetic cohort: coverage and error vs truth and vs Moore; writes scores, summary, posteriors, watermarked figure to `outputs/` (`make sim-study-growth`) | ✅ |
| `run_pose.py` | Local videos + manifest → keypoint files, task-feature table and exclusion log | ⚠️ written, never run on real video |
| `validate_markerless.py` | Paired markerless/mocap table → agreement tables and figures in `outputs/validation/` | ✅ (run on made-up pairs only) |
| `ablation.py` | Runs the ablation grid on the synthetic cohort; writes `results.csv`, `predictions.parquet`, `performance.csv` and `differences.csv` to `outputs/ablation/<run_id>/`; `--report-from` recomputes the interval tables from saved predictions | ✅ |
| `sim_study_sizes.py` | Runs the cohort-size study; writes `replicates.csv`, `summary.csv` and `performance_by_size.png` to `outputs/sim_study_sizes/<run_id>/` | ✅ |
| `fetch_public_data.py` | Downloads the public datasets into git-ignored `data/external/` and verifies their SHA-256 | ✅ |
| `real_growth_berkeley.py` | Growth twin vs Moore/Mirwald on the real Berkeley data; writes `reference.csv`, `per_athlete.csv`, `summary.csv`, `aphv_real.png` to `outputs/real_growth_berkeley/<run_id>/` | ✅ |
| `real_risk_runners.py` | Risk models on real injury outcomes (runners) under three split schemes; athlete-bootstrap intervals, paired differences, calibration, leakage inflation; to `outputs/real_risk_runners/<run_id>/` | ✅ |
| `calibrate_simulation.py` | Finds the growth-related hazard intercept that gives the published incidence, exactly from the true hazards; writes `outputs/calibration/<run_id>/` | ✅ |
| `sim_study_validation.py` | Synthetic validity study: calibrates the markerless error, simulates the planned 36-participant study, repeats it for precision; to `outputs/sim_study_validation/<run_id>/` | ✅ |
| `evaluate.py` | Out-of-fold evaluation of baselines, the confirmatory model and tabular hazards: predictions, performance with intervals, differences, decision curve, calibration, subgroups, SHAP importance, what-if (`make evaluate`) | ✅ |
| `make_figures.py` | Turns the latest runs (evaluation, ablation, growth study, real Berkeley growth, real runners' risk, synthetic validity study) into every figure and table under `paper/`, each with a provenance sidecar (`make paper-figs`) | ✅ |
| `draft_results.py` | Writes `paper/results_draft.md` from the latest run of each kind (`make paper-results`) | ✅ |
| `build_paper.py` | Builds the IEEE manuscript: values, tables and figures from the latest runs, then the PDF with Tectonic (`make paper-pdf`) | ✅ |
| `sample_size.py` | Prints and saves sample-size planning scenarios from `configs/eval/sample_size.yaml` | ✅ |
| `check_cohort.py` | First command on real data: validates a cohort folder against the schema and writes anthropometry QC flags and TEM per measurer | ✅ |
| `draft_methods.py` | Regenerates `paper/methods_draft.md` (`make methods`) | ✅ |
| `build_dashboard.py` | Builds the dashboard bundle from the synthetic cohort and saved growth posteriors (`make dashboard-data`) | ✅ |
| `sample_size.py`, `run_pose.py`, `train_*.py`, `evaluate.py`, `ablation.py`, `make_figures.py` | One CLI per pipeline stage | 🔲 (M1–M7) |

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
| `test_injury_sim.py` | Injuries during time-loss; hazards not following the stated effect directions; training load that does not persist or that shifts the average | ✅ |
| `test_cohort.py` | Non-reproducible cohorts; unlabelled synthetic data; truth leaking into observable tables | ✅ |
| `test_schema.py` | DDL and models drifting apart; full DOB, non-UUID IDs, video paths, undiagnosed injuries | ✅ |
| `test_ingest_qc_deid.py` | Silent row drops; partial loads; synthetic/real mixing; QC editing data; identity map inside the repo | ✅ |
| `test_splines.py` | A spline that is not smooth, not linear in the tails, or differs between NumPy and PyTensor | ✅ |
| `test_growth_state.py` | APHV not at the velocity peak; uncertainty not carried into the state; scoring errors | ✅ |
| `test_sitar_bayes.py` | Wrong marginal likelihood; mis-calibrated or single-peaked athlete draws; update not using new data; unwatermarked synthetic figures. One `slow` end-to-end fit | ✅ |
| `test_landmark_indices.py` | Upper-limb landmarks used for the leg (v2); code and config disagreeing | ✅ |
| `test_angle_conventions.py` | Included angle reported as flexion (v2); wrong signs for hip, ankle, pelvis, trunk | ✅ |
| `test_fppa_synthetic_skeleton.py` | "Valgus" that changes with knee flexion (v2); wrong sign on one limb | ✅ |
| `test_filtering.py` | Velocities per frame or unfiltered (v2); phase lag; wrong cut-off; bridged gaps; sync offset | ✅ |
| `test_events.py` | Events off by more than a few frames; invented events when the clip is incomplete | ✅ |
| `test_event_asymmetry.py` | Frame-wise left/right asymmetry, or asymmetry inside a single-leg trial (v2) | ✅ |
| `test_pose_pipeline.py` | Features that miss the known truth; frame-rate-dependent velocity; silently dropped trials; keypoint files that point at video | ✅ |
| `test_agreement.py` | Wrong Bland-Altman or ICC arithmetic (checked by hand and against pingouin); dropped thin strata; TRC parsing; time-alignment errors | ✅ |
| `test_no_label_leakage.py` | Labels or truth as model inputs; predictors that depend on the future; labels not from diagnoses; athletes on both sides of a split (v2) | ✅ |
| `test_scaler_fit_scope.py` | Preprocessing fitted on test rows or per clip (v2) | ✅ |
| `test_growth_state_collinearity.py` | Growth inputs that are copies of one number (v2) | ✅ |
| `test_person_period.py` | Wrong outcome window, risk set, carry-forward or censoring; exposure using future weeks; load-proxy arithmetic | ✅ |
| `test_hazard_models.py` | Models that do not fit or predict; unfitted estimator exposed for explanation (v2); simulated effect directions not recovered | ✅ |
| `test_encoder.py` | Wrong shapes or oversize models; conditioning that leaks into `none` or does nothing in the others; padding affecting pooling; normalisation using held-out athletes; mirrored augmentations | ✅ |
| `test_neural_hazard.py` | Training that is not reproducible or reads held-out rows; growth uncertainty not reaching risk; wrong draw assembly; ablation grid not running end to end | ✅ |
| `test_cohort_size_study.py` | Cohort-size study replicates that are not independent cohorts or not reproducible; summary not matching its replicates; unknown models accepted | ✅ |
| `test_reference_aphv.py` | Reference fit that misses a known Preece-Baines peak; children without a spurt or coverage not excluded | ✅ |
| `test_berkeley.py` | Wrong recoding of the Berkeley data; windows longer than a season, incomplete or not reproducible; equations not matching; real data not reaching the growth model | ✅ |
| `test_runners.py` | Unbounded ratios kept; label or time used as input; grouped folds sharing runners; temporal split overlapping; predictions not held out | ✅ |
| `test_validation_sim.py` | Validation pairs not from the pipeline, not reproducible, or error not growing with offset or with smaller body size | ✅ |
| `test_results_draft.py` | Untagged numbers in the generated results; synthetic sections not labelled; missing runs silently dropped | ✅ |
| `test_ieee_paper.py` | IEEE values not traceable to their run; pending runs not marked; macros used in `main.tex` that the generator never defines | ✅ |
| `cohort_settings.py` | Test helper: the realistic cohort config plus `sim/test_signal.yaml` | ✅ |
| `test_eval_metrics.py` | Wrong metric arithmetic (hand-worked and against scikit-learn); miscalibration not detected; row-level instead of athlete-level bootstrap; silently dropped subgroups | ✅ |
| `test_shap_runs.py` | SHAP on an unfitted estimator (v2); non-additive explanations; wrong mapping back to inputs | ✅ |
| `test_api_contract.py` | Models with differing call signatures or deprecated arguments (v2); what-if results presented as causal | ✅ |
| `test_sample_size.py`, `test_confirmatory.py` | Wrong sample-size arithmetic; a confirmatory model larger than the events support, or with predictors chosen by the data | ✅ |
| `test_paper_scaffold.py` | Methods text out of step with the configs; numbers in outlines; synthetic results outside the simulation section | ✅ |
| `test_dashboard.py` | Prescriptive or diagnostic prompt wording; risk asserted from maturity alone (v2); uncited prompts; identifying or video fields in the bundle; app crashes | ✅ |
| `test_evaluation_report.py` | Tables that cannot be traced; figures or tables in `paper/` without provenance | ✅ |

## `docs/`, `paper/`, `data/`, `apps/`

| File | Responsible for | Status |
|---|---|---|
| `docs/PROJECT_BRIEF.md` | The full specification (source of truth for scope) | ✅ |
| `docs/FILE_MAP.md` | This file | ✅ |
| `docs/CHANGELOG_REASONING.md` | Why each change and decision was made | ✅ |
| `docs/RESEARCH_STRATEGY.md` | Study-design and positioning decisions with their literature evidence: three-paper plan, sample-size-limited confirmatory model, novelty claims, equation cross-check, what remains for humans | ✅ |
| `docs/plans/Mx.md`, `Mx_done.md` | Plan before, and completion note after, each milestone | ✅ M0–M9 |
| `docs/protocol/01–04_*.md` | DRAFT SOPs: anthropometry, camera setup, screening tasks, injury surveillance | ✅ draft |
| `docs/protocol/05_retrospective_growth_records.md` | What to collect for a second real growth dataset (existing school/academy records), permissions first | ✅ draft |
| `docs/ethics/DRAFT_data_request_CIDAF.md` | Draft email requesting the Coimbra Soccer Longitudinal Project data (external validation on athletes) | ✅ draft |
| `docs/plans/M11.md` | Path A plan: growth twin validated on real data (sequential updating, forecasts, coverage fix) | ✅ |
| `docs/ethics/DRAFT_*.md` | DRAFT consent, assent, data-management plan — need ethics/legal review | ✅ draft |
| `docs/data_dictionary.md` | Every field, unit and allowed value | ✅ |
| `paper/manuscript_outline.md` | Single-paper structure: which layer is real data and which is simulation; numbers only via `paper/results_draft.md` | ✅ |
| `paper/ieee/main.tex`, `paper/ieee/references.bib` | The IEEE-format manuscript (IEEEtran journal): prose and macros only, numbers generated; PDF built to `paper/ieee/YouthTwin_IEEE.pdf` (git-ignored) | ✅ draft |
| `paper/results_draft.md` | Generated results section (`make paper-results`); every number tagged with its run | ✅ generated |
| `paper/preregistration.md` | Hypotheses, primary outcome/metric, ablation set — template to register before real data | ✅ template |
| `paper/tripod_ai_checklist.md` | TRIPOD+AI item → where it is satisfied | ✅ working copy |
| `paper/paperA_outline.md`, `paperB_outline.md` | Paper outlines mapped to reporting items; results as placeholders only | ✅ |
| `paper/paperB1_outline.md` | Outline of the growth-twin paper (separated from the prediction paper by decision) | ✅ |
| `paper/methods_draft.md` | Methods text generated from `configs/`; never edited by hand | ✅ generated |
| `paper/figures/`, `paper/tables/` | Generated only by `scripts/make_figures.py`, each file with a provenance sidecar; git-ignored (regenerate, never edit) | ✅ |
| `data/README.md`, `data/schema/README.md` | What lives where, privacy rules | ✅ |
| `data/schema/schema.sql` | SQL DDL for the nine tables | ✅ |
| `data/schema/growth_records_template.csv` | Column template for retrospective growth records (example rows are invented) | ✅ |
| `apps/twin_dashboard/app.py` | Streamlit research dashboard over a bundle directory; local only | ✅ |
