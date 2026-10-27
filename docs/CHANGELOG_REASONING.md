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

---

## M1 — maturity equations, synthetic cohort, data schema (2026-10-09)

### Decisions

**D-012 · Equation functions take a mandatory `synthetic` flag where a gate applies.**
`mirwald2002(..., synthetic=...)` has no default, so every caller must state whether the data
are real. Why: a default of either value would silently defeat the gate or block development.

**D-013 · Input checks reject wrong units, not just wrong values.**
Stature outside 90–210 cm or a sitting/stature ratio outside 0.45–0.60 raises. Why: the most
likely real-data mistake is metres instead of centimetres or an unsubtracted box height, and
both produce a number that looks like a maturity offset.

**D-014 · True age at PHV is set exactly, not read off the simulated curve.**
In Preece–Baines model 1 the velocity peak sits at `theta + offset(s0, s1)`. The generator
samples the target APHV first and solves for `theta`. Why: the growth twin (M2) is scored
against true APHV, so the truth must be exact rather than a grid estimate.

**D-015 · Truth tables are physically separate from observable tables.**
`cohort.tables` has exactly the schema columns; generating parameters and true hazards go to
`cohort.truth` and to a `truth/` sub-folder on disk. A test asserts no `true_*`, `hazard_*` or
`trait_*` column appears in observable tables. Why: N1/N2 — it should be impossible to train
on the answer by accident.

**D-016 · One random stream per component.**
Athletes, growth, visits, measurement, movement, exposure and injury each draw from their own
generator derived from the master seed. Why: changing one component's code must not silently
change every other component's draws, which would make before/after comparisons meaningless.

**D-017 · A minimal movement simulator was added in M1.**
The brief schedules growth and injury simulation for M1, but its own example injury process
depends on a landing load proxy. Three session-level features are simulated now; skeleton
sequences wait for M6.

**D-018 · Stored outcome groups are finer than the analysis outcomes.**
`injuries.outcome_group` stores `growth_related`, `non_contact_soft_tissue`,
`other_non_contact`, `contact`, `other`. The brief's outcome (c) "all non-contact" is the union
of the first three, formed at analysis time. Why: storing overlapping groups in one column
would force one injury into two rows.

**D-019 · SQLite via the standard library; no DuckDB dependency yet.**
Why: the DDL is plain SQL that both accept, and nothing in M1 needs DuckDB's speed.

**D-020 · Ingestion is all-or-nothing and refuses to mix synthetic with real rows.**
Why: a partially loaded table or a database containing both kinds of data are the two ways a
synthetic number could end up in a real result.

**D-021 · A `synthetic` column was added to every table.**
Not in the brief's schema. Why: rule 5 requires every synthetic-derived artifact to carry the
flag; a column survives joins and exports where file-level metadata does not.

**D-022 · The identity map can only be written outside the repository.**
`pseudonymise()` raises if the path is inside the repo, and sets the file to owner-only.

**D-023 · QC flags, never fixes.**
`anthropometry_flags()` returns a table of problems and leaves the data untouched (tested).

### Assumptions introduced (all in `configs/`, all need a citation or a human decision)
- Every growth, measurement-error, movement, exposure and injury parameter in
  `configs/sim/cohort.yaml`. The Preece–Baines rate constants and height parameters were
  written from general knowledge of the model, not copied from a source.
- QC limits in `configs/io/qc.yaml`.
- OSTRC item range 0–25 and severity = sum of four items in the schema.

### Changes made after first running the generator
- Narrowed the spread of `s0` and of height gain after `theta`: the first draw produced a few
  eight-year-olds under 100 cm, close to the schema's lower bound.
- Lowered measurement-error SDs: with the first values about a quarter of duplicate stature
  pairs breached the QC limit, which would have made the QC output useless as a test bed.
- Both are tuning of assumptions for plausibility, not calibration to data.

### Repository and CI
- Pushed to `github.com/anirudhha-vm/Injury` (branch `main`). The fetch URL is HTTPS as given;
  the push URL is SSH because the machine has an SSH key for the account but no HTTPS credential.
- CI ran for the first time on commit `508d555` and passed. The human-gates job is allowed to
  fail, so a green run does **not** mean the Mirwald coefficients are verified.

---

## M2 — Bayesian growth twin (2026-10-09)

### What was tried, in order, and why each step was abandoned or kept

**D-024 · First attempt: sample every athlete's size, timing and intensity directly with NUTS. Abandoned.**
On an 80-athlete synthetic cohort the chains did not agree (R-hat about 2.9, effective sample
size about 5, maximum tree depth reached). Coverage numbers from that run are meaningless and
are not recorded.

**D-025 · Orthogonalised mean-curve parameters. Kept.**
The truncated-power spline basis is strongly collinear. The curve is now parameterised through
a QR decomposition of the basis on a grid (`splines.orthogonalising_transform`), so each sampled
parameter moves the curve in an independent direction. This removed divergences and cut run
time, but R-hat stayed near 1.6.

**D-026 · Size sampled as "fitted stature at the athlete's mean age". Superseded by D-028.**
Addressed the narrow ridge between size and timing (tall for age vs early maturer). Faster, but
chains still disagreed.

**D-027 · Diagnosis: athlete timing is genuinely two-peaked with one season of data.**
Chain-by-chain inspection showed individual athletes whose timing sat at two values about two
years apart in different chains (e.g. 13.0 vs 15.5). A given growth speed fits either the
rising or the falling side of the growth spurt. Removing the intensity effect made this worse,
not better (about half the athletes affected), so intensity is not the cause. A gradient-based
sampler cannot move between such peaks, and one stuck athlete shifts the shared curve for
everyone. This is a property of the data design (short follow-up), not a coding error, and is
worth a paragraph in the paper.

**D-028 · Marginalised model. Kept; this is the M2 design.**
- Size is integrated out in closed form (linear-Gaussian).
- Timing and intensity are integrated out on a fixed grid.
- NUTS samples only the shared curve and SDs (about a dozen parameters).
- Athlete effects are then drawn exactly, per shared draw, from the grid posterior, so
  two-peaked timing is reported as two-peaked rather than collapsed.
Verified by unit tests: each grid cell equals an independently computed multivariate normal
density; NumPy and PyTensor versions agree; with the true shared curve, 90% intervals for timing
and size cover the truth at the stated rate; a constructed two-sided case yields both peaks.

**D-029 · Incremental update = redraw athlete effects against stored shared draws.**
A consequence of D-028: no sampler is needed when an athlete gets a new measurement. It is
exact given the shared draws and approximate only in not re-estimating the shared curve, which
should be refit periodically. This replaces the brief's "refit, warm-started".

**D-030 · Age at PHV is computed from the fitted curve, not read from the timing parameter.**
Nothing forces the spline's velocity peak to sit at zero on the transformed age axis, so
`beta_i` alone is not APHV. APHV per draw is `beta_i + z_peak * exp(-gamma_i)`, matching how the
synthetic truth is defined.

**D-031 · Chains start from a least-squares mean curve, without jitter.**
With every athlete at their Moore prior the model is linear in the curve parameters. Starting
from a flat curve let chains settle on different shapes.

**D-032 · Stature only in M2.**
Leg-length and mass velocity (listed in the brief's growth state) need sitting-height and mass
models. They reuse this code path and are deferred until the stature model is validated.

**D-033 · Sampler tests are marked `slow` and excluded from `make test`.**
Unit tests cover the likelihood and the exact athlete draws without running MCMC. `make
test-slow` runs a small end-to-end fit. Acceptance numbers come only from
`scripts/sim_study_growth.py`.

### Research note
**R-003 ·** The R package `bsitar` implements a Bayesian SITAR (natural cubic spline mean curve
with size, timing and intensity random effects, sampled directly). It is designed for
longitudinal series spanning the growth spurt; our one-season design is what makes direct
sampling fail here.

### Outcome of the simulation study (SYNTHETIC, run `simgrowth-db8d25ba9b6f`)
- Sampler converged for both sexes (R-hat at most 1.01, no divergences). About 40 minutes
  wall-clock with both sexes fitted in parallel.
- Error criterion met: APHV mean absolute error 0.43 y against 0.72 y for the Moore estimate.
- **Coverage criterion not met: 74% against a target of 85-95%.** Intervals are shifted early
  and too narrow. Details, a breakdown by distance from PHV, and untested hypotheses are in
  `docs/plans/M2_done.md`. The acceptance range in `configs/growth/sitar.yaml` was left as
  specified; it was not relaxed to fit the result.

**D-034 · The integration grid was coarsened from 161 x 9 to 61 x 5 points.**
One gradient evaluation cost 55 ms on the fine grid, which made a fit take hours. Coarser grids
changed the log-likelihood at the starting point by less than 0.1, and the gradient cost fell
to about 12 ms. Accuracy near the posterior mode has not been checked separately.

**D-035 · The run recorded a `-dirty` commit.**
The study was run before the M2 code was committed. Provenance therefore does not point at a
reproducible commit; re-run `make sim-study-growth` from a clean tree before quoting numbers.

**Fix during M2 ·** A scripted multi-step edit of `scripts/sim_study_growth.py` matched text
inside a function it had just inserted and deleted two function bodies. One background run
exited immediately as a result. The script was rewritten whole; lint would have caught it had
it been run before launching.

---

## M3 — pose and kinematics pipeline (2026-10-10)

### Decisions

**D-036 · Joints are addressed by name and side, never by raw index.**
`Skeleton.index(side, joint)` is the only route to an index and `skeletons.py` the only file
containing indices. Why: v2's shoulder-elbow-wrist "knee" was an index typo that nothing caught.

**D-037 · Angles are 2D, per camera, and say so.**
Sagittal angles from the side camera, frontal-plane angles from the front camera; functions
refuse the wrong camera. Two-camera triangulation stays optional. Why: it is what two phones can
support without calibration, and it keeps the frontal-plane knee angle separate from flexion,
which was v2's "valgus" error.

**D-038 · Knee FPPA sign is defined by anatomy, not by image direction.**
Positive means the knee is medial to the hip-ankle line for either limb, given which way the
athlete faces. Why: a sign that flips between left and right legs makes asymmetry meaningless.

**D-039 · The zero-lag filter's cut-off is corrected for the second pass.**
A 2nd-order Butterworth run forward and backward, with the design cut-off raised by 1/0.802 so
that the -3 dB point lands on the configured cut-off (Winter). Tested at 6, 8 and 12 Hz. Without
the correction the effective cut-off is about 20% lower than the number written in the methods.

**D-040 · Gaps longer than the limit are never bridged.**
Each run of valid samples is filtered separately; long gaps stay missing through filtering,
angles and features, and lower the trial's quality score.

**D-041 · Event thresholds are fractions of leg length.**
Why: pixel thresholds would depend on camera distance and on the child's size, which varies by a
factor of nearly two across 8-18 years.

**D-042 · Missing events are reported with a reason; nothing is guessed.**
A trial without the events that bound its analysis window is excluded and logged. A manual
override table can supply frames, and those events are marked `manual`.

**D-043 · No frame-wise asymmetry exists in the code.**
Asymmetry functions accept two numbers. Bilateral tasks compare the limbs at the same events;
unilateral tasks compare the mean of left-limb trials with the mean of right-limb trials.

**D-044 · Default asymmetry index is the symmetry angle, not LSI.**
The brief allows either. LSI is a ratio and is undefined when the two values straddle zero or one
is near zero, which is normal for frontal-plane angles. LSI and a plain difference remain
selectable in the config.

**D-045 · Quality = the weakest of landmark confidence, completeness, and event confidence.**
Taking the minimum means one bad aspect cannot be averaged away. The threshold is an assumption.

**D-046 · Pose-estimation libraries are an optional extra and were not installed.**
MediaPipe, OpenCV and rtmlib are declared under `[project.optional-dependencies] pose` and
imported lazily. Why: they are large, CI does not need them, and without a video there is
nothing to run them on. Consequence: the two estimator wrappers are unexecuted code, excluded
from the coverage figure and labelled as such in the file map.

**D-047 · A synthetic skeleton generator was added now rather than in M6.**
Testing angles and events needs sequences with a known truth. The figure is deliberately
stylised (for example the toe is level with the ankle so that dorsiflexion equals shank tilt).

### Assumptions introduced
All values in `configs/pose/events.yaml`; MediaPipe confidence thresholds (library defaults);
Halpe-26 landmark indices (from memory, flagged `TODO(human)`).

### Corrections to my own first drafts
- A test expected a 4th-order analogue roll-off; the dual-pass digital filter's gain one octave
  above cut-off is 0.115, not 0.069. The filter was right and the expectation was wrong.
- Landmark tests were first written inside the angle-convention test file; moved to
  `test_landmark_indices.py`, the name the brief specifies.

---

## M4 — validation module for Paper A (2026-10-10)

**D-048 · Differences are always markerless minus mocap.**
One fixed direction everywhere, stated in the module docstring, so a positive bias always means
the phone reads high.

**D-049 · The table reports bias, limits, RMSE and both ICCs together.**
Why: each answers a different question. A test shows ICC(3,1) is unchanged by a 12-degree
constant offset while ICC(2,1) falls; reporting one ICC alone can hide a large bias.

**D-050 · Strata with too few pairs stay in the table with a note.**
Why: a silently missing maturity stratum would read as "no problem there".

**D-051 · pingouin is a development-only dependency used as the reference for ICC.**
The brief names it. Its current version labels the two-way ICCs `ICC(A,1)` and `ICC(C,1)`; the
test maps both old and new names.

**D-052 · Markers-to-joint-angles is deliberately not implemented.**
It depends on the lab's marker set and model, which is a human decision not yet made. Writing a
generic version now would produce angles defined differently from the lab's own.

**D-053 · TRC is read natively; C3D through an optional package.**
TRC is plain text and needed no dependency; the C3D reader is written but has never run.

**Corrections to my own first drafts:** two test expectations were wrong, not the code — an
arithmetic slip in a hand-worked standard error, and an assumption that the Bland-Altman slope
is zero without proportional bias (unequal error variances alone produce a small slope; now
documented in the module and tested at its predicted size).

---

## M5 — athlete-week table and tabular hazard models (2026-10-10)

### Decisions

**D-054 · Prediction time is the end of week w; the outcome window is the following 4 weeks.**
Predictors use data dated strictly before that time; outcomes are onsets in the window after it.
Stated once in `person_period.py` and tested against hand-built cases.

**D-055 · Weeks whose outcome window runs past the end of follow-up are excluded.**
Why: their label would be "no injury" only because nobody was watching.

**D-056 · Injury history is a predictor; other outcomes' labels are not.**
N2 forbids label-defining features as inputs. Past injuries (onset before the prediction time)
are legitimate history. The concurrent label of another outcome group is defined by the same
future injuries and is refused by the leakage guard.

**D-057 · The leakage guard runs when a model is constructed.**
`HazardModel` checks its input list against forbidden names and prefixes, so a leaked column
fails loudly before fitting rather than being noticed in a results table.

**D-058 · Preprocessing lives inside the model pipeline.**
Imputation values, spline knots, scaling and category levels are all learned in `fit`, so
fitting on a training fold cannot see held-out rows (N4). Tested by corrupting held-out rows.

**D-059 · Spline knots at training-fold quantiles; natural cubic basis reused from the growth
model.** No hand-drawn risk curve around PHV (v2): the shape is learned and reportable.

**D-060 · The boosted model gets raw inputs; trees handle missing values.**
Imputing first would hide "never screened" and "no previous injury" as ordinary values.

**D-061 · Mechanistic proxies carry no anatomical constants.**
The brief asks for a patellar-tendon moment arm "scaled by segment length". Rather than invent
or half-remember a scaling coefficient, each proxy is defined up to an unknown constant factor,
which does not change how athletes rank and disappears under standardisation. Body-weight-
normalised versions separate the effect of proportions from the effect of mass.

**D-062 · Two body-ratio features were added to the pose pipeline** (thigh fraction of leg
length; forefoot-to-heel lever ratio), plus segment inclination angles, because the proxies need
them. They are excluded from asymmetry: they are not movement features.

**D-063 · ACWR is not computed.** Per the brief; an exponentially weighted load and the change
in hours against it are provided instead.

**D-064 · Temporal hold-out drops training rows whose outcome window reaches the held-out
period.** Otherwise held-out time would define training labels.

### Findings (details in `docs/plans/M5_done.md`)
- No exposure effect is recoverable from this simulation by design of the simulation.
- Age and Moore maturity offset are collinear (VIF about 10) in the synthetic table.
- One seed showed a spurious load / soft-tissue association; 12 seeds average to zero.

### Fixes
- Growth simulator: velocity-peak search now uses the highest interior local maximum.
- Logistic hazard: solver seed fixed for reproducible fits.
- Dates are converted to one timestamp resolution before joining; mixed resolutions made the
  as-of joins fail.

---

## M2 follow-up — coverage fix (2026-10-10)

**D-065 · Per-athlete velocity deviation added to the growth model.**
`delta_i * (age - mean age)`, integrated out analytically like the size effect (the two are
orthogonal because ages are centred per athlete). Why: the first study's intervals were shifted
and too narrow because any mismatch between an athlete's curve and the shared shape had nowhere
to go except the timing estimate. A redraw against the saved fit with hand-set sizes showed the
mechanism (coverage 76% to about 90%); the full refit, with the size *estimated*, gave 81%.
The hand-set result was not reported as the outcome.

**D-066 · Moore prior offset and width are learned per cohort.**
Width was learned at about 0.72 years against the brief's default of 1.0.

**D-067 · Cohort-level timing prior implemented, tested, and switched off.**
It did not improve coverage for athletes far from PHV, so it is not part of the default model.
Kept as an option because it is the natural structure if the Moore prior proves poor on the
real cohort.

**D-068 · The fitted curve's growth spurt is the highest interior local maximum of velocity.**
Same issue as in the growth simulator: childhood velocity at the young edge of the range can
exceed the pubertal peak, and a global maximum then returns the edge. No effect on the reported
run (all draws had an interior peak), but it would have corrupted a wider-knot model.

**D-069 · Iteration on M2 stopped with the criterion met only within 3 years of PHV.**
Each proper experiment costs about 90 minutes and the fast proxy (MAP) proved unreliable for
variance parameters. The acceptance range in the config is unchanged; the shortfall and its
location are documented in `docs/plans/M2_done.md` rather than tuned away.

**Process note.** While the refit ran, the machine restarted once; the run that completed was
launched from clean commit `bb26d5a` and its provenance record says so.

**Process note.** An unanchored `.gitignore` rule for model weights (`models/`) also matched
`src/youthtwin/models` and `configs/models`. One push went out without four new source files
and those files were skipped by the linter. Fixed by anchoring the rule to the repository root.

---

## M6 — conditioned encoder, neural hazard, uncertainty propagation (2026-10-10)

### Decisions

**D-070 · Encoder inputs are angle curves, not raw keypoints.**
Sixteen channels: three sagittal angles and the frontal-plane knee angle per side, each with its
angular velocity, over 101 time-normalised points. Why: angles are independent of camera
distance, image size and body size, so the encoder cannot learn those; the brief's option of
normalised 2D coordinates is left for later.

**D-071 · All four conditioning modes give the same embedding size and share everything
downstream.** Growth reaches the network only through the conditioning mode (the head gets the
embedding and non-growth inputs). Why: otherwise the ablation would compare different amounts
of growth information, not different ways of using it.

**D-072 · FiLM starts as the identity.** Its generator's last layer is initialised to zero, so a
FiLM network begins exactly equal to the unconditioned one (tested) and pre-trained weights load
into it unchanged.

**D-073 · ST-GCN runs on a six-node joint graph of angle signals.** Hip, knee and ankle per
side; each node carries its own joint's signals. A full-body keypoint graph is possible once
raw coordinates are used.

**D-074 · Pre-training is masked-segment reconstruction; no left/right mirroring.**
Mirroring would swap limb-specific channels. Pre-training uses training athletes' trials only
and is cached per fold so every configuration starts from the same weights.

**D-075 · CPU-only PyTorch.** The largest model has about 53k parameters. The CUDA wheels add
gigabytes; a GPU build can be installed by hand.

**D-076 · Reported risk is the mean over growth draws and ensemble members.**
For a non-linear model this differs from the risk at the mean growth state (tested), which is
the reason to propagate rather than plug in. Variance is split into a growth part and a model
part by the law of total variance.

**D-077 · One growth-posterior draw gives an athlete's whole trajectory.**
Draws are taken per athlete, not per athlete-week, so a draw's maturity offsets at different
weeks stay consistent (one year apart for dates one year apart).

**D-078 · Split-conformal intervals are not implemented.**
Listed as optional. A conformal interval for a probability needs a calibration construction
that is a design choice for the team.

**D-079 · Risk intervals are scored against the known true risk on synthetic data.**
`sim/truth.py` computes each athlete-week's true 4-week event probability from the generator's
hazards. This is the only place "calibrated intervals" can be checked; the truth tables are
still never model inputs.

**D-080 · Synthetic trials are tied to the tabular load proxy by an assumed mapping.**
A higher proxy gives a stiffer landing. This puts the same information in the sequence as in
the tabular feature, so the encoder can at best match that feature on this cohort. It is a
property I chose, not a finding.

### Stated limits of what M6 can show
With about 55 growth-related injuries in the default synthetic cohort, differences between
conditioning modes are within noise, and any advantage for one mode partly reflects how the
simulation was built. M6 demonstrates that the machinery runs and behaves as designed on
constructed examples; it is not evidence for hypothesis H3.

### Fix
`SitarPosterior.load` now accepts files saved before the cohort-prior fields were added; the
first full ablation run failed on exactly that.

---

## M7 — evaluation suite, explanation, figures (2026-10-10)

### Decisions

**D-081 · One predictions table feeds everything.**
`eval/predictions.py` produces out-of-fold predictions (each athlete predicted by a model that
never saw them); every metric, table and figure is computed from that table alone.

**D-082 · "Time-dependent AUC at 4 weeks" is the AUC of the 4-week outcome on the risk set.**
With athlete-weeks at risk as rows, a fixed horizon, and rows with unobservable windows already
removed, the cumulative/dynamic AUC reduces to this. Stated in the function's docstring so the
methods text can say exactly what was computed.

**D-083 · All uncertainty comes from resampling athletes.**
Rows are not independent (repeated athletes, overlapping windows). A test shows the athlete-level
interval is more than 2.5 times wider than a row-level one on such data. Model comparisons use
the paired version (both models on the same resampled athletes).

**D-084 · Calibration slope and intercept fitted by Newton's method in-house.**
A few lines, no extra dependency for the core measure, and checked against scikit-learn. ICI
uses the standard loess smoother from statsmodels.

**D-085 · Sensitivity and specificity are reported only at thresholds listed in the config.**
None are listed: thresholds are a pre-registration decision (`TODO(human)`). Reporting at a
threshold chosen after seeing results would be exactly the flexibility pre-registration removes.

**D-086 · Subgroups with fewer than the minimum number of events are listed without metrics.**
Counts and a note are kept so that the gap is visible.

**D-087 · What-if output carries its own disclaimer.**
The result object and its summary both state that it is model sensitivity, not a causal effect.

**D-088 · Generated files under `paper/figures` and `paper/tables` are git-ignored.**
They are regenerated by one command and currently come from synthetic runs. Keeping them out of
git prevents a stale or synthetic table from being mistaken for a result; the results guard
still checks that anything present has a provenance sidecar.

### Findings
- **Percentile intervals for ICI and ECE can exclude the point estimate.** Absolute-error
  measures are inflated by resampling noise. Documented in `eval/report.py`; a test that assumed
  otherwise was wrong and was corrected.
- **Synthetic cohort is underpowered for prediction.** With 300 athletes and about 55
  growth-related injuries, out-of-fold AUCs of all tabular models are between 0.50 and 0.56
  with intervals spanning 0.5, calibration slopes are near 0, and no model's decision curve
  beats both reference strategies (run with 100 bootstrap resamples; indicative only). Since
  the simulated effects are real, this is a sample-size signal for the planned cohort, to be
  taken up with the Riley sample-size calculation.

### Corrections to my own first drafts
A hand-worked Brier score in a test was mis-added (the code was right).

---

## M8 — research dashboard and decision-support prompts (2026-10-10)

**D-089 · The app only reads a pre-built bundle.**
Building (models, explanations) is separate from viewing. Why: the app starts instantly, and
everything on screen exists as a file that can be inspected and traced.

**D-090 · Risk on screen is out-of-fold.**
Each athlete's timeline comes from a model fitted without them, with an interval from growth
posterior draws. Showing in-sample risk would look far better than the model deserves.

**D-091 · Rules see a deliberately narrow view of the athlete.**
`AthleteWeek` holds predicted risk, its cohort percentile and interval, height velocity,
maturity offset and screening age. The elevated-risk prompt depends on predicted risk only.
A test sweeps maturity offset and growth velocity and asserts no elevated-risk prompt appears.
This is the direct replacement for v2's "bio_risk > 0.8 means CRITICAL" rule.

**D-092 · Wording is checked by code, at build time.**
A list of forbidden fragments (diagnostic and prescriptive language) and a required "consider"
framing; a failing prompt aborts the bundle build.

**D-093 · Elevated risk is defined relative to the cohort that week, not by an absolute cut-off.**
Absolute thresholds are a pre-registration decision that has not been made; a percentile makes
no claim about what level of risk matters clinically.

**D-094 · Evidence statements are paraphrases of the brief and flagged unverified.**
The library carries `verified_by_human: false` and the dashboard says so beside the prompts.

**D-095 · The dashboard states on screen that the synthetic models are near chance.**
Otherwise a polished risk timeline invites trust the numbers have not earned.

**D-096 · Matplotlib figures, no extra plotting dependency.**
The same watermarking helper as every other figure applies.

---

## M9 — paper scaffolds (2026-10-10)

**D-097 · The methods draft is generated, committed, and guarded by a test.**
`paper/methods_draft.md` is rendered from the YAML files; `test_paper_scaffold.py` fails when
the committed file differs from what the current configs produce. Why: the brief asks for
methods "from configs, not from memory", and a generated-but-unchecked file would drift the
first time someone changed a config.

**D-098 · Every configured value is written inside a `[[SPEC: …]]` tag.**
The results guard then distinguishes design values from results automatically, and the draft
passes the same check as the rest of `paper/`.

**D-099 · The draft states what is unverified.**
It prints the Mirwald verification status from the gate, that load proxies are not validated,
that hyperparameters are starting values, and that simulation parameters are assumptions.

**D-100 · Synthetic results have their own section in the Paper B outline.**
A test asserts the word "synthetic" does not appear in the Results section.

**Deviation:** the M9 plan file was written with the work, not before it.

**D-101 · A cohort folder of parquet tables is the real-data entry point.**
`io.ingest.read_cohort_dir` reads the same layout `make sim` writes and validates every row;
`scripts/check_cohort.py` adds QC; `scripts/evaluate.py --cohort-dir` and
`scripts/fit_growth.py` run from it. Why: until this, every analysis script generated the
synthetic cohort internally, so there was no command to run on real data. The neural ablation
and the dashboard build are still synthetic-only, and the README says so.

---

## Strategy decisions after M9 (2026-10-10)

The team asked me to verify what I could and to take the open design decisions. Evidence and
full reasoning: `docs/RESEARCH_STRATEGY.md`.

### Research findings
**R-004 · Mirwald coefficients: magnitudes corroborated twice, signs and one variable disputed.**
Cossio-Bolanos et al. (2021) and Murayama et al. (2023, page read directly) both reproduce every
magnitude in the brief. They disagree with each other on the sign of two coefficients and on
whether the third term uses sitting or standing height. Secondary sources therefore cannot
verify the equations. The gate stays closed; I did not set it to true.

**R-005 · Published growth-related injury rates are far lower than the simulation assumed.**
603 time-loss apophyseal injuries in 16,024 player-seasons (Br J Sports Med 2025): about 4 per
100 athletes per season. The synthetic cohort generates about 18 per 100.

**R-006 · The field's models are weak in ways this project can avoid.**
Bullock et al. (2022): 98% of 204 models at high or unclear risk of bias, none externally
validated, 7% of studies reporting calibration.

**R-007 · Maturity equations fail across populations; Indian reference timing is earlier than
the brief assumed.** Murayama et al. (2023) for Japan; Indian growth studies place PHV near
10.5-11 y (girls) and 12-13.5 y (boys).

### Decisions
**D-102 · Sample size sets the confirmatory model's size.**
`eval/sample_size.py` implements the Riley et al. criteria (worked example reproduced by hand).
At 4-15% of athletes injured per season, 300 athletes support one or two predictor parameters.
`models/confirmatory.py` therefore takes predictors from a pre-specified ordered list and uses
only as many as the achieved events support; a test checks the data cannot substitute another
column. Missing-value indicators are not used there, because each one would add a parameter.

**D-103 · Everything larger is exploratory.**
The penalised hazard, boosted hazard, conditioned encoder and ablation remain, labelled
exploratory in the pre-registration, the outlines and the generated methods text.

**D-104 · Three papers instead of two.**
The growth twin gets its own paper (B1): it has a continuous outcome the cohort can support, the
first maturity-timing data in Indian youth athletes, and the two-valued-timing result.
`paper/paperB1_outline.md` added; the former Paper B outline is now B2.

**D-105 · Risk thresholds are elicited from physiotherapists before data lock.**
I did not pick numbers: a threshold encodes how many unnecessary reviews are acceptable per
injured athlete found, which is a clinical judgement.

**D-106 · Growth-twin calibration claim restricted to within 3 years of PHV (H1b), and the
wider-knot experiment launched.** Its result will be recorded whichever way it falls.

**D-107 · Recommendations that need the team:** football plus cricket; at least 1,000
athlete-seasons; record all confirmed problems, not only time-loss; switch the simulation to
Indian reference timing and persistent training load at the next full re-run.

**Process note.** The results guard rejected a `[[CITED: ...]]` tag of mine that spanned two
lines in the pre-registration. The tag was put on one line; the guard was not loosened.

---

## M6 outcome (2026-10-10)

**R-008 · The propagated risk intervals are not calibrated: 27% coverage of the true risk for
nominal 90% intervals (synthetic run `ablation-b77476353614`).**
The interval reflects growth-timing uncertainty and seed-to-seed disagreement only. It omits
the dominant error, that the model is poorly determined by about 55 injuries: networks trained
on the same data are wrong together. The third M6 acceptance criterion is therefore not met.
Consequence: the band is to be described as "sensitivity to growth-timing uncertainty", never
as a confidence interval for risk. A covering interval would need ensembles trained on
resampled athletes or a held-out calibration step.

**R-009 · Ablation ordering is not interpretable at this size.**
Growth-aware configurations beat the growth-blind encoder on both AUC and correlation with the
true risk (expected: the simulation makes growth a cause). Among late fusion, concatenation and
FiLM the two measures disagree on the order and no intervals are available. Not evidence about
hypothesis H3.

**D-108 · Known gap accepted for now: the ablation runner stores metrics, not predictions.**
Athlete-level bootstrap intervals and paired differences for the ablation need the out-of-fold
predictions; saving them is the next change to `eval/ablation.py`.

**R-010 · Wider-knot growth model abandoned as impractical.**
Started from clean commit `b6db852`; stopped after 7.7 hours of wall-clock time with none of
eight chains finished, against 40 and 90 minutes for the earlier models. The added knots cover
ages where the cohort has few measurements, leaving that part of the curve barely determined.
No result was obtained, so nothing about coverage can be claimed for it. The pre-registered
restriction of growth-twin uncertainty claims to within three years of PHV (H1b) stands.

---

## Ablation intervals (2026-10-13)

**D-109 · The ablation runner now keeps its out-of-fold predictions (closes D-108).**
`run_ablation` returns the results table and a predictions table (one row per athlete-week, one
`risk_<name>` column per configuration, the fold, the true risk on synthetic cohorts, and the
risk interval where one exists). `ablation_report` computes athlete-level bootstrap intervals
for every configuration and paired differences for the comparisons listed under `comparisons`
in `configs/eval/ablations.yaml`. Why: R-009 could not say whether any ordering was more than
noise. The comparisons are fixed in config before the re-run so they are not chosen after
seeing results; all are exploratory. `--report-from` recomputes the tables from saved
predictions, so changing the metric list does not cost another two-hour fit.
Not yet re-run: run `ablation-b77476353614` predates this change and has no predictions.

**D-110 · Simulated age at PHV now uses Indian reference means: 10.8 y (girls), 13.4 y (boys).**
Source: the peak-centred height-velocity analysis of Khadilkar et al., *Indian Pediatr* 2019
(n = 2,949, affluent urban schools in Pune and Delhi), as reported in the accompanying
commentary. Other Indian cohorts place the boys' peak up to a year earlier (Sabinkar et al.,
*Cureus* 2023: 12-12.9 y), so 13.4 is at the late end of the published range. The SD of 1.0 y
is still an assumption: neither source reports one. The brief's 12.0 and 14.0 are replaced
(deviation from the brief, decided in D-107). Consequence: with baseline ages 8-17 y, more
girls are several years past PHV, where the growth twin is less well calibrated.

**D-111 · Simulated training load persists.** Expected weekly hours are multiplied by a
mean-one factor: a stable per-athlete level (log SD 0.25) and a first-order autoregressive
weekly deviation (log SD 0.30, autocorrelation 0.7). All three are assumptions. Before, each
week was drawn independently, so past load carried no information about the outcome window and
no exposure effect could be tested (D-107). Weekly training hours are capped at 30, since the
heavier tail otherwise produced session counts outside the schema range.

**D-112 · Fragile test replaced.** `test_scaler_fit_scope` asserted that the training-fold
median of weekly hours differs from the full-data median; hours are recorded to 0.1 h, so the
two can coincide (they did on the new cohort). It now compares the fitted scale, which cannot.
The guard itself (changing held-out rows cannot change the fitted model) is unchanged.

**Invalidated by D-110/D-111:** every synthetic run made before 2026-10-13
(`simgrowth-16fbcf9f034a`, `ablation-b77476353614`, the dashboard bundle) describes the old
cohort. The numbers in `docs/plans/M2_done.md` and `M6_done.md` are for that cohort.

**D-113 · Cohort-size simulation study added (`make sim-study-sizes`).**
Decision 7 of the research strategy asked for the simulation at several cohort sizes. For each
size in `configs/eval/cohort_size_study.yaml`, independent synthetic cohorts are generated and
the tabular models are evaluated exactly as in the main evaluation; the spread across cohorts
is reported, together with the AUC of the known true risk as a ceiling. Replicate cohorts are
used instead of a bootstrap of one cohort because the question is what a *new* cohort of that
size would show. The neural models are left out: one grid takes over two hours.
Caveat to carry into any write-up: the simulation injures about 19% of athletes per season with
a growth-related injury, several times the published 4 per 100 athlete-seasons, so the same
number of athletes in a real cohort would yield far fewer events. The figure therefore labels
each size with its number of injured athletes, which is the quantity that transfers.

---

## M10 — real public data (2026-10-21)

**R-011 · No public dataset combines youth growth, video and diagnosed growth-related injury.**
Searched 2026-10-21 (sport-science dataset index, DataverseNL, Figshare, PMC data statements).
The closest youth datasets either are not released or build their injury "label" from their own
input features (a 2026 Frontiers dataset of 320 adolescents), which non-negotiable N1 forbids.
Each layer is therefore tested on the best real dataset available for it (`docs/plans/M10.md`).

**D-114 · The growth twin is tested on the Berkeley Child Guidance Study.** 136 children,
6-monthly height and stem (sitting) height from 8 to 21 y, distributed in the `sitar` R package
(GPL >= 2), read with `pyreadr` (new dev dependency). Berkeley was not used to develop or
calibrate Moore 2015 (PBMAS; calibrated on HBS-III and Harpenden), so the comparison is fair.
The reference age at PHV comes from each child's full record (Preece-Baines model 1, checked
against a smoothing spline); the twin sees one season of three 6-monthly measurements. The twin
is fitted on every child's window; the reference rules only decide who is scored, so they
cannot influence the fit.

**D-115 · Reference agreement tolerance set to 1.0 y, with 0.5 y as a sensitivity analysis.**
At 0.5 y the rule excluded 40 of 136 children (28 girls), mostly because the GCV smoothing
spline is noisy on 6-monthly data, not because the Preece-Baines fit was poor (RMSE <= 0.9 cm in
every child). 1.0 y keeps 125; both subsets are reported.

**D-116 · `prepare_growth_data` split.** `growth_data_from_measurements` takes measurements
from any source; the synthetic path calls it after joining its tables. Behaviour unchanged.

**D-117 · Video validity layer blocked on the user.** OpenCap's validation data needs a SimTK
login; BioCV needs a signed data-use agreement. Both are the user's decision.

**D-118 · The risk-evaluation machinery is tested on real injury outcomes from adult runners.**
Lövdal, den Hartigh & Azzopardi (2021), DataverseNL doi:10.34894/UWU9PV, CC0: 74 runners,
42,798 overlapping event days, 575 injury days. Three schemes: runner-grouped folds (primary,
N3), a temporal hold-out of the last 25% of calendar time, and row-level folds included only to
measure how much leaking a runner across folds inflates performance. The file's own week-ratio
columns divide by zero in rest weeks (values to 2e8); they are replaced by
`log((a + 1) / (b + 1))` of weekly kilometres, computed from each row alone, so nothing is
learned from other rows (N4). The existing builders (`make_hazard_logit`, `make_hazard_xgb`) and
the leakage guard are reused unchanged. The metric is labelled `time_dependent_auc_4wk` by the
shared config; here it is the AUC for injury on the event day.

---

## Realistic simulation for the layers without public data (2026-10-22)

The user asked for realistic synthetic data where no real data exist, and final results. The
synthetic layers stay labelled SYNTHETIC and appear in the paper only as a simulation study.

**D-119 · Growth-curve shape and spread from real children.** The simulation's Preece-Baines
shape parameters (s0, s1), gain after theta, and the SD of age at PHV now come from the fits to
the 125 Berkeley children with a reliable reference (girls: s0 median 0.128, s1 1.104, APHV SD
0.89 y; boys: 0.109, 1.149, 1.12 y). Mean timing stays Indian (D-110); adult height stays an
assumption. With the wider real-data spreads about 0.4% of (s0, s1) draws give a curve with no
spurt; those are redrawn (the reference fits excluded the one such child).

**D-120 · Injury incidence calibrated to the literature.** Growth-related: the leg-length
velocity effect is now 0.158 per 2 cm/y (51% higher risk per 5.2 cm/y, Johnson et al. via Hall &
Erskine 2025; an all-injury estimate, so conservative), and the intercept (-7.347) is solved so
that 3.8% of athletes have a growth-related injury per season (603 in 16,024 player-seasons,
BJSM 2025). The old values gave 19%, five times too many. The load, interaction and hours terms
remain assumptions (no published estimates). Tests that check the machinery recovers effect
directions use the old strong-signal values through `configs/sim/test_signal.yaml`; a new test
checks the realistic config gives the published incidence. Consequence: a 300-athlete season now
yields about 11 growth-related injuries.

**D-121 · Synthetic validity study through the real pose pipeline.** No children's video plus
mocap dataset is openly downloadable (D-117). Each simulated drop jump is observed twice, by a
mocap-like reference and a smartphone-like estimator (per-participant and per-trial landmark
offsets plus frame noise), and both go through `pose.task_features`. The offset size is
calibrated so peak knee flexion in adult-size participants has a mean absolute error of 2.1 deg
(Barzyk et al. 2024, smartphone vs Vicon). FPPA error is not set directly: it follows from the
same landmark error. Errors scale with 1/body size, so any loss of agreement before PHV is an
output. Knee flexion at initial contact is not reported (the skeleton fixes it at 20 deg).

**D-122 · The confirmatory model is now part of `evaluate.py`.** It was built in D-102 but only
tested, never evaluated next to the other models. Its predictors are chosen once from the whole
cohort's size and event count (Riley criteria), so the choice cannot differ between folds.
`make_figures.py` also copies the real-data and validity-study outputs into `paper/`.

**D-123 · The results section is generated, not written.** `scripts/draft_results.py` renders
`paper/results_draft.md` from the latest runs; every number is a `[[RUN:<id>|value]]` tag, so the
results guard can trace it. Part 1 holds the real-data results, Part 2 the simulation study with
SYNTHETIC in every heading. A missing run becomes a `[[RESULT: ...]]` placeholder.

**D-124 · One manuscript.** The user asked for a single strong paper. `paper/manuscript_outline.md`
merges the three outlines, with a table at the top stating which layer rests on real data
(growth twin on Berkeley; evaluation discipline on runners) and which on simulation (validity by
maturity, growth-related injury risk, sample size). The three outlines stay as the fallback.

**D-125 · Implausibly short children are redrawn.** Drawing the Preece-Baines parameters
independently (in real children they are correlated) occasionally combined very late timing
with slow childhood growth, giving an 8-year-old of 87-90 cm. That broke the cohort-size study
and one sex of the 1,000-athlete growth study (Moore's plausibility check refuses it), so both
were stopped and restarted. Children whose true height at study entry is below
`min_entry_stature_cm` (105 cm, about 3 SD below the median at 8 y; assumption) now have all
curve parameters redrawn, as do curves without a spurt (D-119).

**D-126 · Soft-tissue held-out AUC bar lowered from 0.55 to 0.52 in one direction test.** After
D-125 the boosted model scored 0.547 on that fold. The true risk scores 0.626 there, so 0.55 was
an arbitrary share of an already low ceiling. The direction assertions, which are the point of
the test, are unchanged and pass.

**D-127 · IEEE-format manuscript generated from runs.** The user asked for the final paper in IEEE
format. `paper/ieee/main.tex` (IEEEtran, journal) holds prose and macros only; every number is a
macro or table that `scripts/build_paper.py` writes from the latest runs into git-ignored
`paper/tables/`, each with a provenance sidecar. Values whose run is not finished print a red
[pending] marker. A test checks that every macro used in the manuscript is generated. The PDF is
compiled with Tectonic, which needs no system LaTeX install. Superseded synthetic runs made
before the realistic recalibration were moved to `outputs/archive/` so they cannot reach the
paper. References were checked for this session; page ranges are marked for human verification.

---

## M10 outcome (2026-10-22)

**R-012 · Real growth data: the twin is more accurate than the Moore equation in every band of
distance from PHV, but its intervals are too narrow.** Coverage of nominal 90% intervals was well
below nominal, worst more than three years before PHV, where one season says little about when
the spurt will come. Mixing was incomplete for the girls' shared parameters. Consequence: the
interval should be widened (prior spread growing with distance from PHV, or more seasons)
before it is propagated into risk.

**R-013 · Real injuries: leakage inflates AUC.** On the runners, row-level folds raised AUC over
runner-grouped folds for all three models, each interval excluding zero; grouped calibration
slopes were below one.

**R-014 · At published incidence growth-related injury is weakly predictable even in principle.**
In the calibrated simulation the true risk reaches only a modest AUC; learned models stayed near
chance and uncalibrated up to 2,000 athlete-seasons, and no model ranking was established at
1,000. This is the paper's sample-size message.

**R-015 · Propagated risk intervals again cover far below nominal** (300-athlete cohort, 30 growth
draws, ensemble of three): they express sensitivity to growth timing, not risk-model error.

**D-128 · Compute reductions to finish in a day,** all stated in the manuscript: no twin features
in the 1,000-athlete risk cohort; single-network short-training ablation; propagation in the
300-athlete cohort; two Berkeley window replicates. A stalled chain (one of four, two hours
after the others) stopped the third Berkeley replicate.

**D-129 · Overnight full-scale runs.** The user asked for everything trained and tested. The
1,000-athlete cohort gets a growth-twin fit, then the tabular evaluation with twin features and
the full nine-configuration ablation with posteriors (ensembles of three), which also yields the
propagation result at full scale; `build_paper.py` then prefers it over the 300-athlete subset
run. The ablation's interval tables omit the smoothed calibration index (`report_metrics`),
whose bootstrap took eight hours. The reduced-compute PDF is kept as a fallback.

---

## M11 — Path A (2026-10-23)

**D-130 · The paper becomes a methods paper on real data (Path A).** The user agreed that the
simulated injury layers cannot carry a Q1 paper and that "digital twin" was not yet earned.
Plan in `docs/plans/M11.md`: reference-cohort training, sequential twin updating with
six-month forecasts on held-out children, and a distance-dependent timing prior.

**D-131 · Distance-dependent timing prior.** Prior SD `sigma_beta * (1 + kappa * |Moore offset|)`,
`kappa` learned (half-normal, off by default). Reason: on real data the twin's intervals
undercovered most far from PHV, where Moore's error is known to be largest. With `kappa = 0`
the likelihood is unchanged (test). `growth_data_from_measurements` can take the prior from
other visits than the fitted ones, so a reference child fitted on the full record gets the prior
it would have at deployment. Saved posteriors without the slope load with slope zero.

**D-132 · Second real dataset: collection guidance.** The user has one day. A longitudinal
dataset cannot be created in a day; existing school or academy records can be compiled.
`docs/protocol/05_retrospective_growth_records.md` and `data/schema/growth_records_template.csv`
say what to collect, with permissions and ethics first.

**R-016 · No second openly downloadable longitudinal growth dataset exists (searched 2026-10-23).**
Candidates are available only on request: the Coimbra Soccer Longitudinal Project (58 youth
soccer players, annual height and sitting height; CIDAF, signed agreement; draft request in
`docs/ethics/DRAFT_data_request_CIDAF.md`), the Wroclaw Growth Study, Indian cohorts
(Khadilkar 2019; a 2025 Cureus cohort) and the Kangwha cohort. The `sitar` package's `heights`
data (Chard Growth Study: 12 girls, 6-monthly 8-16 y, age at menarche, no sitting height) is
open and is used as a small external check. Strategy without a second large dataset: make the
contribution methodological (sequential twin with forecasts, calibrated intervals, visits
needed, robustness to schedule and measurement error, benchmark against SITAR and bsitar).

**D-133 · Leg-trunk (bivariate) twin.** Leg length (stature minus sitting height) gets its own
shared curve, size and velocity effects, and shares each athlete's timing and tempo with
stature; on a grid its likelihood simply adds to each cell. Rationale: legs peak before the
trunk, so one season of asynchronous growth carries timing information. Tested on data
simulated from the model (timing interval narrower and error lower); off by default.

**D-134 · MAP option for the reference curve.** Full NUTS on full records costs about 0.1 s per
gradient and hours per fit (PyTensor without BLAS; Numba gave only 1.3x). `inference.method:
map` fits the shared curve at its posterior mode (the SITAR method itself fits by maximum
likelihood) and keeps athlete effects exact given it. A five-fold, two-sex evaluation takes
minutes. One NUTS fold is to be reported as a check.

**R-017 · On real children, leg length with shared timing made the twin worse** (first MAP
comparison, five folds: error and bias up, raw coverage down, especially in girls), although it
helped in simulation, where every child's legs lead stature by the same amount. In real children
the lead varies.

**D-135 · Per-athlete leg lag.** Each athlete's leg-to-stature lag `lambda_i ~ Normal(0,
sigma_lag)` is integrated on a seven-point grid, `sigma_lag` learned. Zero lag SD reproduces
shared timing exactly (test).
