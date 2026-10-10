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
