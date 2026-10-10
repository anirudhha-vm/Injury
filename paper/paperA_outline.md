# Paper A outline — Validity of smartphone markerless motion capture for injury-relevant landing and squat kinematics in youth athletes across maturity stages

> Scaffold. No validation data exist yet. Every result is a `[[RESULT: …]]` placeholder to be
> filled by a script-produced `[[RUN:<run_id>|value]]` from `scripts/validate_markerless.py`.
> Design values are in `paper/methods_draft.md`, generated from `configs/`.
> Reporting guideline for agreement studies (e.g. GRRAS): TODO(human) choose and add a checklist.
> Target journal: TODO(human).

## Introduction
1. Movement screening in youth sport and why laboratory motion capture does not scale.
2. Promise of markerless pose estimation from smartphones (OpenCap, Uhlrich et al. 2023;
   MediaPipe; RTMPose) and what is known about its accuracy in adults.
3. Gaps: children and adolescents; changing body proportions across maturity; populations
   under-represented in training data of pose models.
4. Objective and pre-registered hypothesis H5: agreement with marker-based motion capture
   within a pre-specified limit, and not worse in pre-PHV children.

## Methods
| Section | Source |
|---|---|
| Participants: 30–40 athletes stratified by maturity; eligibility | TODO(human) |
| Maturity estimation and strata | methods draft: maturity estimation; agreement |
| Tasks, trials, events | methods draft: movement screening; `docs/protocol/03_screening_tasks_sop.md` |
| Cameras: two smartphones, placement, frame rate, synchronisation | `docs/protocol/02_camera_setup_sop.md` |
| Reference system: marker set, model, sampling rate | TODO(human): lab and system |
| Pose estimators compared | methods draft: movement screening |
| Processing: gap filling, filtering, angle definitions and sign conventions | methods draft; `src/youthtwin/pose/kinematics.py` |
| Time alignment of the two systems | `src/youthtwin/validation/mocap_io.py` |
| Agreement statistics and acceptance limit | methods draft: agreement |
| Sample size justification for an agreement study | TODO(human) |
| Ethics and consent | TODO(human) |

## Results
1. Participants and trials: [[RESULT: counts by maturity stratum and sex; trials excluded by
   the quality rule, with reasons]].
2. Agreement by variable, task and event: [[RESULT: bias, limits of agreement, RMSE, ICC with
   intervals — one table]]; [[RESULT: Bland–Altman grid]].
3. Agreement by maturity stratum and by sex (H5): [[RESULT: RMSE by stratum with intervals;
   comparison of pre-PHV against the other strata]].
4. Estimator comparison: [[RESULT: the same measures per pose estimator]].
5. Curves over the task window: [[RESULT: overlays of time-normalised curves]].
6. Proportional bias and its interpretation: [[RESULT: slopes]], noting that unequal error
   variances alone produce a small slope.

## Discussion
- Which variables are usable for screening and which are not, against the pre-specified limit.
- Sagittal versus frontal-plane variables; effect of out-of-plane movement on 2D angles.
- Whether accuracy depends on maturity, and plausible reasons.
- Limits: 2D projection angles; a single laboratory and marker model; the ankle angle's
  constant offset; event detection by thresholds.
- What this implies for using these variables as predictors in Paper B.

## Conclusion
[[RESULT: one or two sentences, written after the results exist]].
