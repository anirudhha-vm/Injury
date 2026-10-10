# Paper B outline — An uncertainty-aware, maturity-conditioned digital twin for dynamic prediction of growth-related injury in youth athletes

> Scaffold. No results exist for real athletes. Every result is a `[[RESULT: …]]` placeholder to
> be filled by a script-produced `[[RUN:<run_id>|value]]`. Design values are in
> `paper/methods_draft.md`, generated from `configs/`. TRIPOD+AI item numbers refer to
> `paper/tripod_ai_checklist.md` (working paraphrase; verify against the official checklist).
> Target journal: TODO(human).

## Title and abstract (TRIPOD+AI items 1–2)
- Title names the model type (development and internal validation of a prediction model), the
  population (youth athletes aged 8–18 y) and the outcome (growth-related injury).
- Structured abstract: background, objective, methods, results, conclusions.
  Results: [[RESULT: cohort size, number of events, primary metric with interval, calibration slope]].

## Introduction (items 3–4)
1. Burden of growth-related injury in youth sport (apophysitis, apophyseal avulsion, physeal
   injury, bone stress). Citations: TODO(human).
2. Maturity timing and injury type: growth-related injuries cluster before and around PHV,
   soft-tissue injuries increase after it (Hall et al. 2022; Sports Med Int Open 2024). A single
   "risk peaks at PHV" curve is therefore wrong.
3. Why single screening tests fail to predict injury (Bahr 2016), and what would have to be
   different: repeated screening, time-varying covariates, dynamic prediction, clinical
   usefulness assessed by decision curves.
4. Limits of maturity-offset equations: bias toward the mean, larger error far from PHV,
   derivation samples of mainly European ancestry (Malina & Kozieł 2014; Moore et al. 2015).
   Generalisation to Indian youth is unknown.
5. Gap: no individual, uncertainty-aware, dynamic model combining growth and movement.
6. Objectives and pre-registered hypotheses H1–H4 (`paper/preregistration.md`).

## Methods (items 5–19; order follows TRIPOD+AI)
| Section | Item | Source |
|---|---|---|
| Source of data, dates, setting | 5 | `docs/protocol/`; TODO(human): academies, dates |
| Participants, eligibility | 6 | TODO(human) |
| Data preparation and quality control | 7 | methods draft: movement screening; anthropometry QC |
| Outcome: physiotherapist-diagnosed, OSIICS-coded, three groups; blinding | 8 | methods draft: outcomes; `docs/protocol/04_injury_surveillance_sop.md` |
| Predictors: growth state, screening features and embedding, exposure, history | 9 | methods draft |
| Sample size (Riley et al. 2020) | 10 | TODO: `scripts/sample_size.py` once incidence is known |
| Missing data | 11 | TODO: decide and pre-register |
| Growth twin (C1) | 12 | methods draft: growth twin |
| Movement pipeline and conditioned encoder (C2) | 12 | methods draft: models |
| Mechanistic load proxies (C4), labelled as proxies | 12 | methods draft |
| Hazard models (C3), uncertainty propagation | 12 | methods draft: models |
| Class imbalance: proper likelihood, no resampling of athlete-weeks | 13 | `configs/models/hazard.yaml` |
| Fairness: subgroups by sex, maturity stratum, sport, age band | 14 | methods draft: validation |
| Model output: 4-week risk with interval; thresholds | 15 | TODO(human): pre-specify thresholds |
| Development versus evaluation data: athlete-grouped folds, temporal hold-out | 16 | methods draft: validation |
| Ethics and consent | 17 | TODO(human): committee, approval number; `docs/ethics/` drafts |
| Open science: protocol, registration, code, data | 18 | repository; TODO(human): registration ID |
| Patient and public involvement | 19 | TODO(human) |

## Results (items 20–24)
1. Participant flow and characteristics: [[RESULT: flow diagram counts; table of baseline
   characteristics by sex and maturity stratum]].
2. Injury incidence by type per exposure time: [[RESULT: incidence table]].
3. Growth twin accuracy against athletes who passed PHV during follow-up (H1):
   [[RESULT: error of posterior age at PHV versus Moore and Mirwald estimates; interval coverage]].
4. Discrimination, calibration and net benefit for growth-related injury (H2):
   [[RESULT: primary metric with athlete-level bootstrap interval for each model]];
   [[RESULT: calibration intercept, slope and curve]]; [[RESULT: decision curves]].
5. Ablation of conditioning modes (H3): [[RESULT: nine-configuration table with paired
   differences against the pre-specified comparator]].
6. Effect of propagating growth uncertainty on calibration (H4):
   [[RESULT: calibration measures with and without propagation]].
7. Secondary outcomes (soft-tissue, all non-contact): [[RESULT: same measures]].
8. Subgroups: [[RESULT: performance by sex, maturity stratum, sport, age band]].
9. Explanations: [[RESULT: input importance; example attributions]], presented as descriptions
   of the model, not causes of injury.
10. Full model specification and how to obtain predictions (item 22): TODO.

## Simulation study (synthetic data only)
Reported separately from the results above and labelled as simulation throughout.
- Growth twin recovery of a known age at PHV: [[RESULT: coverage and error from
  `scripts/sim_study_growth.py`, overall and by distance from PHV]].
- Recovery of simulated effect directions by the hazard models: [[RESULT: from tests and
  `scripts/evaluate.py`]].
- Behaviour of the pipeline at the planned sample size: [[RESULT: discrimination and
  calibration on the synthetic cohort]].
- State plainly: the simulation's parameters are assumptions, and the movement-risk link was
  chosen by the authors.

## Discussion (items 25–27)
- Interpretation in light of Bahr (2016): what repeated, dynamic, uncertainty-aware prediction
  changed and what it did not.
- Clinical meaning: decision support for review, not diagnosis; bio-banding and load-management
  implications stated as hypotheses.
- Generalisability to and beyond the Indian cohort.
- Limitations: sample size and number of events; growth-timing uncertainty is not well
  calibrated for children several years before PHV; 2D projection angles; load proxies not
  validated against force plates; single region; one season of follow-up.
- Usability: input data quality needed, who would use the output and how, what must be shown
  alongside a risk estimate.

## Conclusion
[[RESULT: one or two sentences, written after the results exist]].
