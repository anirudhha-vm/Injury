# Paper B1 outline — An uncertainty-aware growth twin for estimating age at peak height velocity from sparse single-season anthropometry in youth athletes

> Scaffold. No results exist for real athletes. Every result is a `[[RESULT: …]]` placeholder to
> be filled by a script-produced `[[RUN:<run_id>|value]]`. Design values are in
> `paper/methods_draft.md`, generated from `configs/`. Rationale for separating this paper:
> `docs/RESEARCH_STRATEGY.md`. Target journal: TODO(human).

## Introduction
1. Why maturity timing matters in youth sport: injury type, bio-banding, training load.
2. Maturity-offset equations: convenient, but biased toward the mean, least accurate far from
   PHV, and derived on samples of mainly European ancestry (Mirwald et al. 2002; Moore et al.
   2015; Malina & Kozieł 2014). Population-specific failure has been shown in Japanese youth
   (Murayama et al. 2023). Nothing equivalent exists for Indian youth athletes: TODO(human)
   confirm by systematic search.
3. Growth-curve models (SITAR; Cole et al. 2010) need longitudinal series spanning the growth
   spurt, which a season of academy measurements does not provide.
4. Gap: a model that works with sparse, short series, says how uncertain it is, corrects a
   maturity equation for the cohort at hand, and can be updated when a new measurement arrives.
5. Objectives and pre-registered hypotheses H1 and H1b.

## Methods
| Section | Source |
|---|---|
| Participants, measurement protocol, technical error of measurement | `docs/protocol/01_anthropometry_sop.md`; TODO(human) |
| Maturity equations and their verification | methods draft: maturity estimation |
| Growth twin: model, priors, integration over athlete effects, sampler | methods draft: growth twin |
| Definition of age at PHV as the age of maximum velocity of the fitted curve | methods draft: growth twin |
| Incremental update with a new measurement | `src/youthtwin/growth/sitar_bayes.py` |
| Reference standard: observed PHV in athletes who pass it during follow-up | TODO(human): define the criterion and minimum series length |
| Performance: error, bias, interval coverage, by distance from PHV and by sex | `src/youthtwin/growth/evaluation.py` |
| Simulation study design | methods draft: simulation study |

## Results
1. Cohort and measurements: [[RESULT: athletes, visits per athlete, measurement error by measurer]].
2. Cohort-level correction of the maturity equation: [[RESULT: estimated offset and spread of
   Moore's estimate in this cohort, by sex]].
3. Accuracy against observed PHV (H1): [[RESULT: mean absolute error and bias of the growth
   twin, Moore and Mirwald estimates, with intervals]].
4. Calibration of uncertainty (H1b): [[RESULT: coverage of posterior intervals within the
   pre-specified distance from PHV, and beyond it]].
5. Two-valued timing: [[RESULT: proportion of athletes whose posterior for age at PHV has two
   separated modes, by number of visits and distance from PHV; example athletes]].
6. Value of each additional measurement: [[RESULT: interval width and error as visits accumulate]].
7. Forecast check: [[RESULT: predicted versus observed stature gain over the forecast horizon]].
8. Sensitivity: [[RESULT: prior width, knot placement, with and without the velocity-deviation
   term, with and without the intensity effect]].

## Simulation study (synthetic data only)
- Recovery of a known age at PHV and interval coverage, overall and by distance from PHV:
  [[RESULT: from `scripts/sim_study_growth.py`]].
- Why direct sampling of athlete timing fails with short series, and what integrating it out
  changes: [[RESULT: convergence diagnostics of both approaches]].
- State plainly: the simulated curves are Preece–Baines curves with assumed parameters, and
  beating a maturity equation on them is a pipeline check, not evidence.

## Discussion
- What a single season can and cannot say about maturity timing; the practical meaning of a
  two-valued estimate for a coach or physiotherapist.
- Where the estimate should not be trusted: several years before PHV.
- Implications for studies that use a maturity-offset point estimate as a predictor or to form
  groups.
- Limitations: stature only so far; reference PHV available only for athletes who pass it during
  follow-up; one region; simulation assumptions.

## Conclusion
[[RESULT: one or two sentences, written after the results exist]].
