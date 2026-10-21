# Manuscript outline — single paper (replaces the A / B1 / B2 split for submission)

**Working title:** A maturity-aware digital twin for growth-related injury risk in youth
athletes: validation on real growth and injury data and a calibrated simulation study

> Structure only. Every number comes from `paper/results_draft.md` (`make paper-results`),
> which is generated from runs; nothing is typed here. The three earlier outlines
> (`paperA_outline.md`, `paperB1_outline.md`, `paperB_outline.md`) remain as the fallback if
> the team splits the work. Target journal: TODO(human) (candidates: *Br J Sports Med*,
> *Scand J Med Sci Sports*, *Med Sci Sports Exerc*).

## What is real and what is simulated

| Layer | Evidence in this paper | Data |
|---|---|---|
| Growth twin: age at PHV from one season of measurements | **Real** | Berkeley Child Guidance Study, 136 children (public, GPL) |
| Risk-model evaluation discipline (grouped splits, temporal hold-out, calibration, leakage) | **Real**, adult runners | Lövdal et al. 2021 runners, 74 athletes (public, CC0) |
| Smartphone versus motion capture, by maturity | **Simulation**, calibrated to a published smartphone error | none public for children |
| Growth-related injury risk with the growth state | **Simulation**, calibrated to published incidence and growth-velocity effect | none public |
| Sample size needed | **Simulation** + Riley criteria | — |

The abstract, results and discussion must keep this split visible. Simulated results are reported
only under "Simulation study" (working rule 4) and never as findings about athletes.

## Abstract (structured)
Background; Methods (real-data validation of the growth twin; real-data check of the evaluation
pipeline; simulation study calibrated to literature); Results (`[[RESULT: from results draft]]`);
Conclusions (what is established on real data; what the simulation predicts; what a prospective
cohort must measure).

## 1. Introduction
1. Growth-related injuries (apophysitis, physeal stress) cluster around the adolescent growth
   spurt; existing prediction models are at high risk of bias and almost never report
   calibration (Bullock et al. 2022).
2. Maturity is usually a point estimate from an equation (Mirwald 2002, Moore 2015) that is
   least accurate far from PHV and carries no uncertainty.
3. Smartphone pose estimation could make movement screening scalable, but its validity in
   children of different maturity is unknown.
4. Gap and contribution: an uncertainty-aware growth twin validated on real longitudinal
   growth data; an evaluation discipline shown on real injury data; and a calibrated simulation
   that sizes the prospective study and states what each component can deliver.

## 2. Methods
- Growth twin (model, priors, marginalisation over athlete timing) — `paper/methods_draft.md`.
- Real growth data: Berkeley; reference age at PHV from the full record (Preece–Baines fit,
  smoothing-spline check, exclusion rules); one-season windows; comparators Moore 2015 and
  Mirwald 2002 (boys; girls' coefficients unverified, TODO(human) gate).
- Real injury data: runners; three split schemes; models; athlete-bootstrap intervals.
- Simulation study: cohort generator and its calibration (Indian timing; Berkeley curve
  shapes; [[CITED: 3.8%]] growth-related incidence, BJSM 2025; growth-velocity effect), synthetic validity study
  through the real pose pipeline, risk models (confirmatory model sized by Riley criteria;
  exploratory penalised, boosted and conditioned neural models), ablation, cohort-size study.
- Reporting: TRIPOD+AI (`paper/tripod_ai_checklist.md`); pre-registration
  (`paper/preregistration.md`).

## 3. Results — Part 1, real data
- Growth twin vs Moore and Mirwald on Berkeley: error, bias, interval coverage, by years from
  PHV; sensitivity to the reference rule. (Results draft, first section; figure
  `real_growth_twin_aphv.png`.)
- Evaluation on real injuries: AUC and calibration under runner-grouped, temporal and leaky
  splits; inflation caused by leakage. (Second section; figure `real_risk_calibration.png`.)

## 4. Results — Part 2, simulation study (SYNTHETIC)
- Smartphone vs motion capture by maturity group; precision of a 36-participant study.
- Growth twin in the simulated cohort of [[SPEC: 1000]] athletes.
- Risk models: confirmatory and exploratory; calibration; decision curves.
- Ablation of growth conditioning with paired intervals.
- Performance against cohort size: the number of injured athletes a real study needs.

## 5. Discussion
- What is established on real data, and its limits (historical US cohort for growth; adult
  runners for injury).
- What the simulation implies, and which of its inputs are assumptions (load proxy,
  interaction, keypoint-error model for valgus).
- Design of the prospective cohort that would test the full twin (size from the cohort-size study, measurements,
  maturity-stratified validation from the validity simulation).
- Decision support, not diagnosis (N9).

## Data and code availability
Code: this repository, commit in each run record. Data: Berkeley via the `sitar` R package;
runners via DataverseNL doi:10.34894/UWU9PV; synthetic data regenerated from seeds
(`make fetch-data`, `make real-growth`, `make real-risk`, simulation targets in `Makefile`).
