# Pre-registration — YouthTwin (TEMPLATE, not yet registered)

> Must be completed, reviewed by the team, and registered (e.g. OSF) **before any real outcome
> data are touched** (non-negotiable N8). Numbers in this file follow the tagging rules in
> `CLAUDE.md`: design values are `[[SPEC: …]]` and must match `configs/`; literature values are
> `[[CITED: …]]`. Design decisions and their evidence: `docs/RESEARCH_STRATEGY.md`.

- Registration ID / URL: TODO(human)
- Date registered: TODO(human)
- Target journals: TODO(human) (suggestions in `docs/RESEARCH_STRATEGY.md`)

## Study design
Prospective cohort of youth athletes aged 8–18 y, followed for at least one season, with
anthropometry and two-camera smartphone screening every 6–8 weeks and weekly injury and exposure
surveillance. TODO(human): academies, sports, number of seasons, final sample size.

## Papers and what each may claim
- **Paper A — validity.** Agreement of smartphone markerless kinematics with marker-based
  motion capture, by maturity stage. Confirmatory.
- **Paper B1 — growth twin.** Accuracy and calibration of posterior age at PHV against maturity
  equations and against observed PHV. Confirmatory.
- **Paper B2 — risk prediction.** A small confirmatory model whose size is set by the number of
  events; all larger models are exploratory (see "Model complexity").

## Outcomes
- **Primary outcome (B2):** growth-related injury (apophysitis, apophyseal avulsion, physeal
  injury, bone stress), physio-diagnosed and OSIICS-coded, with onset in the following 4 weeks.
- **Secondary outcomes (modelled separately, cause-specific):** non-contact soft-tissue injury;
  all non-contact injury.
- All physiotherapist-confirmed problems are recorded, including those without time loss;
  time-loss injuries are analysed as a sensitivity analysis.
- Outcomes are never derived from model inputs, thresholds, or the maturity score (N1, N2).

## Primary metric
Time-dependent AUC (cumulative/dynamic) at 4 weeks with athlete-level cluster-bootstrap
intervals; calibration intercept and slope and decision-curve net benefit are reported with
equal prominence.

## Sample size and model complexity
- Planning figure for academy football, a few athletes in a hundred per season:
  [[CITED: 603 time-loss apophyseal injuries in 16,024 player-seasons, Br J Sports Med 2025;59:1663]].
- Model complexity is set by the criteria of Riley et al. (`scripts/sample_size.py`) applied to
  the **achieved** number of athletes and athletes with an event, before any model is fitted to
  outcome data.
- **Confirmatory model:** logistic regression using the leading predictors of the pre-specified
  ordered list in `configs/models/hazard.yaml` (`confirmatory.candidate_predictors_in_order`),
  one parameter each, as many as the criteria support and at most
  [[SPEC: max_parameters = 5]]. If not even one parameter is supported, the outcome is reported
  descriptively and no confirmatory model is fitted.
- **Exploratory models:** penalised hazard with splines, boosted hazard, conditioned encoder and
  the nine-configuration ablation. Reported with intervals and labelled exploratory unless the
  cohort reaches the size the criteria require for them.

## Hypotheses
- **H1 (B1, confirmatory):** The growth-twin posterior age at PHV has lower error against
  observed age at PHV than the Moore and Mirwald point estimates (athletes who pass PHV during
  follow-up; simulation study reported separately).
- **H1b (B1, confirmatory):** Posterior intervals for age at PHV are calibrated for athletes
  within [[SPEC: 3 years]] of PHV. Athletes further from PHV are reported separately and are
  not covered by this claim (limitation established in simulation; `docs/plans/M2_done.md`).
- **H2 (B2, confirmatory):** For growth-related injury, the confirmatory model has higher
  time-dependent AUC and net benefit than an age-only model.
- **H3 (B2, exploratory):** FiLM conditioning outperforms input concatenation and late fusion.
- **H4 (B2, exploratory):** Propagating growth uncertainty improves calibration (slope closer to
  one; lower ICI).
- **H5 (A, confirmatory):** Markerless sagittal knee and hip angles at task events agree with
  marker-based capture within [[SPEC: RMSE < 5 deg]], and agreement is not worse in pre-PHV
  children. Secondary endpoint: frontal-plane knee projection angle, same limit.

## Pre-specified ablation set (exploratory)
The nine configurations in `configs/eval/ablations.yaml`: age only; maturity offset plus growth
velocity; screening features only; movement encoder only; late fusion; input concatenation;
FiLM; FiLM plus mechanistic proxies; the previous plus uncertainty propagation.

## Analysis plan
- Splits: grouped by athlete, nested for tuning, plus a temporal hold-out of the last
  [[SPEC: 25% of follow-up time]]; leave-one-academy-out if three or more academies.
- Preprocessing statistics are fitted on training folds only.
- Model comparison: athlete-level cluster bootstrap of paired metric differences
  ([[SPEC: n_resamples = 1000]]).
- **Risk thresholds are elicited before data lock.** Each academy physiotherapist is asked how
  many unnecessary reviews are acceptable to identify one athlete who would go on to be injured.
  The answers define the threshold range for decision curves and the thresholds at which
  sensitivity and specificity are reported. No threshold is chosen after seeing results.
  TODO(human): run the elicitation and record the answers here.
- Missing data: TODO (decide and document here before registration). The confirmatory model
  uses median imputation fitted on training folds, without missing-value indicators.
- Maturity equations: coefficients are verified against the original papers before any
  real-data run (`make gate`).

## Deviations
Any deviation from this document after registration is logged here with date and reason.
