# Pre-registration — YouthTwin (TEMPLATE, not yet registered)

> Must be completed, reviewed by the team, and registered (e.g. OSF) **before any real outcome
> data are touched** (non-negotiable N8). Numbers in this file follow the tagging rules in
> `CLAUDE.md`: design values are `[[SPEC: …]]` and must match `configs/`.

- Registration ID / URL: TODO(human)
- Date registered: TODO(human)
- Target journal: TODO(human)

## Study design
Prospective cohort of youth athletes aged 8–18 y from 2–4 academies, followed for at least one
season, with anthropometry and two-camera smartphone screening every 6–8 weeks and weekly injury
and exposure surveillance. TODO(human): academies, sports, final sample size.

## Outcomes
- **Primary outcome:** growth-related injury (apophysitis, apophyseal avulsion, physeal injury,
  bone stress), physio-diagnosed and OSIICS-coded, occurring in the next 4 weeks.
- **Secondary outcomes (modelled separately, cause-specific):** non-contact soft-tissue injury;
  all non-contact injury.
- Outcomes are never derived from model inputs, thresholds, or the maturity score (N1, N2).

## Primary metric
Time-dependent AUC (cumulative/dynamic) at 4 weeks, with athlete-level cluster-bootstrap
confidence intervals; calibration and decision-curve net benefit reported alongside.

## Hypotheses
- **H1:** The growth-twin posterior APHV has lower error against observed APHV than Moore and
  Mirwald point estimates (simulation study, plus athletes who pass PHV during follow-up).
- **H2:** For growth-related injury, the full model improves time-dependent AUC and net benefit
  over (a) age only, (b) maturity offset plus growth velocity, (c) movement only.
- **H3:** FiLM conditioning outperforms input concatenation and late fusion (primary ablation).
- **H4:** Propagating growth uncertainty improves calibration (slope closer to one; lower ECE/ICI).
- **H5 (Paper A):** Markerless sagittal knee and hip angles at task events agree with marker-based
  mocap within a pre-specified limit ([[SPEC: RMSE < 5 deg — provisional, TODO(human) confirm]]),
  with agreement not worse in pre-PHV children.

## Pre-specified ablation set
Exactly the nine configurations in `configs/eval/ablations.yaml`: age only; maturity offset plus
growth velocity; screening features only; movement encoder only; late fusion; input
concatenation; FiLM (proposed); FiLM plus mechanistic proxies; the previous plus uncertainty
propagation.

## Analysis plan
- Splits: grouped by athlete, nested for tuning, plus a temporal hold-out of the last
  [[SPEC: 25% of follow-up time]]; leave-one-academy-out if three or more academies.
- Preprocessing statistics are fit on training folds only.
- Model comparison: athlete-level cluster bootstrap of metric differences
  ([[SPEC: n_resamples = 1000]]).
- Risk thresholds for sensitivity/specificity and decision curves: TODO(human) pre-specify.
- Missing data handling: TODO (decide in M5; document here before registration).
- Sample size: Riley et al. criteria via `scripts/sample_size.py` once incidence is estimated.

## Deviations
Any deviation from this document after registration is logged here with date and reason.
