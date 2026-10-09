# SOP 04 — Weekly injury and exposure surveillance (DRAFT)

> DRAFT for team review. TODO(human): who collects (physio/coach/app), language versions,
> non-response follow-up procedure, exact time-loss and substantial-problem definitions.

## Weekly, every athlete
- **OSTRC questionnaire** (four questions; severity score) — Clarsen et al. 2020.
- **Exposure:** training hours, match hours, number of sessions.

## When a problem is reported
1. Physiotherapist examines and records a diagnosis; this is the **only** source of outcome
   labels (non-negotiable N1). `diagnosed_by` is mandatory.
2. Code with **OSIICS** (Orchard et al. 2020): body area, tissue, pathology.
3. Record onset date, mechanism (contact / non-contact; sudden / gradual), time-loss days.
4. Assign the outcome group (non-negotiable N6):
   - growth-related: apophysitis (Osgood-Schlatter, Sinding-Larsen-Johansson, Sever's),
     apophyseal avulsion, physeal injury, bone stress;
   - non-contact soft-tissue;
   - all non-contact.

## Blinding
The diagnosing physiotherapist does not see model outputs, maturity estimates, or screening
features. Model predictions are not returned to academies during the observation period.

## Data
Entered under pseudonymous athlete ID only. The mapping from OSIICS codes to outcome groups is
fixed in a config before outcome data are analysed (M1 schema).
