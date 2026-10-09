# SOP 01 — Anthropometry (DRAFT)

> DRAFT for team review. TODO(human): confirm equipment models, measurer training, and TEM limits.

**Frequency:** every 6–8 weeks, same time of day where possible, before training.

## Measures (each taken twice; a third if the pair differs by more than the agreed limit)
| Measure | Unit | Method |
|---|---|---|
| Stature | cm (0.1) | Stadiometer, barefoot, Frankfort plane, stretch stature |
| Sitting height | cm (0.1) | Seated on a box of known height against the stadiometer; subtract box height |
| Body mass | kg (0.1) | Calibrated digital scale, light clothing |

Derived: leg length = stature − sitting height.

## Recording
- Enter both raw duplicates (never only the mean), date, and `measurer_id`.
- Technical error of measurement (TEM) is computed and logged per measurer per visit.
- Retake limit for duplicate disagreement: TODO(human) (set per ISAK-style convention).
- Implausible values are flagged by the QC module, never silently corrected.

## Once at enrolment
Sex, date of birth (stored at month precision), sport, playing position, dominant limb, injury
history, and self-reported heights of both biological parents (for Khamis–Roche; self-report
correction applied in analysis).
