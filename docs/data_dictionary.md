# Data dictionary

Source of truth: `src/youthtwin/io/schema.py` (validation) and `data/schema/schema.sql` (DDL);
`tests/test_schema.py` keeps them in step. Lengths in **cm**, mass in **kg**. Dates are
ISO-8601. Every table also has `synthetic` (boolean; `true` only for simulated rows — synthetic
and real rows are never mixed in one database).

## athletes
| Field | Type | Unit / values | Notes |
|---|---|---|---|
| `id` | UUID | — | Pseudonymous. The identity map lives outside the repository. |
| `sex` | text | `male`, `female` | |
| `dob_month` | text | `YYYY-MM` | Month precision only; a full date of birth is rejected. |
| `sport` | text | — | TODO(human): controlled list once sports are chosen. |
| `academy_id` | text | — | Pseudonymous academy code. |
| `dominant_limb` | text | `left`, `right` | |
| `mid_parent_height` | number, optional | cm | Mean of parents' heights (self-report; correction applied in analysis). |

## anthropometrics — one row per visit (every 6–8 weeks)
| Field | Type | Unit | Notes |
|---|---|---|---|
| `athlete_id` | UUID | — | |
| `date` | date | — | |
| `stature_1`, `stature_2` | number | cm | Both raw duplicates are stored, never only the mean. |
| `sitting_height_1`, `sitting_height_2` | number | cm | Box height already subtracted. |
| `mass_1`, `mass_2` | number | kg | |
| `measurer_id` | text | — | Used for technical error of measurement per measurer. |

Derived (not stored): leg length = stature − sitting height.

## screening_sessions
| Field | Type | Notes |
|---|---|---|
| `id` | UUID | |
| `athlete_id` | UUID | |
| `date` | date | |
| `camera_setup` | text | Phone models, fps, placement version. |

## trials — one row per recorded attempt
| Field | Type | Values | Notes |
|---|---|---|---|
| `id` | UUID | | |
| `session_id` | UUID | | |
| `task` | text | keys of `configs/pose/tasks.yaml` | |
| `limb` | text | `left`, `right`, `both` | Test limb; `both` for bilateral tasks. |
| `trial_no` | integer | ≥ 1 | |
| `keypoints_uri` | text | | Path to keypoints. A video path is rejected. |
| `quality` | number | 0–1 | Per-trial quality score. |

## task_features — long format
| Field | Type | Notes |
|---|---|---|
| `athlete_id`, `session_id` | UUID | |
| `task` | text | |
| `trial` | integer, optional | Empty = session-level summary (mean of trials, asymmetry). |
| `limb` | text | `left`, `right`, `both` |
| `feature` | text | e.g. `knee_flexion_at_initial_contact`; angles in degrees, velocities in deg/s. |
| `value` | number | Unit depends on `feature`. |
| `quality` | number | 0–1 |

## exposure_weekly
| Field | Type | Unit | Notes |
|---|---|---|---|
| `athlete_id` | UUID | — | |
| `week_start` | date | — | Monday. |
| `hours_training` | number | hours | |
| `hours_match` | number | hours | |
| `sessions` | integer | count | |

## ostrc_weekly
| Field | Type | Values | Notes |
|---|---|---|---|
| `athlete_id` | UUID | | |
| `week_start` | date | | Monday. |
| `q1`–`q4` | integer | 0–25 each | OSTRC item scores. TODO(human): confirm scoring against Clarsen et al. 2020. |
| `severity` | integer | 0–100 | Must equal `q1 + q2 + q3 + q4`. |

## injuries — the only source of outcome labels
| Field | Type | Values | Notes |
|---|---|---|---|
| `id` | UUID | | |
| `athlete_id` | UUID | | |
| `onset_date` | date | | |
| `osiics_code` | text | OSIICS | Required. |
| `body_area`, `tissue`, `mechanism` | text | | |
| `time_loss_days` | integer | ≥ 0 | |
| `outcome_group` | text | `growth_related`, `non_contact_soft_tissue`, `other_non_contact`, `contact`, `other` | "All non-contact" is the union of the first three, formed at analysis time. |
| `diagnosed_by` | text | | Required: labels come only from a physiotherapist's diagnosis. |

## growth_posteriors
| Field | Type | Notes |
|---|---|---|
| `athlete_id` | UUID | |
| `fit_date` | date | Data cut-off of the fit. |
| `uri` | text | Path to the saved posterior. |

## Synthetic-only truth tables (never model inputs)
Written under `data/synthetic/<cohort>/truth/`.

| Table | Contents |
|---|---|
| `athlete_truth` | Baseline age, true age at PHV, Preece–Baines parameters, movement traits. |
| `athlete_weeks_truth` | Per athlete-week: age, true maturity offset, true height and leg-length velocity, latent load proxy, exposure-spike indicator, risk-set membership, true weekly hazard per injury type. |
