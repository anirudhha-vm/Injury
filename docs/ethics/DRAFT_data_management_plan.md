# DRAFT — Data management and privacy plan

> **DRAFT — NOT APPROVED. For ethics committee and legal review.** Describes what the code
> enforces today and what still needs a human decision.

## Data categories
| Data | Identifiability | Storage | Enforced by |
|---|---|---|---|
| Raw video | Directly identifying (minor) | Local encrypted disk only, `data/raw_video_LOCAL_ONLY/` | `.gitignore`, pre-commit hook, CI check, `tests/test_repo_hygiene.py` |
| Keypoints and derived features | Pseudonymous | Local research database / parquet | Pseudonymous UUIDs only |
| Anthropometry, exposure, injuries | Pseudonymous, health data | Local research database | DOB at month precision |
| ID↔identity map | Identifying | Outside the repository, access-restricted | TODO(human): custodian and location |
| Synthetic data | None | Anywhere | `synthetic=True` metadata |

## Cloud services
No personal data of participants is sent to any cloud service by this codebase. A Supabase
project exists for the team; until the ethics committee explicitly approves otherwise it may
hold **synthetic data only**. TODO(human): decide whether pseudonymous non-video data may be
held in a cloud database, in which region, and under what agreement.

## Access, retention, deletion
- Access list and roles: TODO(human).
- Retention period for video and for pseudonymous data: TODO(human).
- Withdrawal: all data for the athlete UUID is deleted on request; procedure TODO(human).
- Breach response and grievance contact (DPDP Act 2023): TODO(human).

## Model outputs
Research outputs are decision support, never diagnoses, and are not returned to academies during
the observation period.
