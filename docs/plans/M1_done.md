# M1 done — 2026-10-09

## Acceptance criteria
| Criterion | Result |
|---|---|
| Reference-value tests pass | ✅ All five equations checked against arithmetic worked out by hand from the brief's formulas |
| Mirwald-girls gate in place | ✅ `mirwald2002("female", …, synthetic=False)` raises; synthetic runs warn; `make gate` still fails by design |
| Synthetic cohort, 300 athletes × 12 months, reproducible from a seed | ✅ `make sim`; two separate processes produce identical files; all tables validate against the schema |
| `make test lint` green | ✅ 143 passed, 1 deselected; coverage 100%; `mypy --strict` clean on 31 files |

## What the reference tests do and do not show
They show the code computes the formulas **as written in the brief**. They do not show the
brief's coefficients match the original papers — that is still a human check (R-001).

## Synthetic cohort at the default seed (descriptive, SYNTHETIC — not a finding)
Produced by `make sim` (seed and config hash in `data/synthetic/cohort_seed20261009/_metadata.json`):
300 athletes; 2110 visits; 15600 athlete-weeks; 158 simulated injuries (55 growth-related,
76 soft-tissue, 27 other non-contact).

## Limitations to carry forward
- Every generator parameter is an assumption (`configs/sim/cohort.yaml`); none is cited yet.
- Movement is simulated as three session-level summary features only; no skeleton sequences.
- No OSTRC questionnaire rows are simulated.
- Injury effect sizes are arbitrary; the simulation study must report them as chosen, not found.
- Extreme late maturers can be implausibly short at age 8 (Preece–Baines extrapolated far
  before PHV).

## Next: M2
Bayesian SITAR growth twin and growth state, scored against the true APHV in the truth tables.
Needs PyMC (large install).
