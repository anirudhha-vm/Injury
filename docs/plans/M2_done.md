# M2 status — updated 2026-10-10: coverage criterion met near PHV, NOT met overall

All numbers are from **synthetic data** and are not findings about real athletes. Final run:
`simgrowth-16fbcf9f034a`, seed 20261009, 300 athletes, commit `bb26d5a` (clean tree), produced
by `scripts/sim_study_growth.py`. The earlier run `simgrowth-db8d25ba9b6f` is superseded.

## Acceptance criteria (brief §6.2)
| Criterion | First model | Final model |
|---|---|---|
| 90% intervals for APHV cover the truth 85–95% of the time | ❌ 74.3% | ❌ **80.7% overall** (girls 87.1% ✅, boys 73.8% ❌) |
| APHV MAE lower than the Moore point estimate | ✅ 0.43 vs 0.72 y | ✅ 0.43 vs 0.72 y, in every sex and stratum |
| Incremental update works | ✅ | ✅ unit-tested |
| Figures watermarked SYNTHETIC | ✅ | ✅ |
| Sampler converged | ✅ R-hat ≤ 1.01 | ⚠️ weaker: R-hat up to 1.04, minimum bulk ESS 141, no divergences; 80–100 minutes per sex |

## Where the final model is and is not calibrated
| True years from PHV at mid follow-up | n | Coverage | Twin MAE (y) | Moore MAE (y) | Twin bias (y) | Interval width (y) |
|---|---|---|---|---|---|---|
| more than 3 before | 49 | **0.53** | 0.99 | 1.09 | −0.93 | 1.88 |
| 3 to 1 before | 65 | 0.88 | 0.33 | 0.46 | −0.12 | 1.41 |
| within 1 | 73 | 0.89 | 0.27 | 0.57 | −0.09 | 1.10 |
| 1 to 3 after | 64 | 0.88 | 0.18 | 0.58 | −0.08 | 0.73 |
| more than 3 after | 49 | 0.78 | 0.54 | 1.11 | +0.29 | 1.71 |

**Within 3 years of PHV (202 of 300 athletes) coverage is 88–89%, inside the target.** The
shortfall is concentrated in children more than 3 years before PHV, whose age at PHV is
estimated about 0.9 years too early with intervals that do not reflect it.

## What changed between the two models
1. A per-athlete velocity deviation, so that mismatch between an athlete's curve and the shared
   shape widens the interval instead of shifting timing. The model estimates its size
   (about 0.44 cm/year for girls). This is what moved coverage from 74% to 81%.
2. A learned cohort-level offset and width for the Moore prior (width learned at about 0.72 y).

## What was tried for the remaining gap and did not work
- **A cohort-level prior on timing** (athletes with little information shrink toward the
  cohort's typical age at PHV): tested against the saved fit with plug-in values; coverage for
  the far-before-PHV group stayed at about 0.55. Left in the code, switched off.
- **Wider spline knots** (flexible from 9 instead of 6 years before PHV, ten knots): a fast
  optimisation-based fit collapses a variance parameter and gives unusable intervals. The
  proper run was then started from clean commit `b6db852` and **stopped after 7.7 hours of
  wall-clock time with none of its eight chains finished** (the machine was shared with other
  jobs for part of that time; the two earlier models took 40 and 90 minutes). The extra knots
  sit where the cohort has little data, so the curve there is barely determined and the sampler
  crawls. No coverage figure exists for this variant. It is not a practical fix as specified.

## Interpretation
For a child several years before the growth spurt, one season of steady childhood growth
contains almost no information about when the spurt will come. The estimate then rests on the
prior and on how well the shared curve describes childhood growth, and both are imperfect here.
This matches the literature's warning that maturity prediction degrades with distance from PHV.

## Consequences for later milestones
- Growth-state uncertainty passed to the risk model (M6) is trustworthy for athletes within
  about 3 years of PHV and **understated** for younger children. Any calibration claim for the
  risk model has to be made with that stratification.
- A practical safeguard to consider: flag growth states with estimated maturity offset below −3
  years as low-information.

## Open, for a team decision
- A wider-knot model would need a stronger prior on the curve's childhood section (or a
  parametric childhood term) to be estimable; untested.
- Model the Moore prior's error as growing with distance from PHV.
- Or accept and pre-register the restriction: uncertainty claims apply within 3 years of PHV.

## Also not done in M2
- Sitting-height and mass models (no leg-length or mass velocity in the growth state yet; the
  athlete-week table uses finite-difference velocities instead).
- Prior predictive checks for the weakly-informative priors.
- H1 on real athletes.

## Caveat on "beats Moore"
Moore's equation was not derived on this synthetic population, so beating it here is weak
evidence. It is a pipeline check, not support for H1.
