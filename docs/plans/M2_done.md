# M2 status — 2026-10-09: implemented, one acceptance criterion NOT met

All numbers below are from **synthetic data** (simulation study run `simgrowth-db8d25ba9b6f`,
seed 20261009, 300 athletes, produced by `scripts/sim_study_growth.py`). They are not findings
about real athletes. The run was made from an uncommitted working tree (`git_commit` ends in
`-dirty`), so it must be re-run from a clean commit before any number is used in the paper.

## Acceptance criteria (brief §6.2)
| Criterion | Result |
|---|---|
| 90% posterior intervals for APHV cover the truth 85–95% of the time | ❌ **Not met: 74.3%** (girls 80.0%, boys 68.3%) |
| APHV MAE lower than the Moore point estimate | ✅ 0.43 y vs 0.72 y (true in every sex and maturity stratum) |
| Incremental update works | ✅ Unit-tested: more visits narrow the posterior and reduce error; new athletes can be added |
| Figures watermarked SYNTHETIC | ✅ Enforced in `utils/plotting.save_figure`, tested |
| Sampler converged | ✅ R-hat ≤ 1.01, minimum bulk ESS 612, no divergences, velocity peak interior in all draws |
| `make test lint` green | ✅ 178 passed (2 deselected: human gate, slow sampler test); `mypy --strict` clean |

## What the under-coverage looks like
Standardised errors (posterior mean − truth) / posterior SD have mean −0.55 and SD 1.36; a
calibrated posterior gives 0 and 1. So intervals are both shifted (APHV estimated too early)
and too narrow.

| True years from PHV at mid follow-up | n | Coverage | Twin MAE (y) | Moore MAE (y) | Twin bias (y) | Interval width (y) |
|---|---|---|---|---|---|---|
| more than 3 before | 49 | 0.65 | 0.92 | 1.09 | −0.81 | 2.17 |
| 3 to 1 before | 65 | 0.78 | 0.38 | 0.46 | −0.09 | 1.47 |
| within 1 | 73 | 0.78 | 0.30 | 0.57 | −0.14 | 1.01 |
| 1 to 3 after | 64 | 0.64 | 0.23 | 0.58 | −0.17 | 0.62 |
| more than 3 after | 49 | 0.86 | 0.47 | 1.11 | +0.08 | 1.50 |

## Likely causes (hypotheses, not yet tested)
1. **Model mismatch.** The synthetic curves are Preece–Baines with four individually varying
   shape parameters; the model allows three (size, timing, intensity) around one shared shape.
   The unit test that draws athlete effects under the *true* curve gets 84–96% coverage, so the
   machinery is calibrated when the model is right.
2. **Residual SD is estimated at the measurement-noise level (about 0.15 cm)**, so any shape
   mismatch is absorbed into timing with false confidence. Most visible 1–3 years after PHV,
   where intervals are narrowest (0.62 y) and coverage is worst.
3. **The Moore prior is biased for the youngest athletes** (Moore bias −0.50 y pre-PHV) and they
   have little growth-spurt information to correct it.

## Options for the next step (a decision for the team)
- Add a per-athlete model-discrepancy term (e.g. a random velocity deviation or a slowly varying
  residual) so shape mismatch widens intervals instead of shifting timing.
- Make the simulation harder to match on purpose and report coverage as a function of mismatch.
- Learn the timing prior SD and a prior offset per sex instead of fixing the SD at 1.0.
- Each full run takes about 40 minutes on this machine, so these were not explored in M2.

## Also not done in M2
- Sitting-height and mass models, so the growth state has no leg-length or mass velocity yet.
- Prior predictive checks for the weakly-informative priors.
- H1 on real athletes (needs real data).

## Caveat on "beats Moore"
Moore's equation was not derived on this synthetic population, so beating it here is weak
evidence. It is a pipeline check, not support for H1.
