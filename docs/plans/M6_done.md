# M6 status — 2026-10-10: machinery built and running; risk intervals NOT calibrated

All numbers are from **synthetic data** (run `ablation-b77476353614`, commit `1d8d909`, seed 0,
300 athletes, 5 athlete-grouped folds, ensembles of 3, at most 12 epochs, growth posteriors from
`simgrowth-16fbcf9f034a`; 143 minutes). They are not findings about real athletes.

## Acceptance criteria (brief §7)
| Criterion | Result |
|---|---|
| Ablation grid runs end to end on synthetic data | ✅ All nine configurations; one tidy table with a run ID per configuration |
| Each conditioning mode trains | ✅ For both backbones (tested: loss falls for `none`, `concat_input`, `late_fusion`, `film`) |
| Ensemble + posterior-draw uncertainty produces calibrated intervals on synthetic data | ❌ **Not met.** The 90% risk intervals contain the true 4-week risk for 27% of athlete-weeks |
| `make test lint` green | ✅ 458 passed, 2 deselected |

## Why the intervals are too narrow
The interval combines two things: uncertainty about the athlete's growth timing, and
disagreement between three networks trained from different seeds. Its mean width is 0.004 on a
risk of about 0.014. It leaves out the largest source of error by far: **the model itself is
poorly determined** by about 55 injuries. Three networks trained on the same data agree with
each other and are wrong together. So the interval answers "how much does the estimate move if
growth timing is uncertain", not "how far is this from the true risk". It should be described
that way and never presented as a confidence interval for risk.

What would be needed for an interval that covers the true risk: resampling athletes when
training (bootstrap ensembles), or a calibration step on held-out athletes. Neither is built.

## Ablation table (growth-related injury; out-of-fold; no confidence intervals)
| # | Configuration | AUC | Correlation with true risk |
|---|---|---|---|
| 1 | age only | 0.52 | 0.15 |
| 2 | maturity offset + growth velocity | 0.56 | 0.35 |
| 3 | screening features only | 0.50 | 0.16 |
| 4 | movement encoder, no growth | 0.52 | 0.16 |
| 5 | late fusion | 0.62 | 0.40 |
| 6 | input concatenation (v2 design) | 0.59 | 0.45 |
| 7 | FiLM (proposed) | 0.60 | 0.51 |
| 8 | FiLM + mechanistic proxies | 0.60 | 0.51 |
| 9 | #8 + uncertainty propagation | 0.60 | 0.51 |

14,058 athlete-weeks, 199 of them with an injury in the following 4 weeks (from about 55
injuries). Brier scores are indistinguishable (0.0139–0.0140) and mean predicted risk matches
prevalence in every configuration.

## How to read the table
- **Giving the model the growth state helps here**: configurations 5–7 are above 4 on both
  columns. That is expected, because the simulation makes growth velocity a cause of injury.
- **The ordering among late fusion, concatenation and FiLM must not be interpreted.** AUC puts
  late fusion first; correlation with the true risk puts FiLM first. With about 55 injuries, one
  seed and no intervals, differences of a few hundredths are noise. This is not evidence for or
  against hypothesis H3.
- Any advantage of conditioning partly reflects how I built the simulation (risk depends on
  growth velocity times a movement-derived load).
- Configurations 8 and 7 are identical because the simplified synthetic cohort has no side-view
  features to compute the mechanistic proxies from; the table says so in its note column.
- Configuration 9 changes predictions only slightly: averaging over growth draws barely moves a
  nearly linear model.

## Not done
- Confidence intervals and paired comparisons for the ablation: the runner saves metrics, not
  the out-of-fold predictions that the athlete-level bootstrap needs.
- The ST-GCN backbone was tested but not run through the full grid.
- Mechanistic proxies in a fitted model.
- Contrastive pre-training (only masked reconstruction is implemented); expert movement-quality
  scores as auxiliary targets; split-conformal intervals.
- The full grid takes over two hours on CPU for one outcome.

## Consequence for the papers
Consistent with `docs/RESEARCH_STRATEGY.md`: the conditioned encoder is exploratory. The
uncertainty band shown on the dashboard reflects growth-timing uncertainty only and is labelled
as such.
