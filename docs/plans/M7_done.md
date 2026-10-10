# M7 done — 2026-10-10: evaluation suite, explanations and figure generation

## Acceptance criteria
| Criterion | Result |
|---|---|
| Every figure and table regenerated from one command | ✅ `make paper-figs` rebuilds everything under `paper/figures` and `paper/tables` from the latest runs; `make pipeline` runs cohort → evaluation → figures |
| `check_no_hardcoded_results.py` passes | ✅ on the repository and on a generated paper folder; a table without provenance is caught (tested) |
| `make test lint` green | ✅ 402 passed, 2 deselected; `mypy --strict` clean on 80 files |

## What was built
- Metrics: AUC at the 4-week horizon, PR-AUC, Brier score, calibration intercept and slope, ICI,
  ECE, sensitivity/specificity at pre-specified thresholds.
- Decision-curve analysis with treat-all and treat-none references.
- Athlete-level cluster bootstrap for intervals and for paired model differences.
- Subgroup tables by sex, maturity stratum, sport and age band.
- Explanations: SHAP on the fitted tabular models (additive, mapped back to inputs), integrated
  gradients for networks, what-if analysis labelled as sensitivity.
- `scripts/evaluate.py` and `scripts/make_figures.py`, with provenance sidecars on every output.

## Verification
Each metric has a hand-worked value and, where scikit-learn implements the same quantity, a
comparison with it. Calibration measures recover known miscalibration with the right sign.
The bootstrap interval for a known quantity covers it at about the nominal rate.

## What the synthetic run shows (SYNTHETIC, indicative: 100 bootstrap resamples)
For growth-related injury on the default cohort, all five tabular models have out-of-fold AUC
between 0.50 and 0.56 with intervals spanning 0.5, calibration slopes near 0, and no net benefit
over the reference strategies. The simulated effects are real, so this reflects too few events
(about 55 injuries among 300 athletes) rather than an absent signal. **Implication for the real
study: the planned sample size needs checking against the expected number of events.**

## Not done
- Risk thresholds are not pre-specified, so no sensitivity/specificity is reported.
- Hyperparameters are not tuned in inner folds (the nested splits exist; the tuning loop does not).
- The neural models are evaluated by `scripts/ablation.py` with AUC and Brier only; they are not
  yet routed through the full evaluation tables.
- Integrated gradients is implemented and tested on small networks but not yet run on a trained
  encoder to produce a figure.
- Time-dependent AUC at horizons other than the pre-specified one; SPM curve comparison.
- `scripts/sample_size.py` (Riley criteria) is still to be written.

## Next: M8
Streamlit research dashboard and recommendation rules with an evidence library.
