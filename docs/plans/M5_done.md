# M5 done — 2026-10-10: athlete-week table and tabular hazard models, verified on synthetic data

## Acceptance criteria
| Criterion | Result |
|---|---|
| Leakage tests pass | ✅ `test_no_label_leakage.py`: predictors exclude labels, IDs and simulation truth; label-defining columns are refused as inputs; rewriting everything after a cut-off date leaves earlier athlete-weeks' inputs unchanged; with no diagnosed injuries there are no positive labels |
| Hazard model recovers simulated effect directions | ✅ with one caveat (below). Growth-related risk rises with leg-length velocity and the load proxy; soft-tissue risk rises with maturity offset; the load proxy acts on growth-related risk more than on soft-tissue risk. Both the penalised logistic and the boosted model |
| Split overlap = 0 | ✅ Athlete-grouped folds, nested inner folds, leave-one-academy-out all checked; a row-level split is caught |
| Scalers fitted on training folds only | ✅ `test_scaler_fit_scope.py`: fitted statistics equal training-fold statistics; corrupting held-out rows changes nothing |
| `make test lint` green | ✅ 321 passed, 2 deselected; `mypy --strict` clean on 59 files |

## Caveat: no exposure direction could be checked
In the simulation, training hours raise injury risk in the **same** week and hours are drawn
independently each week. This week's hours therefore carry no information about injuries in the
following four weeks, and neither model finds an exposure effect. That is the correct answer
for this simulation, not a recovered effect. To test exposure handling the simulation needs
load that persists over weeks, or a hazard that depends on accumulated load (a change for the
M6 simulation study).

## Things found along the way
- **Collinearity.** Age and Moore maturity offset have variance inflation factors of about 10
  in the synthetic table, and height velocity about 9. With both members of such a pair in a
  model, the sign of either one alone is not determined; the first direction test failed for
  that reason, not because of a pipeline fault. The direction tests use one representative per
  correlated group, and `eval/collinearity.py` warns above a VIF of 10.
- **A chance association.** With one seed, athletes with a high load trait had fewer soft-tissue
  injuries although no such effect was simulated. Across 12 seeds the association averages
  about zero (correlation −0.013, standard error 0.008). The test was rewritten to assert only
  what the simulation actually contains, rather than moved to a luckier seed.
- **Generator bug fixed.** For a rare pair of growth parameters the childhood velocity at the
  edge of the search window was marginally higher than the pubertal peak, and the peak search
  raised an error (seen with other seeds at 1,200 athletes). It now takes the highest interior
  local maximum; a regression test covers the case. The default cohort is unaffected.
- **Solver randomness.** The logistic solver shuffles data; without a fixed seed two fits of the
  same model differed in the seventh decimal. The model now takes a seed.

## Not done
- Growth-twin features in the hazard models on the full cohort: the table builder accepts the
  posteriors and is tested on a small case, but the comparison of models with and without twin
  features waits for the M2 refit and belongs to the ablation in M6–M7.
- Mechanistic proxies are implemented and tested, but the simplified synthetic cohort has no
  side-view features to compute them from, so they are not yet in any fitted model.
- Hyperparameter tuning in inner folds, calibration, decision curves, bootstrap intervals (M7).
- No discrimination figures are reported here: the only numbers are synthetic, and the tests
  assert only that held-out AUC exceeds 0.55.

## Next: M6
Encoder, conditioning modes, self-supervised pre-training, neural hazard head, uncertainty
propagation.
