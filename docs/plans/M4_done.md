# M4 done — 2026-10-10: agreement statistics implemented and verified; no lab data yet

## Acceptance criteria
| Criterion | Result |
|---|---|
| Agreement metrics verified on synthetic paired data with known bias and noise | ✅ Bias, SD, limits of agreement and RMSE recovered; limits contain 95% of differences; the bias confidence interval has nominal coverage over 400 repeats |
| Bland–Altman and ICC match reference implementations | ✅ ICC(2,1) and ICC(3,1) equal `pingouin.intraclass_corr` to 1e-9 on three datasets, with matching confidence intervals; Bland–Altman checked against hand-worked values and an independent SciPy recomputation |
| `make test lint` green | ✅ 265 passed, 2 deselected; `mypy --strict` clean on 50 files |

## What was built
- `validation/agreement.py`: RMSE, MAE, Bland–Altman (with confidence intervals for the bias
  and both limits, and proportional bias), ICC(2,1), ICC(3,1), and a stratified agreement table
  that keeps thin groups with a note instead of dropping them.
- `validation/mocap_io.py`: TRC reader and writer, optional C3D reader, time alignment of a
  mocap curve to a markerless curve by cross-correlation.
- `validation/plots.py`: Bland–Altman grid, RMSE by maturity stratum, curve overlay.
- `scripts/validate_markerless.py`: paired table → agreement tables and figures (run once on
  made-up pairs to confirm it executes end to end).

## Not done, and why
- **No real validation.** There is no lab subset yet. `TODO(human)`: mocap lab, system, marker
  set, and the joint-angle definitions the lab exports.
- **Marker trajectories → joint angles is not implemented.** It depends on the lab's marker
  model. The module compares angle or feature values already computed on each side.
- **C3D reading is unexecuted** (optional `ezc3d` not installed; no file to read).
- **SPM curve comparison** (optional in the brief) is not implemented; only a mean ± SD overlay.
- The 5° RMSE acceptance limit is the brief's example and is still provisional.

## A point for the paper
The proportional-bias slope in a Bland–Altman analysis is not zero when the two methods simply
have different error variances; a test documents the size of this effect. Report it alongside
each method's spread rather than as evidence of a scale error on its own.

## Next: M5
Person-period (athlete-week) table, exposure features, mechanistic load proxies, baseline and
hazard models. M5 consumes the growth state from M2, so M2's refit result matters there.
