# M12 done — SITAR benchmark for the growth twin (2026-10-28)

## Goal
Answer the reviewer question "why not just SITAR?" with the established method itself: classical
SITAR (Cole et al. 2010, R package `sitar`, `nlme` fit) evaluated on exactly the held-out
children, entry ages, visits and folds of the primary twin run.

## Design
- **As a practitioner would use it.** For each held-out child and each visit count `k`, refit
  SITAR on the fold's training children (full records, 8-18 y) plus the child's first `k`
  visits, and read the child's age at PHV off the peak of its fitted velocity curve
  (`predict(..., deriv = 1)` on a fine age grid over the training age range). No data of a
  held-out child after visit `k`, and no other held-out child, enters the fit.
- Stature only (SITAR models one measure); spline `df` from config (ASSUMPTION: 5).
- **Fit failures are results**, not dropped silently: status per row; error reported on the rows
  where SITAR succeeded, with the twin on the same rows, plus the failure rate.
- Point estimates only (SITAR gives no interval for a new child's APHV), so the same cross-fold
  conformal interval as the ML baseline is put on it, for a like-for-like interval comparison.
- Comparisons with child-level bootstrap intervals: twin − SITAR overall, by visit 1 and 6, and
  by maturity timing group; SITAR growth-spurt score AUC.

## Files
`configs/external/sitar_benchmark.yaml`, `scripts/r/sitar_refit.R`,
`src/youthtwin/external/sitar_benchmark.py`, `scripts/real_twin_sitar.py`; additions to
`scripts/real_twin_analysis.py` and `src/youthtwin/paper/twin_paper.py`; tests.

## Acceptance
- Tasks for a child at visit `k` contain only its first `k` visits; training rows contain no
  held-out child (tests).
- On data simulated from a SITAR model the refit recovers each child's APHV (R-gated test).
- Held-out rows match the twin's primary run exactly (checked at merge, as for the ML baseline).
- Outputs `synthetic=False` with provenance; the paper's values all come from the run.

## Outcome (2026-10-28)
All acceptance items met (tests `tests/test_sitar_benchmark.py`; merge check in
`scripts/real_twin_analysis.py`). Runs: `sitar-60127d624a42`, analysis
`outputs/real_twin_analysis/` (latest). Findings in CHANGELOG R-025, R-026: SITAR matches the twin's
accuracy; the twin's advantage is child-specific interval width. Added beyond the plan: conditional
coverage analysis (D-142), because conformal SITAR matched the twin's average coverage.
