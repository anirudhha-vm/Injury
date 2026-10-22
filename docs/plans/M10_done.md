# M10 status — 2026-10-22: real public data, calibrated simulation, IEEE manuscript

## What was done
- Growth twin on **real** growth records (Berkeley Child Guidance Study, run
  `realgrowth-5a5a4542dce2`, commit `b51fe6f`+): 125 of 136 children scored, 2 window draws.
- Risk-evaluation discipline on **real** injuries (Lövdal et al. runners, run
  `realrisk-bd8f6e8daa37`, commit `1fbe9ef`).
- Realistic simulation for the layers without public data (D-119 to D-125): Berkeley curve
  shapes, Indian timing, published incidence and growth-velocity effect, keypoint-level
  smartphone error calibrated to a published knee-flexion error.
- IEEE manuscript `paper/ieee/main.tex`; every number generated (`make paper-pdf`).

## Results (numbers live in the generated tables; summarised here in words)
- Real growth: the twin beat Moore in every band of distance from PHV; its intervals were too
  narrow (coverage well below nominal, worst far before PHV); mixing incomplete for some
  shared parameters (girls).
- Real injuries: athlete-grouped AUC modest; row-level splits inflated AUC significantly;
  calibration slopes below one under grouped splits.
- Simulation, validity: knee and hip flexion within the limit at every maturity stage; FPPA
  marginal and worse before PHV.
- Simulation, risk: at published incidence no model could be ranked at 1,000 athletes; even the
  true risk is only modestly discriminating; cohort-size study shows learned models stay far
  from that ceiling up to 2,000 athlete-seasons.
- Simulation, propagation: growth-draw + ensemble intervals covered the true risk far below
  nominal (as in M6): they express growth-timing sensitivity, not risk-model error.

## Compute reductions (stated in the manuscript)
- The 1,000-athlete risk cohort has no growth-twin features; its ablation used one network per
  configuration, six epochs, five pre-training epochs (run `ablation-cf16007fc298`).
- Propagation tested in the 300-athlete cohort with growth posteriors (`simgrowth-7c1512cfe89d`,
  400 draws), ensemble of three (`outputs/ablation_propagation/propagation-300`).
- Berkeley: 2 window replicates (a third stalled on one chain and was stopped).

## Not done / needs the user
- Smartphone video vs mocap on real data (OpenCap login or BioCV agreement).
- Affiliations, ethics, funding, journal choice; verify reference page ranges.
- Coverage fixes for the twin (prior spread growing with distance from PHV; more seasons) and
  covering risk intervals (bootstrap ensembles or conformal calibration).
