"""Run the pre-registered ablation grid (brief §6.7) with athlete-grouped cross-validation.

Every configuration in ``configs/eval/ablations.yaml`` is fitted on the training athletes of
each fold and scored on that fold's held-out athletes; out-of-fold predictions are pooled.
The output is one tidy table with a run ID per row. Nothing in the table is typed by hand.

Configurations 1-3 are the logistic baselines; 4-9 are neural hazards that differ only in how
the growth state conditions the movement encoder, then add mechanistic proxies and uncertainty
propagation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from youthtwin.eval import metrics, splits
from youthtwin.features.mech_load import MECH_PREFIX
from youthtwin.growth.growth_state import SitarPosterior
from youthtwin.models import dataset
from youthtwin.models.baselines import make_baseline
from youthtwin.models.encoder.sequences import ChannelScaler, SessionSequences
from youthtwin.models.pretrain import ssl
from youthtwin.models.risk import uncertainty
from youthtwin.models.risk.hazard_neural import fit_neural_hazard
from youthtwin.utils.provenance import config_hash

FloatArray = NDArray[np.float64]
RESULT_COLUMNS = [
    "run_id",
    "ablation_id",
    "name",
    "label",
    "metric",
    "value",
    "n_rows",
    "n_events",
    "n_folds",
    "synthetic",
    "note",
]


def _baseline_predictions(
    spec: Mapping[str, Any],
    frame: pd.DataFrame,
    label: str,
    folds: list[splits.Split],
    configs: Mapping[str, Any],
) -> FloatArray:
    out = np.full(len(frame), np.nan)
    for fold in folds:
        model = make_baseline(
            spec["baseline"], frame, configs["baselines"], configs["person_period"]
        )
        model.fit(frame[fold.train], label)
        out[fold.test] = model.predict_proba(frame[fold.test])
    return out


def _neural_predictions(
    spec: Mapping[str, Any],
    frame: pd.DataFrame,
    label: str,
    folds: list[splits.Split],
    sequences: SessionSequences,
    posteriors: Mapping[str, SitarPosterior] | None,
    configs: Mapping[str, Any],
    seed: int,
    pretrained_cache: dict[int, Any],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, list[str]]:
    """Out-of-fold risk, its SD and interval from uncertainty propagation (NaN if off), notes."""
    neural, encoder = configs["neural"], configs["encoder"]
    notes: list[str] = []
    growth = dataset.growth_columns(frame, neural)
    if not growth[0].startswith("twin_"):
        notes.append("no growth-twin posteriors: growth vector from observed measurements")
    tabular = [c for c in neural["tabular_columns"] if c in frame.columns]
    if spec.get("mech_proxies"):
        mech = [c for c in frame.columns if c.startswith(MECH_PREFIX)]
        tabular += mech
        if not mech:
            notes.append("mechanistic proxies unavailable in this table: same inputs as without")
    data = dataset.neural_data(
        frame,
        sequences,
        label=label,
        growth=growth,
        tabular=tabular,
        leakage_config=configs["person_period"],
    )
    propagate = bool(spec.get("propagate_uncertainty"))
    draws = None
    if propagate:
        if posteriors and growth[0].startswith("twin_"):
            draws = dataset.growth_draw_arrays(
                frame,
                posteriors,
                growth,
                n_draws=int(neural["uncertainty"]["n_growth_draws"]),
                seed=seed,
            )
        else:
            notes.append("uncertainty propagation needs growth-twin posteriors: not applied")
    risk = np.full(len(frame), np.nan)
    spread = np.full(len(frame), np.nan)
    low, high = spread.copy(), spread.copy()
    level = float(neural["uncertainty"]["interval_level"])
    for k, fold in enumerate(folds):
        pretrained = pretrained_cache.get(k)
        use_pretraining = neural["pretraining"]["enabled"] and encoder["backbone"] == "tcn"
        if use_pretraining and pretrained is None:
            train_athletes = set(frame.loc[fold.train, "athlete_id"])
            scaler = ChannelScaler.fit(sequences, train_athletes)
            rows = np.array([a in train_athletes for a in sequences.athlete_ids])
            real = sequences.mask[rows].astype(bool)
            trials = scaler.transform(sequences.trials[rows][real])
            pretrained, _ = ssl.pretrain(trials, encoder, neural["pretraining"], seed=seed + k)
            pretrained_cache[k] = pretrained  # the same for every configuration of this fold
        fitted = fit_neural_hazard(
            data,
            sequences,
            fold.train,
            mode=spec["conditioning"],
            sides=dataset.SIDES,
            encoder_config=encoder,
            config=neural,
            seed=seed + k,
            pretrained=pretrained,
        )
        test = np.flatnonzero(fold.test)
        if draws is None:
            risk[test] = fitted.predict_members(data, sequences, test).mean(axis=0)
        else:

            def predict(g: FloatArray, _f: Any = fitted, _t: Any = test) -> FloatArray:
                return _f.predict_members(data, sequences, _t, growth=g)  # type: ignore[no-any-return]

            distribution = uncertainty.propagate_growth_draws(predict, draws[:, test, :])
            risk[test], spread[test] = distribution.mean, distribution.sd
            low[test], high[test] = distribution.interval(level)
    return risk, spread, low, high, notes


def run_ablation(
    frame: pd.DataFrame,
    sequences: SessionSequences,
    *,
    label: str,
    configs: Mapping[str, Any],
    posteriors: Mapping[str, SitarPosterior] | None = None,
    n_folds: int,
    seed: int,
    true_risk: FloatArray | None = None,
) -> pd.DataFrame:
    """Run every configuration; return the tidy results table.

    ``true_risk`` (synthetic cohorts only) is the known event probability per row. If given,
    the table also reports how close predictions are to it and, where risk intervals exist,
    how often they contain it.

    ``configs`` holds the loaded YAML blocks: ``ablations, baselines, person_period, neural,
    encoder``.
    """
    folds = splits.grouped_stratified_folds(
        frame["athlete_id"], frame[label], n_splits=n_folds, seed=seed
    )
    base_id = config_hash(
        {"configs": dict(configs), "label": label, "seed": seed, "folds": n_folds}
    )
    synthetic = bool(frame["synthetic"].all())
    outcome = frame[label].to_numpy()
    rows: list[dict[str, Any]] = []
    pretrained_cache: dict[int, Any] = {}
    for spec in configs["ablations"]["ablations"]:
        spread = np.full(len(frame), np.nan)
        low, high = spread.copy(), spread.copy()
        if "baseline" in spec:
            notes: list[str] = []
            risk = _baseline_predictions(spec, frame, label, folds, configs)
        else:
            risk, spread, low, high, notes = _neural_predictions(
                spec, frame, label, folds, sequences, posteriors, configs, seed, pretrained_cache
            )
        values = {
            "auc": metrics.auc(outcome, risk),
            "brier": metrics.brier_score(outcome, risk),
            "mean_predicted_risk": float(np.mean(risk)),
            "observed_prevalence": float(np.mean(outcome)),
        }
        if np.isfinite(spread).any():
            values["mean_risk_sd_from_uncertainty"] = float(np.nanmean(spread))
            values["mean_risk_interval_width"] = float(np.nanmean(high - low))
        if true_risk is not None:
            known = np.isfinite(true_risk)
            values["mae_vs_true_risk"] = float(np.mean(np.abs(risk[known] - true_risk[known])))
            values["correlation_with_true_risk"] = float(
                np.corrcoef(risk[known], true_risk[known])[0, 1]
            )
            if np.isfinite(low).any():
                inside = (true_risk[known] >= low[known]) & (true_risk[known] <= high[known])
                values["interval_coverage_of_true_risk"] = float(inside.mean())
        for metric, value in values.items():
            rows.append(
                {
                    "run_id": f"ablation-{base_id}-{spec['id']}",
                    "ablation_id": int(spec["id"]),
                    "name": spec["name"],
                    "label": label,
                    "metric": metric,
                    "value": value,
                    "n_rows": len(frame),
                    "n_events": int(outcome.sum()),
                    "n_folds": n_folds,
                    "synthetic": synthetic,
                    "note": "; ".join(notes),
                }
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
