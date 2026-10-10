"""Evaluation metrics: hand-worked values, scikit-learn agreement, known-truth simulations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from youthtwin.eval import bootstrap, calibration, dca, metrics, subgroups
from youthtwin.eval.predictions import out_of_fold_predictions, risk_columns
from youthtwin.eval.splits import Split, grouped_stratified_folds
from youthtwin.features.person_period import PersonPeriod
from youthtwin.models.baselines import make_baseline
from youthtwin.utils.config import load_config

Y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
P = np.array([0.1, 0.4, 0.35, 0.2, 0.8, 0.7, 0.6, 0.9])


def _calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = 1 / (1 + np.exp(-rng.normal(-2.0, 1.2, n)))
    return (rng.random(n) < risk).astype(float), risk


# ----------------------------------------------------------------------------- discrimination


def test_auc_hand_worked_and_against_sklearn() -> None:
    # 4 events (scores 0.35, 0.8, 0.7, 0.9) and 4 non-events (0.1, 0.4, 0.2, 0.6): 16 pairs.
    # 0.35 beats 0.1 and 0.2 (2 pairs); 0.8, 0.7 and 0.9 each beat all four (12) -> 14 / 16.
    assert metrics.auc(Y, P) == pytest.approx(14 / 16)
    assert metrics.auc(Y, P) == pytest.approx(skm.roc_auc_score(Y, P))
    assert metrics.time_dependent_auc(Y, P) == metrics.auc(Y, P)
    tied = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.5])
    assert metrics.auc(Y, tied) == pytest.approx(skm.roc_auc_score(Y, tied))  # ties count half
    assert metrics.auc(Y, 1 - P) == pytest.approx(1 - 14 / 16)
    assert np.isnan(metrics.auc(np.zeros(5), np.linspace(0, 1, 5)))
    with pytest.raises(ValueError, match="same length"):
        metrics.auc(Y, P[:-1])
    with pytest.raises(ValueError, match="0 or 1"):
        metrics.auc(np.array([0, 2]), np.array([0.1, 0.2]))


def test_pr_auc_and_brier() -> None:
    assert metrics.pr_auc(Y, P) == pytest.approx(skm.average_precision_score(Y, P))
    y, risk = _calibrated(3000, 0)
    assert metrics.pr_auc(y, risk) == pytest.approx(skm.average_precision_score(y, risk))
    rng = np.random.default_rng(1)
    assert metrics.pr_auc(y, rng.random(3000)) == pytest.approx(y.mean(), abs=0.03)  # chance
    assert np.isnan(metrics.pr_auc(np.zeros(4), np.ones(4)))
    # Brier: squares of (0.1, 0.4, 0.65, 0.2, 0.2, 0.3, 0.6, 0.1) sum to 1.1325; / 8 = 0.1415625
    assert metrics.brier_score(Y, P) == pytest.approx(0.1415625)
    assert metrics.brier_score(Y, P) == pytest.approx(skm.brier_score_loss(Y, P))


def test_sensitivity_and_specificity_at_a_threshold() -> None:
    result = metrics.sensitivity_specificity(Y, P, 0.5)
    # Flagged (risk >= 0.5): 0.8, 0.7, 0.6, 0.9 -> 3 events, 1 non-event.
    assert result["sensitivity"] == pytest.approx(3 / 4)
    assert result["specificity"] == pytest.approx(3 / 4)
    assert result["ppv"] == pytest.approx(3 / 4) and result["npv"] == pytest.approx(3 / 4)
    assert result["flagged_fraction"] == pytest.approx(0.5)
    nobody = metrics.sensitivity_specificity(Y, P, 0.99)
    assert nobody["sensitivity"] == 0.0 and np.isnan(nobody["ppv"])


# ----------------------------------------------------------------------------- calibration


def test_well_calibrated_predictions() -> None:
    y, risk = _calibrated(60000, 2)
    assert calibration.calibration_intercept(y, risk) == pytest.approx(0.0, abs=0.05)
    assert calibration.calibration_slope(y, risk) == pytest.approx(1.0, abs=0.04)
    assert calibration.integrated_calibration_index(y, risk) < 0.01
    assert calibration.expected_calibration_error(y, risk) < 0.01


def test_miscalibration_is_detected_with_the_right_sign() -> None:
    y, risk = _calibrated(60000, 3)
    logit = np.log(risk / (1 - risk))
    too_high = 1 / (1 + np.exp(-(logit + 0.7)))  # every risk overstated
    assert calibration.calibration_intercept(y, too_high) == pytest.approx(-0.7, abs=0.05)
    assert calibration.calibration_slope(y, too_high) == pytest.approx(1.0, abs=0.04)
    too_extreme = 1 / (1 + np.exp(-2.0 * logit))  # overfitting-like: slope should be 1/2
    assert calibration.calibration_slope(y, too_extreme) == pytest.approx(0.5, abs=0.03)
    too_timid = 1 / (1 + np.exp(-0.5 * logit))
    assert calibration.calibration_slope(y, too_timid) == pytest.approx(2.0, abs=0.1)
    assert calibration.integrated_calibration_index(y, too_high) > 0.05
    assert calibration.expected_calibration_error(y, too_high) > 0.05


def test_calibration_slope_matches_logistic_regression() -> None:
    from sklearn.linear_model import LogisticRegression

    y, risk = _calibrated(5000, 4)
    noisy = np.clip(risk * np.random.default_rng(0).uniform(0.5, 1.8, risk.size), 1e-4, 0.999)
    logit = np.log(noisy / (1 - noisy))[:, None]
    reference = LogisticRegression(C=np.inf, max_iter=1000).fit(logit, y)
    assert calibration.calibration_slope(y, noisy) == pytest.approx(reference.coef_[0, 0], rel=1e-4)


def test_calibration_bins_and_edge_cases() -> None:
    bins = calibration.calibration_bins(Y, P, n_bins=4)
    assert bins["n"].tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(bins["predicted"], [0.15, 0.375, 0.65, 0.85])
    np.testing.assert_allclose(bins["observed"], [0.0, 0.5, 0.5, 1.0])
    # ECE = mean |0.15 - 0|, |0.375 - 0.5|, |0.65 - 0.5|, |0.85 - 1| = (0.15+0.125+0.15+0.15)/4
    assert calibration.expected_calibration_error(Y, P, n_bins=4) == pytest.approx(0.14375)
    with pytest.raises(ValueError, match="both events and non-events"):
        calibration.calibration_slope(np.zeros(10), np.full(10, 0.2))
    smooth = calibration.smoothed_observed_risk(*_calibrated(2000, 5))
    assert smooth.min() >= 0 and smooth.max() <= 1


# ----------------------------------------------------------------------------- decision curves


def test_net_benefit_hand_worked() -> None:
    # Threshold 0.5: TP = 3, FP = 1, n = 8 -> 3/8 - 1/8 * (0.5 / 0.5) = 0.25
    assert dca.net_benefit(Y, P, 0.5) == pytest.approx(0.25)
    # Threshold 0.2: flagged = all but 0.1 -> TP = 4, FP = 3 -> 4/8 - 3/8 * 0.25 = 0.40625
    assert dca.net_benefit(Y, P, 0.2) == pytest.approx(0.40625)
    # Treat all at 0.2: prevalence 0.5 - 0.5 * 0.25 = 0.375
    assert dca.net_benefit_treat_all(Y, 0.2) == pytest.approx(0.375)
    assert dca.net_benefit(Y, np.ones(8), 0.2) == pytest.approx(dca.net_benefit_treat_all(Y, 0.2))
    assert dca.net_benefit(Y, np.zeros(8), 0.2) == 0.0  # nobody selected = treat none
    with pytest.raises(ValueError, match="between 0 and 1"):
        dca.net_benefit(Y, P, 1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        dca.net_benefit_treat_all(Y, 0.0)


def test_decision_curve_table_and_usefulness() -> None:
    y, risk = _calibrated(40000, 6)
    rng = np.random.default_rng(0)
    useless = np.full(y.size, y.mean()) + rng.normal(0, 1e-6, y.size)
    thresholds = np.linspace(0.02, 0.4, 20)
    table = dca.decision_curve(y, {"model": risk, "useless": useless}, thresholds)
    assert set(table["strategy"]) == {"treat_all", "treat_none", "model", "useless"}
    wide = table.pivot(index="threshold", columns="strategy", values="net_benefit")
    best_reference = wide[["treat_all", "treat_none"]].max(axis=1)
    assert (wide["model"] >= best_reference - 1e-3).all()  # a calibrated model is never worse
    assert (wide["model"] > best_reference + 1e-3).mean() > 0.7  # and usually better
    assert (wide["useless"] <= best_reference + 2e-3).all()  # a constant predictor never helps
    assert (wide["treat_none"] == 0).all()


# ----------------------------------------------------------------------------- cluster bootstrap


def _clustered(n_athletes: int, weeks: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcomes and risks that are nearly constant within athlete (like overlapping windows)."""
    rng = np.random.default_rng(seed)
    level = rng.normal(-1.5, 1.0, n_athletes)
    athlete = np.repeat(np.arange(n_athletes), weeks)
    risk = 1 / (1 + np.exp(-(level[athlete] + rng.normal(0, 0.1, athlete.size))))
    outcome = np.repeat((rng.random(n_athletes) < 1 / (1 + np.exp(-level))).astype(float), weeks)
    return outcome, risk, athlete


def test_bootstrap_resamples_whole_athletes() -> None:
    clusters = bootstrap._row_groups(np.array(["b", "a", "b", "c", "a", "b"]))
    assert [c.tolist() for c in clusters] == [[1, 4], [0, 2, 5], [3]]
    rows = bootstrap._resampled_rows(clusters, np.random.default_rng(0))
    labels = np.array(["b", "a", "b", "c", "a", "b"])[rows]
    counts = pd.Series(labels).value_counts()
    # Each athlete appears a whole number of times their own row count (a: 2, b: 3, c: 1).
    assert all(counts[k] % size == 0 for k, size in (("a", 2), ("b", 3), ("c", 1)) if k in counts)


def test_cluster_interval_is_wider_than_a_row_interval() -> None:
    outcome, risk, athlete = _clustered(150, 20, 1)
    clustered = bootstrap.cluster_bootstrap(
        metrics.auc, outcome, risk, athlete, n_resamples=300, seed=0
    )
    by_row = bootstrap.cluster_bootstrap(
        metrics.auc, outcome, risk, np.arange(outcome.size), n_resamples=300, seed=0
    )
    assert clustered.ci_low < clustered.estimate < clustered.ci_high
    assert clustered.estimate == pytest.approx(metrics.auc(outcome, risk))
    # Treating 20 near-identical rows per athlete as independent understates uncertainty.
    assert (clustered.ci_high - clustered.ci_low) > 2.5 * (by_row.ci_high - by_row.ci_low)
    assert clustered.n_resamples == 300 and clustered.level == 0.95


def test_cluster_interval_has_about_nominal_coverage() -> None:
    """The 90% interval for mean risk covers the population value about 90% of the time."""
    hits = 0
    for seed in range(120):
        outcome, risk, athlete = _clustered(60, 8, 100 + seed)
        result = bootstrap.cluster_bootstrap(
            lambda y, p: float(np.mean(y)),
            outcome,
            risk,
            athlete,
            n_resamples=200,
            level=0.90,
            seed=seed,
        )
        hits += result.ci_low <= 0.2227 <= result.ci_high  # E[sigmoid(N(-1.5, 1))]
    assert hits / 120 == pytest.approx(0.90, abs=0.07)


def test_paired_difference_between_models() -> None:
    outcome, risk, athlete = _clustered(200, 10, 2)
    rng = np.random.default_rng(3)
    worse = rng.permutation(risk)  # no information
    slightly_noisy = np.clip(risk + rng.normal(0, 1e-4, risk.size), 0, 1)
    clear = bootstrap.cluster_bootstrap_difference(
        metrics.auc, outcome, risk, worse, athlete, n_resamples=300, seed=0
    )
    assert clear.estimate > 0.15 and clear.ci_low > 0  # interval excludes zero
    same = bootstrap.cluster_bootstrap_difference(
        metrics.auc, outcome, risk, slightly_noisy, athlete, n_resamples=300, seed=0
    )
    assert same.ci_low < 0 < same.ci_high or abs(same.estimate) < 1e-3
    # Paired resampling: the difference interval is much narrower than either model's own.
    own = bootstrap.cluster_bootstrap(metrics.auc, outcome, risk, athlete, n_resamples=300, seed=0)
    assert (same.ci_high - same.ci_low) < 0.2 * (own.ci_high - own.ci_low)


def test_bootstrap_with_undefined_resamples() -> None:
    outcome = np.array([0.0, 0.0, 0.0, 0.0])
    result = bootstrap.cluster_bootstrap(
        metrics.auc, outcome, np.linspace(0, 1, 4), np.arange(4), n_resamples=20
    )
    assert result.n_resamples == 0 and np.isnan(result.ci_low)


# ----------------------------------------------------------------------------- predictions


@pytest.fixture(scope="module")
def predictions(person_period: PersonPeriod, person_period_config: dict[str, Any]) -> pd.DataFrame:
    frame = person_period.frame
    baselines = load_config("models/baselines.yaml")
    folds = grouped_stratified_folds(
        frame["athlete_id"], frame["y_all_non_contact"], n_splits=3, seed=0
    )
    factories = {
        name: (lambda f, _n=name: make_baseline(_n, f, baselines, person_period_config))
        for name in ("age_only", "maturity_growth_velocity")
    }
    return out_of_fold_predictions(
        frame,
        "y_all_non_contact",
        factories,
        folds,
        keep=["sex", "sport", "age", "moore_maturity_offset"],
    )


def test_out_of_fold_predictions(predictions: pd.DataFrame, person_period: PersonPeriod) -> None:
    frame = person_period.frame
    assert len(predictions) == len(frame) and predictions["synthetic"].all()
    assert risk_columns(predictions) == {
        "age_only": "risk_age_only",
        "maturity_growth_velocity": "risk_maturity_growth_velocity",
    }
    assert predictions[list(risk_columns(predictions).values())].notna().all().all()
    assert set(predictions["fold"]) == {0, 1, 2}
    # Every athlete's rows are in a single fold: predicted by a model that never saw them.
    assert (predictions.groupby("athlete_id")["fold"].nunique() == 1).all()
    assert (predictions["outcome"].to_numpy() == frame["y_all_non_contact"].to_numpy()).all()


def test_predictions_reject_bad_folds(
    person_period: PersonPeriod, person_period_config: dict[str, Any]
) -> None:
    frame = person_period.frame
    half = np.arange(len(frame)) < len(frame) // 2
    baselines = load_config("models/baselines.yaml")
    factory = {"age_only": lambda f: make_baseline("age_only", f, baselines, person_period_config)}
    with pytest.raises(AssertionError, match="both train and test"):
        out_of_fold_predictions(frame, "y_all_non_contact", factory, [Split("rows", ~half, half)])
    one = grouped_stratified_folds(
        frame["athlete_id"], frame["y_all_non_contact"], n_splits=3, seed=0
    )[:1]
    with pytest.raises(ValueError, match="exactly one test part"):
        out_of_fold_predictions(frame, "y_all_non_contact", factory, one)


def test_subgroup_table(predictions: pd.DataFrame) -> None:
    config = load_config("eval/metrics.yaml")["subgroups"]
    with_strata = subgroups.add_strata(predictions, config, "moore_maturity_offset")
    assert set(with_strata["age_band"].dropna()) == {"8-11", "11-14", "14-19"}
    assert set(with_strata["maturity_stratum"].dropna()) == {"pre_phv", "circa_phv", "post_phv"}
    table = subgroups.subgroup_table(
        with_strata,
        "risk_age_only",
        config["by"],
        {"auc": metrics.auc, "brier": metrics.brier_score},
        min_events=config["min_events"],
    )
    assert set(table["subgroup"]) == {"sex", "maturity_stratum", "sport", "age_band"}
    by_sex = table[table["subgroup"] == "sex"]
    assert by_sex["n_rows"].sum() == len(predictions)
    assert by_sex["n_athletes"].sum() == predictions["athlete_id"].nunique()
    assert by_sex["n_events"].sum() == int(predictions["outcome"].sum())
    assert table.loc[table["note"] == "", "auc"].between(0, 1).all()
    # A subgroup with too few events is listed with counts and a note, not silently dropped.
    strict = subgroups.subgroup_table(
        with_strata, "risk_age_only", ["sport"], {"auc": metrics.auc}, min_events=10_000
    )
    assert len(strict) == 3 and strict["auc"].isna().all()
    assert strict["note"].str.contains("fewer than").all()
