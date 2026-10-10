"""Agreement statistics: hand-worked values, a reference implementation, and known-truth data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from youthtwin.utils.config import load_config
from youthtwin.validation import agreement as ag
from youthtwin.validation import mocap_io, plots

# A small example worked by hand. Differences (method - reference): 1, 2, 0, 3, -1.
METHOD = np.array([11.0, 14.0, 9.0, 18.0, 12.0])
REFERENCE = np.array([10.0, 12.0, 9.0, 15.0, 13.0])


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return load_config("validation/agreement.yaml")


def _pairs(
    rng: np.random.Generator, n: int, bias: float, noise: float, slope: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.normal(60.0, 15.0, n)
    reference = truth + rng.normal(0.0, 0.5, n)
    method = truth + bias + slope * (truth - 60.0) + rng.normal(0.0, noise, n)
    return method, reference


# ----------------------------------------------------------------------------- hand-worked


def test_bland_altman_hand_worked() -> None:
    # mean difference = 5 / 5 = 1.0; squared deviations 0, 1, 1, 4, 4 -> SD = sqrt(10 / 4) = 1.5811
    result = ag.bland_altman(METHOD, REFERENCE)
    assert result.n == 5
    assert result.bias == pytest.approx(1.0)
    assert result.sd == pytest.approx(1.5811388)
    assert result.loa_low == pytest.approx(1.0 - 1.959964 * 1.5811388, abs=1e-5)
    assert result.loa_high == pytest.approx(1.0 + 1.959964 * 1.5811388, abs=1e-5)
    # CI of the bias: 1.0 +/- t(0.975, 4) * SD / sqrt(5) = 1.0 +/- 2.776445 * 0.7071068
    assert result.bias_ci_low == pytest.approx(1.0 - 2.776445 * 0.7071068, abs=1e-5)
    # SE of a limit: SD * sqrt(1/5 + 1.959964^2 / 8) = 1.5811388 * sqrt(0.680182) = 1.304014
    half_width = 2.776445 * 1.304014
    assert result.loa_high_ci[1] - result.loa_high == pytest.approx(half_width, abs=1e-4)
    assert result.loa_low - result.loa_low_ci[0] == pytest.approx(half_width, abs=1e-4)


def test_rmse_and_mae_hand_worked() -> None:
    # squared differences 1, 4, 0, 9, 1 -> mean 3 -> RMSE sqrt(3); |d| mean = 7 / 5
    assert ag.rmse(METHOD, REFERENCE) == pytest.approx(np.sqrt(3.0))
    assert ag.mean_absolute_error(METHOD, REFERENCE) == pytest.approx(1.4)


def test_icc_hand_worked() -> None:
    # Two-way ANOVA for the 5 x 2 table above:
    #   row means 10.5, 13, 9, 16.5, 12.5; column means 12.8, 11.8; grand mean 12.3
    #   SS_rows = 2 * (3.24 + 0.49 + 10.89 + 17.64 + 0.04) = 64.6   -> MSR = 16.15
    #   SS_cols = 5 * (0.25 + 0.25) = 2.5                           -> MSC = 2.5
    #   SS_total = 72.1 -> SS_error = 5.0                           -> MSE = 1.25
    #   ICC(3,1) = (16.15 - 1.25) / (16.15 + 1.25) = 0.856322
    #   ICC(2,1) = 14.9 / (16.15 + 1.25 + 2 * (2.5 - 1.25) / 5) = 14.9 / 17.9 = 0.832402
    assert ag.icc_consistency(METHOD, REFERENCE).value == pytest.approx(0.856322, abs=1e-6)
    assert ag.icc_agreement(METHOD, REFERENCE).value == pytest.approx(0.832402, abs=1e-6)


# ----------------------------------------------------------------------------- reference library


@pytest.mark.parametrize(("bias", "noise", "n"), [(0.0, 2.0, 30), (4.0, 3.0, 40), (-2.0, 8.0, 25)])
def test_icc_matches_pingouin(bias: float, noise: float, n: int) -> None:
    pingouin = pytest.importorskip("pingouin")
    method, reference = _pairs(np.random.default_rng(n), n, bias, noise)
    long = pd.DataFrame(
        {
            "subject": np.tile(np.arange(n), 2),
            "rater": np.repeat(["markerless", "mocap"], n),
            "score": np.concatenate([method, reference]),
        }
    )
    table = pingouin.intraclass_corr(
        data=long, targets="subject", raters="rater", ratings="score"
    ).set_index("Type")
    # pingouin (McGraw & Wong naming): ICC(A,1) = absolute agreement = Shrout-Fleiss ICC(2,1);
    # ICC(C,1) = consistency = ICC(3,1). Older versions call them ICC2 and ICC3.
    names = {"ICC(A,1)": "ICC2", "ICC(C,1)": "ICC3"}
    table = table.rename(index=names)
    column = next(c for c in table.columns if c.startswith("CI95"))
    for ours, kind in (
        (ag.icc_agreement(method, reference), "ICC2"),
        (ag.icc_consistency(method, reference), "ICC3"),
    ):
        assert ours.value == pytest.approx(float(table.loc[kind, "ICC"]), abs=1e-9)
        low, high = table.loc[kind, column]
        # pingouin rounds its interval to two decimals.
        assert ours.ci_low == pytest.approx(float(low), abs=0.006)
        assert ours.ci_high == pytest.approx(float(high), abs=0.006)
        assert ours.n == n and ours.ci_low < ours.value < ours.ci_high


def test_bland_altman_matches_scipy_building_blocks() -> None:
    """Independent recomputation of the limits and the proportional-bias regression."""
    from scipy import stats

    method, reference = _pairs(np.random.default_rng(7), 60, 2.0, 3.0, slope=0.1)
    result = ag.bland_altman(method, reference)
    difference, mean = method - reference, (method + reference) / 2
    assert result.bias == pytest.approx(difference.mean())
    assert result.loa_high - result.loa_low == pytest.approx(2 * 1.959964 * difference.std(ddof=1))
    fit = stats.linregress(mean, difference)
    assert result.proportional_slope == pytest.approx(fit.slope)
    assert result.proportional_p == pytest.approx(fit.pvalue)


# ----------------------------------------------------------------------------- known truth


def test_known_bias_and_noise_are_recovered() -> None:
    method, reference = _pairs(np.random.default_rng(0), 4000, bias=3.0, noise=4.0)
    result = ag.bland_altman(method, reference)
    spread = np.sqrt(4.0**2 + 0.5**2)  # method noise plus reference noise
    assert result.bias == pytest.approx(3.0, abs=0.2)
    assert result.sd == pytest.approx(spread, rel=0.05)
    assert result.loa_low == pytest.approx(3.0 - 1.96 * spread, abs=0.4)
    assert result.bias_ci_low < 3.0 < result.bias_ci_high
    assert ag.rmse(method, reference) == pytest.approx(np.sqrt(3.0**2 + spread**2), rel=0.05)
    # No proportional bias was built in, but unequal error variances of the two methods induce
    # a small slope of the difference on the mean: (var_m - var_r) / (2 var_mean)
    # = (16 - 0.25) / (2 * (225 + 4.06)) = 0.034. A known property of the Bland-Altman plot.
    assert result.proportional_slope == pytest.approx(0.034, abs=0.012)


def test_limits_of_agreement_contain_95_percent_of_differences() -> None:
    method, reference = _pairs(np.random.default_rng(1), 5000, bias=-1.0, noise=2.5)
    result = ag.bland_altman(method, reference)
    inside = np.mean((method - reference > result.loa_low) & (method - reference < result.loa_high))
    assert inside == pytest.approx(0.95, abs=0.01)


def test_confidence_interval_of_the_bias_has_nominal_coverage() -> None:
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(400):
        method, reference = _pairs(rng, 20, bias=1.5, noise=3.0)
        result = ag.bland_altman(method, reference)
        hits += result.bias_ci_low < 1.5 < result.bias_ci_high
    assert hits / 400 == pytest.approx(0.95, abs=0.035)


def test_proportional_bias_is_detected() -> None:
    method, reference = _pairs(np.random.default_rng(3), 300, bias=0.0, noise=2.0, slope=0.15)
    result = ag.bland_altman(method, reference)
    assert result.proportional_slope == pytest.approx(0.15, abs=0.03)
    assert result.proportional_p < 0.001


def test_icc_agreement_penalises_offset_but_consistency_does_not() -> None:
    method, reference = _pairs(np.random.default_rng(4), 500, bias=0.0, noise=3.0)
    shifted = method + 12.0
    assert ag.icc_consistency(shifted, reference).value == pytest.approx(
        ag.icc_consistency(method, reference).value
    )
    assert (
        ag.icc_agreement(shifted, reference).value < ag.icc_agreement(method, reference).value - 0.1
    )
    assert ag.icc_agreement(reference, reference + 1e-9).value == pytest.approx(1.0, abs=1e-6)


def test_missing_pairs_and_small_samples() -> None:
    method = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    reference = np.array([1.5, np.nan, 3.0, 4.5, 5.5])
    assert ag.bland_altman(method, reference).n == 3  # only complete pairs are used
    with pytest.raises(ValueError, match="at least 3"):
        ag.rmse([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        ag.rmse([1.0, 2.0, 3.0], [1.0, 2.0])
    flat = ag.bland_altman([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert np.isnan(flat.proportional_slope)


# ----------------------------------------------------------------------------- tables and figures


def _study(rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    specification = {"knee_flexion": (1.0, 2.0), "knee_fppa": (4.0, 6.0)}
    for variable, (bias, noise) in specification.items():
        for group in ("pre_phv", "circa_phv", "post_phv"):
            method, reference = _pairs(rng, 30, bias, noise)
            frames.append(
                pd.DataFrame(
                    {
                        "athlete_id": [f"{group}{k}" for k in range(30)],
                        "variable": variable,
                        "task": "drop_vertical_jump",
                        "event": "initial_contact",
                        "markerless": method,
                        "mocap": reference,
                        "maturity_group": group,
                        "sex": np.where(np.arange(30) % 2 == 0, "female", "male"),
                        "synthetic": True,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_agreement_table(config: dict[str, Any]) -> None:
    pairs = _study(np.random.default_rng(5))
    overall = ag.agreement_table(pairs, config).set_index("variable")
    assert overall.loc["knee_flexion", "n"] == 90 and overall["synthetic"].all()
    assert overall.loc["knee_flexion", "bias"] == pytest.approx(1.0, abs=0.6)
    assert bool(overall.loc["knee_flexion", "meets_threshold"]) is True  # RMSE about 2.3 deg
    assert bool(overall.loc["knee_fppa", "meets_threshold"]) is False  # RMSE about 7.2 deg
    for column in ("loa_low_ci_low", "loa_high_ci_high", "icc21", "icc31_ci_low", "rmse", "mae"):
        assert column in overall.columns

    by_group = ag.agreement_table(pairs, config, by=["maturity_group"])
    assert len(by_group) == 6 and set(by_group["maturity_group"]) == set(plots.STRATUM_ORDER)
    assert (by_group["n"] == 30).all()
    by_both = ag.agreement_table(pairs, config, by=["maturity_group", "sex"])
    assert len(by_both) == 12 and (by_both["n"] == 15).all()


def test_groups_with_too_few_pairs_are_kept_with_a_note(config: dict[str, Any]) -> None:
    pairs = _study(np.random.default_rng(6))
    sparse = pairs[~((pairs["variable"] == "knee_fppa") & (pairs["maturity_group"] == "pre_phv"))]
    extra = pairs[(pairs["variable"] == "knee_fppa") & (pairs["maturity_group"] == "pre_phv")].head(
        2
    )
    table = ag.agreement_table(pd.concat([sparse, extra]), config, by=["maturity_group"])
    row = table[(table["variable"] == "knee_fppa") & (table["maturity_group"] == "pre_phv")].iloc[0]
    assert row["n"] == 2 and "at least 3" in row["note"] and np.isnan(row["rmse"])
    assert len(table) == 6  # the thin group is reported, not dropped
    with pytest.raises(ValueError, match="missing columns"):
        ag.agreement_table(pairs.drop(columns="mocap"), config)


def test_figures(config: dict[str, Any], tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from youthtwin.utils.plotting import is_watermarked, save_figure

    pairs = _study(np.random.default_rng(8))
    grid = plots.bland_altman_grid(pairs, config)
    assert len([a for a in grid.axes if a.get_visible()]) == 2
    save_figure(grid, tmp_path / "ba.png", {"synthetic": True})
    assert is_watermarked(grid) and (tmp_path / "ba.png.provenance.json").is_file()
    bars = plots.rmse_by_stratum(ag.agreement_table(pairs, config, by=["maturity_group"]), config)
    labels = {t.get_text() for t in bars.axes[0].get_legend().get_texts()}
    assert labels == {*plots.STRATUM_ORDER, "pre-specified limit"}
    percent = np.linspace(0, 100, 101)
    curves = 60 * np.sin(np.pi * percent / 100)[None, :] + np.random.default_rng(0).normal(
        0, 2, (6, 101)
    )
    overlay = plots.curve_overlay(percent, curves, curves + 2.0, "knee flexion")
    assert len(overlay.axes[0].lines) == 2
    plt.close("all")


# ----------------------------------------------------------------------------- mocap I/O


def test_trc_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    positions = rng.normal(0.5, 0.2, (40, 3, 3))
    positions[5, 1] = np.nan  # an occluded marker
    data = mocap_io.MarkerData(("RASI", "RKNE", "RANK"), np.arange(40) / 200.0, positions, 200.0)
    loaded = mocap_io.read_trc(mocap_io.write_trc(data, tmp_path / "trial.trc"))
    assert loaded.names == data.names and loaded.rate_hz == 200.0
    np.testing.assert_allclose(loaded.positions, positions, atol=1e-7)
    np.testing.assert_allclose(loaded.time_s, data.time_s, atol=1e-6)
    assert np.isnan(loaded.marker("RKNE")[5]).all()
    with pytest.raises(KeyError, match="No marker"):
        loaded.marker("LKNE")
    (tmp_path / "bad.trc").write_text("not a trc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a TRC"):
        mocap_io.read_trc(tmp_path / "bad.trc")


def test_c3d_reader_needs_the_optional_package(tmp_path: Path) -> None:
    try:
        import ezc3d  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="ezc3d"):
            mocap_io.read_c3d(tmp_path / "trial.c3d")
    else:  # pragma: no cover - only when the optional reader is installed
        pytest.skip("ezc3d is installed")


def test_time_alignment_recovers_a_known_lag(config: dict[str, Any]) -> None:
    def knee(t: np.ndarray) -> np.ndarray:
        return 20 + 60 * np.exp(-((t - 1.2) ** 2) / 0.05)

    markerless_time = np.arange(0, 3.0, 1 / 120.0)
    mocap_time = np.arange(0, 3.5, 1 / 200.0)
    # The mocap system started 0.37 s earlier, so the same instant reads 0.37 s later there.
    alignment = mocap_io.align_curves(
        knee(markerless_time),
        markerless_time,
        knee(mocap_time - 0.37),
        mocap_time,
        max_lag_s=config["alignment"]["max_lag_s"],
    )
    assert alignment.offset_s == pytest.approx(-0.37, abs=0.005)
    assert alignment.peak_correlation > 0.95
    valid = np.isfinite(alignment.mocap_on_markerless_clock)
    assert valid.mean() > 0.8
    np.testing.assert_allclose(
        alignment.mocap_on_markerless_clock[valid], knee(markerless_time)[valid], atol=0.5
    )
    with pytest.raises(ValueError, match="exceeds the allowed"):
        mocap_io.align_curves(
            knee(markerless_time),
            markerless_time,
            knee(mocap_time - 0.37),
            mocap_time,
            max_lag_s=0.1,
        )
