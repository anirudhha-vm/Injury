"""Sample-size criteria of Riley et al., checked against a worked example."""

from __future__ import annotations

import math

import pytest

from youthtwin.eval.sample_size import (
    max_cox_snell_r2,
    required_sample_size,
    supported_parameters,
)


def test_worked_example() -> None:
    """24 parameters, outcome proportion 0.174, anticipated Cox-Snell R-squared 0.288.

    By hand:
      maxR2 = 1 - exp(2 * (0.174 ln 0.174 + 0.826 ln 0.826)) = 1 - exp(-0.92434) = 0.6032
      criterion 1: 24 / ((0.9 - 1) * ln(1 - 0.288 / 0.9)) = 24 / (0.1 * 0.38566) = 622.3 -> 623
      criterion 2: S = 0.288 / (0.288 + 0.05 * 0.6032) = 0.9052;
                   24 / ((0.9052 - 1) * ln(1 - 0.288 / 0.9052)) = 661.1 -> 662
      criterion 3: (1.96 / 0.05)^2 * 0.174 * 0.826 = 220.9 -> 221
    The largest, 662, is the figure reported for this example with the pmsampsize software
    (recalled from the Riley et al. tutorial; TODO(human): confirm against the paper).
    """
    result = required_sample_size(parameters=24, prevalence=0.174, r2_cox_snell=0.288)
    assert result.max_r2 == pytest.approx(0.6032, abs=5e-4)
    assert (result.n_shrinkage, result.n_optimism, result.n_overall_risk) == (623, 662, 221)
    assert result.n_required == 662
    assert result.events_required == pytest.approx(115.2, abs=0.1)
    assert result.events_per_parameter == pytest.approx(4.8, abs=0.01)


def test_max_r2_properties() -> None:
    assert max_cox_snell_r2(0.5) == pytest.approx(0.75)  # 1 - (0.5)^2
    assert max_cox_snell_r2(0.1) == pytest.approx(max_cox_snell_r2(0.9))
    assert max_cox_snell_r2(0.02) < max_cox_snell_r2(0.2)  # rare outcomes carry less information
    with pytest.raises(ValueError, match="between 0 and 1"):
        max_cox_snell_r2(0.0)


def test_default_assumes_fifteen_percent_of_maximum() -> None:
    result = required_sample_size(parameters=10, prevalence=0.12)
    assert result.r2_cox_snell == pytest.approx(0.15 * max_cox_snell_r2(0.12))
    # By hand: maxR2 = 0.5199, R2 = 0.07799; 10 / (0.1 * -ln(1 - 0.07799 / 0.9)) = 1103.2 -> 1104
    assert result.n_shrinkage == 1104


def test_requirements_scale_as_expected() -> None:
    base = required_sample_size(parameters=5, prevalence=0.1)
    assert required_sample_size(parameters=10, prevalence=0.1).n_shrinkage == pytest.approx(
        2 * base.n_shrinkage, abs=1
    )
    assert required_sample_size(parameters=5, prevalence=0.04).n_required > base.n_required
    strong = required_sample_size(parameters=5, prevalence=0.1, r2_cox_snell=0.2)
    assert strong.n_required < base.n_required  # a more predictable outcome needs fewer athletes
    with pytest.raises(ValueError, match="at least 1"):
        required_sample_size(parameters=0, prevalence=0.1)
    with pytest.raises(ValueError, match="maximum"):
        required_sample_size(parameters=5, prevalence=0.1, r2_cox_snell=0.9)


def test_supported_parameters_is_the_inverse() -> None:
    for prevalence in (0.04, 0.15):
        for parameters in (3, 10):
            needed = required_sample_size(parameters=parameters, prevalence=prevalence)
            n = max(needed.n_shrinkage, needed.n_optimism)
            assert supported_parameters(n=n, prevalence=prevalence) == parameters
            assert (
                supported_parameters(n=n - math.ceil(n / parameters), prevalence=prevalence)
                < parameters
            )
    assert supported_parameters(n=20, prevalence=0.04) == 0
