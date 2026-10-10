"""Minimum sample size for developing a prediction model with a binary outcome.

Criteria of Riley et al. (Stat Med 2019; BMJ 2020;368:m441), as implemented in ``pmsampsize``:

1. **Shrinkage**: expected uniform shrinkage of predictor effects at least ``S`` (0.9):
   ``n = p / ((S - 1) * ln(1 - R2 / S))``.
2. **Optimism**: apparent minus adjusted Nagelkerke R-squared at most ``delta`` (0.05):
   the same formula with ``S = R2 / (R2 + delta * maxR2)``.
3. **Overall risk**: the outcome proportion estimated to within ``margin`` (0.05):
   ``n = (1.96 / margin)^2 * phi * (1 - phi)``.

``R2`` is the anticipated Cox-Snell R-squared, ``p`` the number of candidate predictor
parameters (spline terms and dummy variables each count), ``phi`` the outcome proportion, and
``maxR2 = 1 - (phi^phi * (1 - phi)^(1 - phi))^2`` the largest Cox-Snell R-squared possible at
that proportion. With no prior estimate, Riley et al. suggest assuming the model reaches 15% of
``maxR2``.

Scope: the unit is one independent observation per participant (for example, "any
growth-related injury during the season"). Athlete-weeks from one athlete are not independent,
so counting athlete-weeks as the sample size would overstate the information available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_SHRINKAGE = 0.9
DEFAULT_OPTIMISM = 0.05
DEFAULT_MARGIN = 0.05
DEFAULT_FRACTION_OF_MAX_R2 = 0.15
_Z = 1.959963984540054  # standard normal 97.5th percentile


def max_cox_snell_r2(prevalence: float) -> float:
    """Largest possible Cox-Snell R-squared for a binary outcome with this proportion."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between 0 and 1")
    log_likelihood_null = prevalence * math.log(prevalence) + (1 - prevalence) * math.log(
        1 - prevalence
    )
    return 1.0 - math.exp(2.0 * log_likelihood_null)


def _n_for_shrinkage(parameters: int, r2: float, shrinkage: float) -> float:
    return parameters / ((shrinkage - 1.0) * math.log(1.0 - r2 / shrinkage))


@dataclass(frozen=True)
class SampleSize:
    parameters: int
    prevalence: float
    r2_cox_snell: float
    max_r2: float
    n_shrinkage: int  # criterion 1
    n_optimism: int  # criterion 2
    n_overall_risk: int  # criterion 3

    @property
    def n_required(self) -> int:
        return max(self.n_shrinkage, self.n_optimism, self.n_overall_risk)

    @property
    def events_required(self) -> float:
        return self.n_required * self.prevalence

    @property
    def events_per_parameter(self) -> float:
        return self.events_required / self.parameters


def required_sample_size(
    *,
    parameters: int,
    prevalence: float,
    r2_cox_snell: float | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    optimism: float = DEFAULT_OPTIMISM,
    margin: float = DEFAULT_MARGIN,
) -> SampleSize:
    """Minimum number of participants for a model with ``parameters`` candidate parameters.

    If ``r2_cox_snell`` is omitted, 15% of the maximum is assumed.
    """
    if parameters < 1:
        raise ValueError("parameters must be at least 1")
    max_r2 = max_cox_snell_r2(prevalence)
    r2 = DEFAULT_FRACTION_OF_MAX_R2 * max_r2 if r2_cox_snell is None else r2_cox_snell
    if not 0.0 < r2 < max_r2:
        raise ValueError(f"r2_cox_snell must be between 0 and its maximum {max_r2:.3f}")
    shrinkage_for_optimism = r2 / (r2 + optimism * max_r2)
    return SampleSize(
        parameters=parameters,
        prevalence=prevalence,
        r2_cox_snell=r2,
        max_r2=max_r2,
        n_shrinkage=math.ceil(_n_for_shrinkage(parameters, r2, shrinkage)),
        n_optimism=math.ceil(_n_for_shrinkage(parameters, r2, shrinkage_for_optimism)),
        n_overall_risk=math.ceil((_Z / margin) ** 2 * prevalence * (1 - prevalence)),
    )


def supported_parameters(
    *,
    n: int,
    prevalence: float,
    r2_cox_snell: float | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    optimism: float = DEFAULT_OPTIMISM,
) -> int:
    """Largest number of candidate parameters that ``n`` participants support (criteria 1-2)."""
    max_r2 = max_cox_snell_r2(prevalence)
    r2 = DEFAULT_FRACTION_OF_MAX_R2 * max_r2 if r2_cox_snell is None else r2_cox_snell
    limits = []
    for s in (shrinkage, r2 / (r2 + optimism * max_r2)):
        limits.append(n * (s - 1.0) * math.log(1.0 - r2 / s))
    return max(0, math.floor(min(limits)))
