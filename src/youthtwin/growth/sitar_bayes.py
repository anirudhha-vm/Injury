"""Hierarchical Bayesian SITAR-style growth model (contribution C1), marginalised form.

    stature_ij ~ Normal(intercept + alpha_i + h((age_ij - beta_i) * exp(gamma_i)), sigma)

- ``h`` is a shared natural cubic spline (``youthtwin.growth.splines``).
- ``alpha_i`` (size, cm) ~ Normal(0, sigma_alpha).
- ``beta_i`` (timing, years) ~ Normal(Moore-2015 APHV of athlete i, sd_years). Moore's estimate
  is biased toward the mean (see ``maturity_equations``); the prior SD lets the longitudinal
  data pull each athlete away from it.
- ``gamma_i`` (intensity, log scale) ~ Normal(0, sigma_gamma); optional.
- Optional velocity deviation ``delta_i * (age - mean_age_i)``, ``delta_i`` ~ Normal(0,
  sigma_delta): lets an athlete grow a little faster or slower than the shared shape implies.
  Without it, any mismatch between an athlete's curve and the shared shape is forced into the
  timing estimate, which made intervals too narrow in the first simulation study.
- Optional cohort-level ``timing_offset`` added to every Moore prior centre, and a learned prior
  SD: the model estimates how biased and how imprecise Moore is in *this* cohort.
- Optional cohort prior ``beta_i`` ~ Normal(cohort_mean, cohort_sd), both learned. With it the
  Moore value is treated as a noisy measurement of timing rather than as the only prior, so an
  athlete whose own data say little (years away from the growth spurt) is pulled toward the
  cohort's typical timing instead of toward a Moore value that is biased at that distance.
- One model per sex.

Why the per-athlete effects are integrated out rather than sampled
------------------------------------------------------------------
With about one season of data an athlete's timing is often genuinely two-peaked: a given growth
speed fits either the rising or the falling side of the growth spurt, roughly two years apart.
A gradient-based sampler cannot move between those peaks, so sampling ``beta_i`` directly gives
chains that disagree (observed during development; see docs/CHANGELOG_REASONING.md).

Instead:

1. ``alpha_i`` and ``delta_i`` are integrated out in closed form (linear-Gaussian).
2. ``beta_i`` and ``gamma_i`` are integrated out numerically on a fixed grid.
3. NUTS samples only the shared parameters (mean curve, SDs), about a dozen dimensions.
4. Per-athlete effects are then drawn exactly from their conditional distribution given each
   posterior draw of the shared parameters, so two-peaked timing is represented faithfully.

Step 4 needs no sampler, which also gives the incremental update: when an athlete has a new
measurement, their effects are redrawn against the stored shared draws (``update``).

After Cole, Donaldson & Ben-Shlomo (2010); the R package ``bsitar`` is a Bayesian SITAR that
samples the random effects directly. Age at PHV is derived from the fitted curve, not read off
``beta_i`` — see ``growth_state.SitarPosterior``.

All priors, grids and sampler settings come from ``configs/growth/sitar.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import NDArray

from youthtwin.growth import splines
from youthtwin.growth.growth_state import SitarPosterior, age_on
from youthtwin.growth.maturity_equations import Sex, moore2015

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
InferenceData = Any  # ArviZ containers are dynamically typed

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GrowthData:
    """Model-ready stature data for one sex."""

    athlete_ids: tuple[str, ...]
    athlete_index: IntArray  # per observation
    age: FloatArray  # per observation, decimal years
    stature: FloatArray  # per observation, cm (mean of duplicate measures)
    prior_aphv: FloatArray  # per athlete, Moore-2015 predicted age at PHV
    sex: Sex
    synthetic: bool
    # per athlete, |Moore maturity offset| at the visit the prior came from (years); widens the
    # timing prior when the distance-dependent prior is on. None means zeros.
    prior_distance: FloatArray | None = None

    @property
    def distance(self) -> FloatArray:
        if self.prior_distance is None:
            return np.zeros(self.n_athletes)
        return np.asarray(self.prior_distance, dtype=np.float64)

    @property
    def n_athletes(self) -> int:
        return len(self.athlete_ids)

    @property
    def n_observations(self) -> int:
        return int(self.age.size)

    @property
    def n_per_athlete(self) -> IntArray:
        return np.bincount(self.athlete_index, minlength=self.n_athletes).astype(np.int64)

    @property
    def mean_age(self) -> FloatArray:
        """Mean age at measurement, per athlete (pivot of the velocity deviation)."""
        totals = np.bincount(self.athlete_index, weights=self.age, minlength=self.n_athletes)
        return np.asarray(totals / self.n_per_athlete, dtype=np.float64)

    def padded(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(age, stature, mask)`` as (n_athletes, max_visits) arrays; mask is 1 where real."""
        counts = self.n_per_athlete
        shape = (self.n_athletes, int(counts.max()))
        age, stature, mask = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        order = np.argsort(self.athlete_index, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        column = np.arange(self.n_observations) - np.repeat(starts, counts)
        rows = self.athlete_index[order]
        age[rows, column] = self.age[order]
        stature[rows, column] = self.stature[order]
        mask[rows, column] = 1.0
        return age, stature, mask


def prepare_growth_data(
    athletes: pd.DataFrame, anthropometrics: pd.DataFrame, sex: Sex
) -> GrowthData:
    """Join athletes and visits for one sex and compute each athlete's Moore prior centre.

    The prior centre is the Moore APHV from the visit at which the athlete is estimated to be
    closest to PHV, because the equation's error grows with distance from PHV.
    """
    flags = set(athletes["synthetic"]) | set(anthropometrics["synthetic"])
    if len(flags) != 1:
        raise ValueError("Refusing to fit a mix of synthetic and real data")
    chosen = athletes.loc[athletes["sex"] == sex, ["id", "dob_month"]]
    frame = anthropometrics.merge(chosen, left_on="athlete_id", right_on="id", how="inner")
    if frame.empty:
        raise ValueError(f"No anthropometric data for sex '{sex}'")
    frame = frame.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    frame["age"] = [
        age_on(dob, pd.Timestamp(day).date())
        for dob, day in zip(frame["dob_month"], frame["date"], strict=True)
    ]
    frame["stature"] = frame[["stature_1", "stature_2"]].mean(axis=1)
    frame["sitting_height"] = frame[["sitting_height_1", "sitting_height_2"]].mean(axis=1)
    return growth_data_from_measurements(frame, sex, synthetic=bool(next(iter(flags))))


def growth_data_from_measurements(
    frame: pd.DataFrame,
    sex: Sex,
    *,
    synthetic: bool,
    prior_frame: pd.DataFrame | None = None,
) -> GrowthData:
    """Model-ready data from one sex's measurements, whatever their source.

    ``frame`` has columns ``athlete_id, age, stature, sitting_height``. Each athlete's prior
    centre is the Moore APHV from the visit at which the athlete is estimated to be closest to
    PHV, because the equation's error grows with distance from PHV. ``prior_frame`` (same
    columns) supplies the visits the prior is computed from, if they differ from the visits the
    curve is fitted to: a reference cohort fitted on full records but given priors from one
    season, as at deployment.
    """
    frame = frame.sort_values(["athlete_id", "age"]).reset_index(drop=True)
    source = frame if prior_frame is None else prior_frame.sort_values(["athlete_id", "age"])
    source = source.reset_index(drop=True)
    estimate = moore2015(
        sex,
        source["age"].to_numpy(),
        source["stature"].to_numpy(),
        source["sitting_height"].to_numpy(),
    )
    source = source.assign(moore_aphv=estimate.aphv, moore_distance=np.abs(estimate.mo))
    closest = source.loc[source.groupby("athlete_id")["moore_distance"].idxmin()]
    prior = closest.set_index("athlete_id")
    athlete_ids = tuple(str(a) for a in sorted(frame["athlete_id"].astype(str).unique()))
    missing = set(athlete_ids) - set(prior.index.astype(str))
    if missing:
        raise ValueError(f"No prior visits for athletes: {sorted(missing)[:5]}")
    prior.index = prior.index.astype(str)
    prior = prior.loc[list(athlete_ids)]
    codes = {athlete: k for k, athlete in enumerate(athlete_ids)}
    return GrowthData(
        athlete_ids=athlete_ids,
        athlete_index=np.asarray(frame["athlete_id"].astype(str).map(codes), dtype=np.int64),
        age=frame["age"].to_numpy(dtype=np.float64),
        stature=frame["stature"].to_numpy(dtype=np.float64),
        prior_aphv=prior["moore_aphv"].to_numpy(dtype=np.float64),
        sex=sex,
        synthetic=synthetic,
        prior_distance=prior["moore_distance"].to_numpy(dtype=np.float64),
    )


# --------------------------------------------------------------------------- grids


@dataclass(frozen=True)
class EffectGrids:
    """Standardised grids for timing and intensity with normalised log prior weights."""

    timing: FloatArray  # in units of the timing prior SD
    timing_log_weight: FloatArray
    intensity: FloatArray  # in units of sigma_gamma
    intensity_log_weight: FloatArray

    @property
    def timing_step(self) -> float:
        return float(self.timing[1] - self.timing[0])


def _normal_grid(half_width_sd: float, n_points: int) -> tuple[FloatArray, FloatArray]:
    nodes = np.linspace(-half_width_sd, half_width_sd, n_points)
    log_weight = -0.5 * nodes**2
    log_weight = log_weight - np.log(np.sum(np.exp(log_weight)))
    return nodes, log_weight


def effect_grids(config: Mapping[str, Any]) -> EffectGrids:
    """Grids from the ``marginalisation`` and ``intensity`` config blocks."""
    spec = config["marginalisation"]
    timing, timing_weight = _normal_grid(
        float(spec["timing_half_width_sd"]), int(spec["timing_points"])
    )
    if config["intensity"]["enabled"]:
        intensity, intensity_weight = _normal_grid(
            float(spec["intensity_half_width_sd"]), int(spec["intensity_points"])
        )
    else:
        intensity, intensity_weight = np.zeros(1), np.zeros(1)
    return EffectGrids(timing, timing_weight, intensity, intensity_weight)


def _grid_log_weights(
    xp: Any,
    *,
    age: Any,
    stature: Any,
    mask: Any,
    n_visits: Any,
    prior_aphv: Any,
    grids: EffectGrids,
    knots: tuple[float, ...],
    intercept: Any,
    linear: Any,
    coefs: Any,
    sigma: Any,
    sigma_alpha: Any,
    sigma_beta: Any,
    sigma_gamma: Any,
    sigma_delta: Any,
    timing_offset: Any,
    cohort_mean: Any = None,
    cohort_sd: Any = None,
    prior_scale: Any = None,
) -> tuple[Any, Any, Any]:
    """Log posterior weight of every (timing, intensity) grid cell for every athlete.

    Returns ``(log_weight, residual_sum, residual_slope_sum)``, each shaped (athletes, timing,
    intensity), with the two linear effects integrated out: size ``alpha_i`` and the velocity
    deviation ``delta_i * (age - mean_age_i)``. The two sums are what their conditional draws
    need. Runs on NumPy or PyTensor (``xp``); a leading draw dimension is not supported.

    Because ages are centred on each athlete's mean, the two effects are orthogonal and the
    marginal covariance ``var I + var_alpha 11' + var_delta cc'`` factorises term by term.
    """
    scale = 1.0 if prior_scale is None else prior_scale[:, None]  # distance-dependent prior SD
    beta = prior_aphv[:, None] + timing_offset + sigma_beta * scale * grids.timing[None, :]
    stretch = xp.exp(sigma_gamma * grids.intensity)  # (G,)
    z = (age[:, :, None, None] - beta[:, None, :, None]) * stretch[None, None, None, :]
    fitted = intercept + splines.spline_value(z, linear, coefs, knots, xp=xp)
    residual = (stature[:, :, None, None] - fitted) * mask[:, :, None, None]
    centred = (age - (age * mask).sum(axis=1, keepdims=True) / n_visits[:, None]) * mask
    total = residual.sum(axis=1)
    total_sq = (residual**2).sum(axis=1)
    slope_total = (residual * centred[:, :, None, None]).sum(axis=1)
    n = n_visits[:, None, None]
    spread = (centred**2).sum(axis=1)[:, None, None]
    var, var_alpha, var_delta = sigma**2, sigma_alpha**2, sigma_delta**2
    quadratic = (
        total_sq / var
        - var_alpha * total**2 / (var * (var + n * var_alpha))
        - var_delta * slope_total**2 / (var * (var + spread * var_delta))
    )
    log_det = (
        (n - 2.0) * xp.log(var) + xp.log(var + n * var_alpha) + xp.log(var + spread * var_delta)
    )
    log_likelihood = -0.5 * (quadratic + log_det + n * _LOG_2PI)
    log_weight = (
        log_likelihood
        + grids.timing_log_weight[None, :, None]
        + grids.intensity_log_weight[None, None, :]
    )
    if cohort_sd is not None:
        # Cohort-level prior: beta_i ~ Normal(cohort_mean, cohort_sd). Together with the term
        # above, the Moore estimate then acts as a noisy, possibly biased measurement of beta_i.
        standardised = (beta - cohort_mean) / cohort_sd
        cohort_term = -0.5 * standardised**2 - xp.log(cohort_sd) - 0.5 * _LOG_2PI
        log_weight = log_weight + cohort_term[:, :, None]
    return log_weight, total, slope_total


# --------------------------------------------------------------------------- model


def build_model(data: GrowthData, config: Mapping[str, Any]) -> pm.Model:
    """PyMC model over the shared parameters, with athlete effects integrated out."""
    knots = splines.check_knots(config["spline"]["knots_years"])
    priors = config["priors"]
    grids = effect_grids(config)
    transform = splines.orthogonalising_transform(knots)
    age, stature, mask = data.padded()
    coords = {"basis": list(range(splines.n_basis(knots))), "curve": list(range(len(transform)))}
    with pm.Model(coords=coords) as model:
        # Mean curve in an orthogonalised parameterisation (splines.orthogonalising_transform).
        theta = pm.Normal("theta", mu=0.0, sigma=priors["curve_sd_cm"], dims="curve")
        curve = pt.dot(transform, theta)
        intercept = pm.Deterministic("intercept", float(np.mean(data.stature)) + curve[0])
        linear = pm.Deterministic("linear", curve[1])
        coefs = pm.Deterministic("coefs", curve[2:], dims="basis")
        sigma = pm.HalfNormal("sigma", sigma=priors["residual_sd_cm"])
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=priors["size_sd_cm"])
        if config["intensity"]["enabled"]:
            sigma_gamma: Any = pm.HalfNormal("sigma_gamma", sigma=config["intensity"]["sd_prior"])
        else:
            sigma_gamma = 0.0
        deviation = config["velocity_deviation"]
        sigma_delta: Any = (
            pm.HalfNormal("sigma_delta", sigma=deviation["sd_prior"])
            if deviation["enabled"]
            else 0.0
        )
        timing = config["timing_prior"]
        timing_offset: Any = (
            pm.Normal("timing_offset", mu=0.0, sigma=timing["offset_sd_years"])
            if timing["learn_offset"]
            else 0.0
        )
        sigma_beta: Any = (
            pm.HalfNormal("sigma_beta", sigma=timing["sd_years"])
            if timing["learn_sd"]
            else float(timing["sd_years"])
        )
        slope = timing.get("distance_slope", {"enabled": False})
        prior_scale: Any = None
        if slope["enabled"]:
            sigma_beta_slope = pm.HalfNormal("sigma_beta_slope", sigma=slope["sd_prior"])
            prior_scale = 1.0 + sigma_beta_slope * data.distance
        cohort = timing["cohort_prior"]
        cohort_mean: Any = None
        cohort_sd: Any = None
        if cohort["enabled"]:
            cohort_mean = pm.Normal(
                "cohort_mean", mu=float(np.mean(data.prior_aphv)), sigma=cohort["mean_sd_years"]
            )
            cohort_sd = pm.HalfNormal("cohort_sd", sigma=cohort["sd_scale_years"])
        log_weight, _, _ = _grid_log_weights(
            pt,
            age=age,
            stature=stature,
            mask=mask,
            n_visits=data.n_per_athlete.astype(np.float64),
            prior_aphv=data.prior_aphv,
            grids=grids,
            knots=knots,
            intercept=intercept,
            linear=linear,
            coefs=coefs,
            sigma=sigma,
            sigma_alpha=sigma_alpha,
            sigma_beta=sigma_beta,
            sigma_gamma=sigma_gamma,
            sigma_delta=sigma_delta,
            timing_offset=timing_offset,
            cohort_mean=cohort_mean,
            cohort_sd=cohort_sd,
            prior_scale=prior_scale,
        )
        flat = log_weight.reshape((data.n_athletes, -1))
        pm.Potential("marginal_likelihood", pt.logsumexp(flat, axis=1).sum())
    return model


def initial_values(data: GrowthData, config: Mapping[str, Any]) -> dict[str, Any]:
    """Starting point: the mean curve fitted by least squares with every athlete at their prior.

    With ``beta_i`` at the Moore prior centre and no size or intensity effects, the model is
    linear in the curve parameters, so all chains can start from the same sensible curve.
    """
    knots = splines.check_knots(config["spline"]["knots_years"])
    z = data.age - data.prior_aphv[data.athlete_index]
    design = np.column_stack([np.ones_like(z), z, *splines.basis(z, knots)])
    transform = splines.orthogonalising_transform(knots)
    theta, *_ = np.linalg.lstsq(
        design @ transform, data.stature - float(np.mean(data.stature)), rcond=None
    )
    return {"theta": theta}


def fit(
    data: GrowthData,
    config: Mapping[str, Any],
    *,
    seed: int,
    progressbar: bool = False,
) -> InferenceData:
    """Sample the shared parameters by NUTS (default) or approximate them by ADVI."""
    settings = config["inference"]
    with build_model(data, config):
        if settings["method"] == "advi":
            approximation = pm.fit(
                n=int(settings["advi_iterations"]),
                method="advi",
                random_seed=seed,
                progressbar=progressbar,
            )
            return approximation.sample(int(settings["draws"]), random_seed=seed)
        if settings["method"] != "nuts":
            raise ValueError(f"Unknown inference method '{settings['method']}'")
        return pm.sample(
            draws=int(settings["draws"]),
            tune=int(settings["tune"]),
            chains=int(settings["chains"]),
            cores=int(settings.get("cores", settings["chains"])),
            target_accept=float(settings["target_accept"]),
            init=str(settings["init"]),
            initvals=initial_values(data, config),
            random_seed=seed,
            progressbar=progressbar,
        )


# --------------------------------------------------------------------------- athlete effects


@dataclass(frozen=True)
class SharedDraws:
    """Posterior draws of the shared parameters (S draws)."""

    intercept: FloatArray
    linear: FloatArray
    coefs: FloatArray  # (S, K - 2)
    sigma: FloatArray
    sigma_alpha: FloatArray
    sigma_gamma: FloatArray
    sigma_delta: FloatArray
    timing_offset: FloatArray
    sigma_beta: FloatArray
    cohort_mean: FloatArray  # NaN when the cohort prior is disabled
    cohort_sd: FloatArray  # NaN when the cohort prior is disabled
    sigma_beta_slope: FloatArray | None = None  # zeros when the prior SD does not grow

    @property
    def n_draws(self) -> int:
        return int(self.sigma.size)

    @property
    def slope(self) -> FloatArray:
        if self.sigma_beta_slope is None:
            return np.zeros(self.n_draws)
        return np.asarray(self.sigma_beta_slope, dtype=np.float64)


def shared_draws(
    idata: InferenceData, n_draws: int, seed: int, config: Mapping[str, Any]
) -> SharedDraws:
    """Stack chains and keep a random subset of ``n_draws`` draws.

    Parameters that were switched off in ``config`` are filled with their fixed values.
    """
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    total = int(stacked.sizes["sample"])
    keep = np.sort(np.random.default_rng(seed).choice(total, min(n_draws, total), replace=False))

    def draws(name: str) -> FloatArray:
        values = np.asarray(stacked[name].values, dtype=np.float64)
        return values[keep] if values.ndim == 1 else values.T[keep]

    sigma = draws("sigma")

    def optional(name: str, fixed: float) -> FloatArray:
        return draws(name) if name in stacked else np.full_like(sigma, fixed)

    return SharedDraws(
        intercept=draws("intercept"),
        linear=draws("linear"),
        coefs=draws("coefs"),
        sigma=sigma,
        sigma_alpha=draws("sigma_alpha"),
        sigma_gamma=optional("sigma_gamma", 0.0),
        sigma_delta=optional("sigma_delta", 0.0),
        timing_offset=optional("timing_offset", 0.0),
        sigma_beta=optional("sigma_beta", float(config["timing_prior"]["sd_years"])),
        cohort_mean=optional("cohort_mean", float("nan")),
        cohort_sd=optional("cohort_sd", float("nan")),
        sigma_beta_slope=optional("sigma_beta_slope", 0.0),
    )


def draw_athlete_effects(
    shared: SharedDraws, data: GrowthData, config: Mapping[str, Any], *, seed: int
) -> SitarPosterior:
    """Draw (alpha, beta, gamma, delta) for every athlete, exactly, given each shared draw.

    For each shared draw: pick a (timing, intensity) grid cell with probability proportional to
    its posterior weight, jitter timing uniformly within the cell, then draw ``alpha_i``
    and ``delta_i`` from their independent Normal conditionals. Two-peaked timing posteriors
    come out two-peaked.
    """
    knots = splines.check_knots(config["spline"]["knots_years"])
    grids = effect_grids(config)
    age, stature, mask = data.padded()
    n_visits = data.n_per_athlete.astype(np.float64)
    mean_age = data.mean_age
    spread = (((age - mean_age[:, None]) * mask) ** 2).sum(axis=1)
    rng = np.random.default_rng(seed)
    n, n_timing, n_intensity = data.n_athletes, grids.timing.size, grids.intensity.size
    alpha, beta, gamma, delta = (np.empty((shared.n_draws, n)) for _ in range(4))
    use_cohort = bool(np.isfinite(shared.cohort_sd).all())
    rows = np.arange(n)
    slopes, distance = shared.slope, data.distance
    for s in range(shared.n_draws):
        sigma_beta = shared.sigma_beta[s]
        prior_scale = 1.0 + slopes[s] * distance
        log_weight, residual_sum, slope_sum = _grid_log_weights(
            np,
            age=age,
            stature=stature,
            mask=mask,
            n_visits=n_visits,
            prior_aphv=data.prior_aphv,
            grids=grids,
            knots=knots,
            intercept=shared.intercept[s],
            linear=shared.linear[s],
            coefs=shared.coefs[s],
            sigma=shared.sigma[s],
            sigma_alpha=shared.sigma_alpha[s],
            sigma_beta=sigma_beta,
            sigma_gamma=shared.sigma_gamma[s],
            sigma_delta=shared.sigma_delta[s],
            timing_offset=shared.timing_offset[s],
            cohort_mean=shared.cohort_mean[s] if use_cohort else None,
            cohort_sd=shared.cohort_sd[s] if use_cohort else None,
            prior_scale=prior_scale,
        )
        flat = log_weight.reshape(n, -1)
        probability = np.exp(flat - flat.max(axis=1, keepdims=True))
        cumulative = np.cumsum(probability, axis=1)
        cell = (cumulative < rng.random((n, 1)) * cumulative[:, -1:]).sum(axis=1)
        cell = np.minimum(cell, n_timing * n_intensity - 1)
        timing_cell, intensity_cell = np.divmod(cell, n_intensity)
        jitter = rng.uniform(-0.5, 0.5, n) * grids.timing_step
        centre = data.prior_aphv + shared.timing_offset[s]
        beta[s] = centre + sigma_beta * prior_scale * (grids.timing[timing_cell] + jitter)
        gamma[s] = shared.sigma_gamma[s] * grids.intensity[intensity_cell]
        var, var_alpha = shared.sigma[s] ** 2, shared.sigma_alpha[s] ** 2
        picked = (rows, timing_cell, intensity_cell)
        alpha_var = 1.0 / (n_visits / var + 1.0 / var_alpha)
        alpha[s] = alpha_var * residual_sum[picked] / var + np.sqrt(
            alpha_var
        ) * rng.standard_normal(n)
        var_delta = shared.sigma_delta[s] ** 2
        if var_delta > 0:
            delta_var = 1.0 / (spread / var + 1.0 / var_delta)
            delta[s] = delta_var * slope_sum[picked] / var + np.sqrt(
                delta_var
            ) * rng.standard_normal(n)
        else:
            delta[s] = 0.0
    return SitarPosterior(
        athlete_ids=data.athlete_ids,
        knots=knots,
        intercept=shared.intercept,
        linear=shared.linear,
        coefs=shared.coefs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sigma=shared.sigma,
        sigma_alpha=shared.sigma_alpha,
        sigma_gamma=shared.sigma_gamma,
        delta=delta,
        sigma_delta=shared.sigma_delta,
        timing_offset=shared.timing_offset,
        sigma_beta=shared.sigma_beta,
        cohort_mean=shared.cohort_mean,
        cohort_sd=shared.cohort_sd,
        mean_age=mean_age,
        sex=data.sex,
        synthetic=data.synthetic,
        sigma_beta_slope=shared.slope,
    )


def extract_posterior(
    idata: InferenceData, data: GrowthData, config: Mapping[str, Any], *, seed: int
) -> SitarPosterior:
    """Full posterior (shared draws + athlete effects) from a fit."""
    n_draws = int(config["growth_state"]["n_posterior_draws"])
    shared = shared_draws(idata, n_draws, seed, config)
    return draw_athlete_effects(shared, data, config, seed=seed)


def update(
    previous: SitarPosterior, data: GrowthData, config: Mapping[str, Any], *, seed: int
) -> SitarPosterior:
    """Incremental update without refitting the shared curve.

    Redraws every athlete's effects against the stored shared draws using ``data`` (which should
    contain all measurements to date, including new visits and new athletes). This is exact
    given the shared draws, and approximate only in that the shared curve is not re-estimated;
    refit with ``fit`` periodically (e.g. when the cohort's data have grown substantially).
    """
    if previous.sex != data.sex:
        raise ValueError("Cannot update a posterior with data from the other sex")
    if previous.synthetic != data.synthetic:
        raise ValueError("Cannot update across synthetic and real data")
    shared = SharedDraws(
        intercept=previous.intercept,
        linear=previous.linear,
        coefs=previous.coefs,
        sigma=previous.sigma,
        sigma_alpha=previous.sigma_alpha,
        sigma_gamma=previous.sigma_gamma,
        sigma_delta=previous.sigma_delta,
        timing_offset=previous.timing_offset,
        sigma_beta=previous.sigma_beta,
        cohort_mean=previous.cohort_mean,
        cohort_sd=previous.cohort_sd,
        sigma_beta_slope=previous.sigma_beta_slope,
    )
    return draw_athlete_effects(shared, data, config, seed=seed)


def diagnostics(idata: InferenceData) -> dict[str, float]:
    """Convergence summary for the sampled parameters: worst R-hat, smallest ESS, divergences."""
    sampled = (
        "theta",
        "sigma",
        "sigma_alpha",
        "sigma_gamma",
        "sigma_delta",
        "sigma_beta_slope",
        "timing_offset",
        "sigma_beta",
        "cohort_mean",
        "cohort_sd",
    )
    names = [n for n in sampled if n in idata.posterior]
    summary = az.summary(idata, var_names=names, kind="diagnostics")
    stats = getattr(idata, "sample_stats", None)
    divergences = int(stats["diverging"].sum()) if stats is not None and "diverging" in stats else 0
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergences": float(divergences),
    }
