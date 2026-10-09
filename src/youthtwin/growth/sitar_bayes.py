"""Hierarchical Bayesian SITAR-style growth model (contribution C1), marginalised form.

    stature_ij ~ Normal(intercept + alpha_i + h((age_ij - beta_i) * exp(gamma_i)), sigma)

- ``h`` is a shared natural cubic spline (``youthtwin.growth.splines``).
- ``alpha_i`` (size, cm) ~ Normal(0, sigma_alpha).
- ``beta_i`` (timing, years) ~ Normal(Moore-2015 APHV of athlete i, sd_years). Moore's estimate
  is biased toward the mean (see ``maturity_equations``); the prior SD lets the longitudinal
  data pull each athlete away from it.
- ``gamma_i`` (intensity, log scale) ~ Normal(0, sigma_gamma); optional.
- One model per sex.

Why the per-athlete effects are integrated out rather than sampled
------------------------------------------------------------------
With about one season of data an athlete's timing is often genuinely two-peaked: a given growth
speed fits either the rising or the falling side of the growth spurt, roughly two years apart.
A gradient-based sampler cannot move between those peaks, so sampling ``beta_i`` directly gives
chains that disagree (observed during development; see docs/CHANGELOG_REASONING.md).

Instead:

1. ``alpha_i`` is integrated out in closed form (the model is linear-Gaussian in it).
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

    @property
    def n_athletes(self) -> int:
        return len(self.athlete_ids)

    @property
    def n_observations(self) -> int:
        return int(self.age.size)

    @property
    def n_per_athlete(self) -> IntArray:
        return np.bincount(self.athlete_index, minlength=self.n_athletes).astype(np.int64)

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
    estimate = moore2015(
        sex,
        frame["age"].to_numpy(),
        frame["stature"].to_numpy(),
        frame["sitting_height"].to_numpy(),
    )
    frame["moore_aphv"] = estimate.aphv
    frame["moore_distance"] = np.abs(estimate.mo)
    closest = frame.loc[frame.groupby("athlete_id")["moore_distance"].idxmin()]
    prior = closest.set_index("athlete_id")["moore_aphv"]
    athlete_ids = tuple(str(a) for a in prior.index)
    codes = {athlete: k for k, athlete in enumerate(athlete_ids)}
    return GrowthData(
        athlete_ids=athlete_ids,
        athlete_index=np.asarray(frame["athlete_id"].map(codes), dtype=np.int64),
        age=frame["age"].to_numpy(dtype=np.float64),
        stature=frame["stature"].to_numpy(dtype=np.float64),
        prior_aphv=prior.to_numpy(dtype=np.float64),
        sex=sex,
        synthetic=bool(next(iter(flags))),
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
) -> tuple[Any, Any]:
    """Log posterior weight of every (timing, intensity) grid cell for every athlete.

    Returns ``(log_weight, residual_sum)``, both shaped (athletes, timing, intensity), with
    ``alpha_i`` integrated out. ``residual_sum`` is what the conditional draw of ``alpha_i``
    needs. Runs on NumPy or PyTensor (``xp``); a leading draw dimension is not supported.
    """
    beta = prior_aphv[:, None] + sigma_beta * grids.timing[None, :]  # (n, B)
    stretch = xp.exp(sigma_gamma * grids.intensity)  # (G,)
    z = (age[:, :, None, None] - beta[:, None, :, None]) * stretch[None, None, None, :]
    fitted = intercept + splines.spline_value(z, linear, coefs, knots, xp=xp)
    residual = (stature[:, :, None, None] - fitted) * mask[:, :, None, None]
    total = residual.sum(axis=1)
    total_sq = (residual**2).sum(axis=1)
    n = n_visits[:, None, None]
    var, var_alpha = sigma**2, sigma_alpha**2
    # y_i - mu_i ~ Normal(0, var * I + var_alpha * 11'), written out for compound symmetry.
    quadratic = total_sq / var - var_alpha * total**2 / (var * (var + n * var_alpha))
    log_det = (n - 1.0) * xp.log(var) + xp.log(var + n * var_alpha)
    log_likelihood = -0.5 * (quadratic + log_det + n * _LOG_2PI)
    log_weight = (
        log_likelihood
        + grids.timing_log_weight[None, :, None]
        + grids.intensity_log_weight[None, None, :]
    )
    return log_weight, total


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
        log_weight, _ = _grid_log_weights(
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
            sigma_beta=float(config["timing_prior"]["sd_years"]),
            sigma_gamma=sigma_gamma,
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

    @property
    def n_draws(self) -> int:
        return int(self.sigma.size)


def shared_draws(idata: InferenceData, n_draws: int, seed: int) -> SharedDraws:
    """Stack chains and keep a random subset of ``n_draws`` draws."""
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    total = int(stacked.sizes["sample"])
    keep = np.sort(np.random.default_rng(seed).choice(total, min(n_draws, total), replace=False))

    def draws(name: str) -> FloatArray:
        values = np.asarray(stacked[name].values, dtype=np.float64)
        return values[keep] if values.ndim == 1 else values.T[keep]

    sigma = draws("sigma")
    return SharedDraws(
        intercept=draws("intercept"),
        linear=draws("linear"),
        coefs=draws("coefs"),
        sigma=sigma,
        sigma_alpha=draws("sigma_alpha"),
        sigma_gamma=draws("sigma_gamma") if "sigma_gamma" in stacked else np.zeros_like(sigma),
    )


def draw_athlete_effects(
    shared: SharedDraws, data: GrowthData, config: Mapping[str, Any], *, seed: int
) -> SitarPosterior:
    """Draw (alpha, beta, gamma) for every athlete, exactly, given each shared draw.

    For each shared draw: pick a (timing, intensity) grid cell with probability proportional to
    its posterior weight, jitter timing uniformly within the cell, then draw ``alpha_i`` from
    its Normal conditional. Two-peaked timing posteriors come out two-peaked.
    """
    knots = splines.check_knots(config["spline"]["knots_years"])
    grids = effect_grids(config)
    sigma_beta = float(config["timing_prior"]["sd_years"])
    age, stature, mask = data.padded()
    n_visits = data.n_per_athlete.astype(np.float64)
    rng = np.random.default_rng(seed)
    n, n_timing, n_intensity = data.n_athletes, grids.timing.size, grids.intensity.size
    alpha, beta, gamma = (np.empty((shared.n_draws, n)) for _ in range(3))
    rows = np.arange(n)
    for s in range(shared.n_draws):
        log_weight, residual_sum = _grid_log_weights(
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
        )
        flat = log_weight.reshape(n, -1)
        probability = np.exp(flat - flat.max(axis=1, keepdims=True))
        cumulative = np.cumsum(probability, axis=1)
        cell = (cumulative < rng.random((n, 1)) * cumulative[:, -1:]).sum(axis=1)
        cell = np.minimum(cell, n_timing * n_intensity - 1)
        timing_cell, intensity_cell = np.divmod(cell, n_intensity)
        jitter = rng.uniform(-0.5, 0.5, n) * grids.timing_step
        beta[s] = data.prior_aphv + sigma_beta * (grids.timing[timing_cell] + jitter)
        gamma[s] = shared.sigma_gamma[s] * grids.intensity[intensity_cell]
        var, var_alpha = shared.sigma[s] ** 2, shared.sigma_alpha[s] ** 2
        conditional_var = 1.0 / (n_visits / var + 1.0 / var_alpha)
        conditional_mean = conditional_var * residual_sum[rows, timing_cell, intensity_cell] / var
        alpha[s] = conditional_mean + np.sqrt(conditional_var) * rng.standard_normal(n)
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
        sex=data.sex,
        synthetic=data.synthetic,
    )


def extract_posterior(
    idata: InferenceData, data: GrowthData, config: Mapping[str, Any], *, seed: int
) -> SitarPosterior:
    """Full posterior (shared draws + athlete effects) from a fit."""
    n_draws = int(config["growth_state"]["n_posterior_draws"])
    return draw_athlete_effects(shared_draws(idata, n_draws, seed), data, config, seed=seed)


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
    )
    return draw_athlete_effects(shared, data, config, seed=seed)


def diagnostics(idata: InferenceData) -> dict[str, float]:
    """Convergence summary for the sampled parameters: worst R-hat, smallest ESS, divergences."""
    names = [n for n in ("theta", "sigma", "sigma_alpha", "sigma_gamma") if n in idata.posterior]
    summary = az.summary(idata, var_names=names, kind="diagnostics")
    stats = getattr(idata, "sample_stats", None)
    divergences = int(stats["diverging"].sum()) if stats is not None and "diverging" in stats else 0
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergences": float(divergences),
    }
