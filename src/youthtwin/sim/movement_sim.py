"""Synthetic screening summary features (minimal; skeleton sequences are not simulated yet).

Each athlete has a stable standardised trait per feature. The latent value at any time is
``trait + maturity_shift * sigmoid(true maturity offset)``; a screening session observes it
with session noise and on the feature's own scale. The latent ``knee_extensor_load_proxy``
drives the simulated growth-related injury hazard.

Synthetic by definition; parameters are ASSUMPTIONS from ``configs/sim/cohort.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LOAD_PROXY = "knee_extensor_load_proxy"


def sample_traits(
    rng: np.random.Generator, n_athletes: int, config: Mapping[str, Any]
) -> dict[str, FloatArray]:
    """One standardised trait per athlete per feature."""
    return {name: rng.normal(0.0, config["trait_sd"], n_athletes) for name in config["features"]}


def latent_value(
    trait: FloatArray, maturity_offset: FloatArray, config: Mapping[str, Any]
) -> FloatArray:
    """Noise-free standardised value given the true maturity offset (years)."""
    shift = config["maturity_shift"] / (1.0 + np.exp(-maturity_offset))
    return np.asarray(trait + shift, dtype=np.float64)


def observe_feature(
    rng: np.random.Generator,
    name: str,
    trait: FloatArray,
    maturity_offset: FloatArray,
    config: Mapping[str, Any],
) -> FloatArray:
    """Session-level observed value on the feature's scale."""
    spec = config["features"][name]
    noisy = latent_value(trait, maturity_offset, config) + rng.normal(
        0.0, config["session_noise_sd"], trait.shape
    )
    return np.asarray(spec["mean"] + spec["scale"] * noisy, dtype=np.float64)


# ------------------------------------------------------------------------------ trial sequences


def session_trials(
    rng: np.random.Generator, load_proxy: float, config: Mapping[str, Any]
) -> list[Any]:
    """Synthetic drop-jump trials for one screening session.

    The session's (standardised) load proxy sets how the athlete lands: a higher proxy means a
    stiffer landing with less knee flexion and more frontal-plane knee angle. This puts the same
    information into the skeleton sequence that the tabular screening feature carries, so that
    a sequence encoder has something to find. The mapping is an ASSUMPTION of the simulation
    (``movement.sequences`` in ``configs/sim/cohort.yaml``), not a biomechanical claim.
    """
    from youthtwin.sim import skeleton_sim  # local import: keeps the cohort generator light

    spec = config["sequences"]
    trials = []
    for _ in range(int(spec["trials_per_session"])):
        flexion = (
            spec["peak_knee_flexion_mean"]
            - spec["peak_knee_flexion_per_load"] * load_proxy
            + rng.normal(0.0, spec["trial_noise_deg"])
        )
        fppa = spec["peak_fppa_mean"] + spec["peak_fppa_per_load"] * load_proxy
        low, high = spec["peak_knee_flexion_range"]
        trials.append(
            skeleton_sim.simulate_drop_vertical_jump(
                rng,
                fps=float(spec["fps"]),
                peak_knee_flexion=float(np.clip(flexion, low, high)),
                peak_fppa=(float(fppa), float(fppa)),
                noise_px=float(spec["noise_px"]),
            )
        )
    return trials
