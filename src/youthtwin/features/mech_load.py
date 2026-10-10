"""Mechanistic load PROXIES for apophyseal traction (contribution C4).

These are geometric, quasi-static stand-ins, **not** measured or validated tendon loads. They
ignore the dynamics of landing (ground reaction forces are several times body weight and are
not measured by a phone) and muscle co-contraction. They are meant to encode how body mass and
segment proportions change the lever arithmetic as a child grows. Report them as proxies until
they have been compared with force-plate data.

Definitions
-----------
- **Knee-extensor moment proxy** (N m) = ``mass * g * thigh_length * sin(thigh inclination)``:
  the moment about the knee of body weight acting at the hip, with the thigh inclined from
  vertical by the given angle.
- **Patellar-tendon load proxy** (N) = knee moment proxy / shank length. The true patellar-tendon
  moment arm is roughly proportional to segment length; dividing by shank length gives tendon
  force *up to that unknown proportionality constant*.
- **Achilles load proxy** (N) = ``mass * g * (forefoot lever / heel lever)`` using the horizontal
  ankle-to-toe and ankle-to-heel distances from the side view, again up to a constant.

No anatomical constants are assumed: an unknown constant factor rescales a proxy without
changing its ranking of athletes, and the models standardise their inputs. Do not interpret
the absolute values in newtons as real tendon forces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

STANDARD_GRAVITY = 9.80665  # m/s^2 (defined constant)
_CM_TO_M = 0.01
MECH_PREFIX = "mech_"


def _array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def knee_extensor_moment_proxy(
    mass_kg: ArrayLike, thigh_length_m: ArrayLike, thigh_inclination_deg: ArrayLike
) -> FloatArray:
    """Quasi-static knee moment proxy in N m."""
    lever = _array(thigh_length_m) * np.sin(np.radians(_array(thigh_inclination_deg)))
    return np.asarray(_array(mass_kg) * STANDARD_GRAVITY * lever, dtype=np.float64)


def patellar_tendon_load_proxy(
    knee_moment_proxy: ArrayLike, shank_length_m: ArrayLike
) -> FloatArray:
    """Knee moment proxy divided by shank length (N, up to a constant)."""
    return np.asarray(_array(knee_moment_proxy) / _array(shank_length_m), dtype=np.float64)


def achilles_load_proxy(mass_kg: ArrayLike, forefoot_heel_lever_ratio: ArrayLike) -> FloatArray:
    """Body weight times forefoot lever / heel lever (N, up to a constant)."""
    return np.asarray(
        _array(mass_kg) * STANDARD_GRAVITY * _array(forefoot_heel_lever_ratio), dtype=np.float64
    )


def mech_proxies(
    *,
    mass_kg: ArrayLike,
    leg_length_cm: ArrayLike,
    thigh_length_fraction: ArrayLike,
    thigh_inclination_deg: ArrayLike,
    forefoot_heel_lever_ratio: ArrayLike,
) -> dict[str, FloatArray]:
    """All proxies from anthropometry (mass, leg length) and side-view screening features.

    ``leg_length_cm`` is stature minus sitting height; it is split into thigh and shank by the
    fraction seen in the side view. Also returns body-weight-normalised versions, which isolate
    the effect of proportions and posture from the effect of mass.
    """
    mass = _array(mass_kg)
    leg = _array(leg_length_cm) * _CM_TO_M
    fraction = _array(thigh_length_fraction)
    thigh, shank = leg * fraction, leg * (1.0 - fraction)
    moment = knee_extensor_moment_proxy(mass, thigh, thigh_inclination_deg)
    patellar = patellar_tendon_load_proxy(moment, shank)
    achilles = achilles_load_proxy(mass, forefoot_heel_lever_ratio)
    weight = mass * STANDARD_GRAVITY
    return {
        f"{MECH_PREFIX}knee_extensor_moment_proxy": moment,
        f"{MECH_PREFIX}patellar_tendon_load_proxy": patellar,
        f"{MECH_PREFIX}achilles_load_proxy": achilles,
        f"{MECH_PREFIX}patellar_tendon_load_proxy_bw": np.asarray(patellar / weight),
        f"{MECH_PREFIX}achilles_load_proxy_bw": np.asarray(achilles / weight),
    }


def add_mech_proxies(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Add proxy columns to an athlete-week table when the source columns are present.

    ``config["sources"]`` names the screening columns to use. If any source column is absent
    (for example the simplified synthetic cohort has no side-view features) the table is
    returned unchanged; the caller can check for ``mech_`` columns.
    """
    sources = config["sources"]
    needed = ["mass", "leg_length", *sources.values()]
    if any(column not in frame.columns for column in needed):
        return frame
    proxies = mech_proxies(
        mass_kg=frame["mass"],
        leg_length_cm=frame["leg_length"],
        thigh_length_fraction=frame[sources["thigh_length_fraction"]],
        thigh_inclination_deg=frame[sources["thigh_inclination"]],
        forefoot_heel_lever_ratio=frame[sources["forefoot_heel_lever_ratio"]],
    )
    return frame.assign(**proxies)
