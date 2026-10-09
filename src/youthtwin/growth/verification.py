"""Human-verification gate for published coefficients.

Some coefficient sets were transcribed from secondary sources and must be checked by a human
against the original paper before any real-data run. Each has a ``<name>_verified`` flag in
``configs/growth/equations.yaml`` that only a human may set to ``true``.

Synthetic runs are allowed while a flag is false (with a warning) so development is not blocked;
real-data runs raise.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from youthtwin.utils.config import load_config

EQUATIONS_CONFIG = "growth/equations.yaml"


class UnverifiedCoefficientsError(RuntimeError):
    """Raised when a real-data run would use coefficients no human has verified."""


class UnverifiedCoefficientsWarning(UserWarning):
    """Emitted when a synthetic run uses coefficients no human has verified."""


def verification_message(name: str) -> str:
    return (
        f"Coefficients for '{name}' have not been verified against the original paper. "
        f"TODO(human): check them against the source PDF, then set "
        f"`{name}_verified: true` in configs/{EQUATIONS_CONFIG}."
    )


def is_verified(name: str, config_path: str | Path = EQUATIONS_CONFIG) -> bool:
    """Return the ``<name>_verified`` flag; a missing or non-boolean flag is an error."""
    key = f"{name}_verified"
    config = load_config(config_path)
    if key not in config:
        raise KeyError(f"No verification flag '{key}' in {config_path}")
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"'{key}' must be a boolean, got {value!r}")
    return value


def require_verified(
    name: str, *, synthetic: bool, config_path: str | Path = EQUATIONS_CONFIG
) -> None:
    """Raise on real data, warn on synthetic data, if ``name`` is still unverified."""
    if is_verified(name, config_path):
        return
    if synthetic:
        warnings.warn(verification_message(name), UnverifiedCoefficientsWarning, stacklevel=2)
        return
    raise UnverifiedCoefficientsError(verification_message(name))
