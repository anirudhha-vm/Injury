"""Khamis-Roche percentage of predicted adult height (optional maturity indicator).

Needs the age- and sex-specific coefficient table from Khamis & Roche (1994), Pediatrics
94:504-507 (plus the published erratum), and a correction for self-reported parental height.

TODO(human): obtain the coefficient table from the original paper and add it to
``configs/growth/khamis_roche.yaml``. Coefficients must not be reconstructed from memory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def percent_predicted_adult_height(
    sex: str,
    age: ArrayLike,
    stature: ArrayLike,
    mass: ArrayLike,
    mid_parent_height: ArrayLike,
) -> NDArray[np.float64]:
    """Not available until a human supplies the Khamis-Roche coefficient table."""
    raise NotImplementedError(
        "Khamis-Roche coefficients are not in the repository. TODO(human): obtain the table "
        "from Khamis & Roche (1994) Pediatrics 94:504-507 before using this indicator."
    )
