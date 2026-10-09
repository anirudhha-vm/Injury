"""FAILS BY DESIGN until a human verifies the Mirwald (2002) girls coefficients.

Excluded from `make test`; run with `make gate`.
"""

from __future__ import annotations

import pytest

from youthtwin.growth.verification import is_verified, verification_message


@pytest.mark.human_gate
def test_mirwald_girls_coefficients_verified_by_human() -> None:
    assert is_verified("mirwald_girls"), verification_message("mirwald_girls")
