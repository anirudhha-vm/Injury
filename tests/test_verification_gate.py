"""The coefficient verification guard: real-data runs raise, synthetic runs warn."""

from __future__ import annotations

from pathlib import Path

import pytest

from youthtwin.growth.verification import (
    UnverifiedCoefficientsError,
    UnverifiedCoefficientsWarning,
    is_verified,
    require_verified,
    verification_message,
)


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "equations.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_unverified_real_data_run_raises_with_clear_message(tmp_path: Path) -> None:
    config = _config(tmp_path, "mirwald_girls_verified: false\n")
    with pytest.raises(UnverifiedCoefficientsError, match=r"TODO\(human\)"):
        require_verified("mirwald_girls", synthetic=False, config_path=config)


def test_unverified_synthetic_run_warns_but_continues(tmp_path: Path) -> None:
    config = _config(tmp_path, "mirwald_girls_verified: false\n")
    with pytest.warns(UnverifiedCoefficientsWarning, match="mirwald_girls_verified: true"):
        require_verified("mirwald_girls", synthetic=True, config_path=config)


def test_verified_passes_silently(tmp_path: Path, recwarn: pytest.WarningsRecorder) -> None:
    config = _config(tmp_path, "mirwald_girls_verified: true\n")
    assert is_verified("mirwald_girls", config)
    require_verified("mirwald_girls", synthetic=False, config_path=config)
    assert len(recwarn) == 0


def test_missing_flag_is_an_error(tmp_path: Path) -> None:
    config = _config(tmp_path, "something_else: 1\n")
    with pytest.raises(KeyError, match="mirwald_girls_verified"):
        is_verified("mirwald_girls", config)


def test_non_boolean_flag_is_an_error(tmp_path: Path) -> None:
    config = _config(tmp_path, "mirwald_girls_verified: 'yes'\n")
    with pytest.raises(TypeError, match="boolean"):
        is_verified("mirwald_girls", config)


def test_message_names_the_config_file() -> None:
    assert "configs/growth/equations.yaml" in verification_message("mirwald_girls")


def test_repo_config_has_the_gate_flag() -> None:
    assert isinstance(is_verified("mirwald_girls"), bool)
