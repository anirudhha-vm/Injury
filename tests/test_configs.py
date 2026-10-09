"""Config skeleton: every YAML loads, and values fixed by the brief are encoded correctly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from youthtwin.utils.config import ROOT_ENV_VAR, configs_dir, load_config, repo_root

ALL_CONFIGS = sorted(configs_dir().rglob("*.yaml"))

EXPECTED_CONFIGS = {
    "growth/equations.yaml",
    "growth/sitar.yaml",
    "pose/estimator.yaml",
    "pose/filter.yaml",
    "pose/tasks.yaml",
    "pose/skeleton.yaml",
    "features/task_features.yaml",
    "features/mech.yaml",
    "models/tcn_film.yaml",
    "models/stgcn_film.yaml",
    "models/baselines.yaml",
    "models/hazard.yaml",
    "eval/splits.yaml",
    "eval/metrics.yaml",
    "eval/ablations.yaml",
    "sim/cohort.yaml",
}


def _walk_keys(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [str(k) for k in node] + [key for v in node.values() for key in _walk_keys(v)]
    if isinstance(node, list):
        return [key for item in node for key in _walk_keys(item)]
    return []


def test_all_expected_configs_exist() -> None:
    present = {p.relative_to(configs_dir()).as_posix() for p in ALL_CONFIGS}
    assert present >= EXPECTED_CONFIGS


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_config_loads_as_mapping(path: Path) -> None:
    assert load_config(path)


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
    listing = tmp_path / "list.yaml"
    listing.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(listing)


def test_repo_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    assert repo_root() == tmp_path.resolve()


def test_repo_root_default_contains_pyproject() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_blazepose_lower_limb_indices() -> None:
    """v2 defect: knee used 11/13/15 and ankle 13/14 (upper-limb landmarks)."""
    landmarks = load_config("pose/skeleton.yaml")["blazepose"]["landmarks"]
    assert (landmarks["left_hip"], landmarks["right_hip"]) == (23, 24)
    assert (landmarks["left_knee"], landmarks["right_knee"]) == (25, 26)
    assert (landmarks["left_ankle"], landmarks["right_ankle"]) == (27, 28)
    assert (landmarks["left_heel"], landmarks["right_heel"]) == (29, 30)
    assert (landmarks["left_foot_index"], landmarks["right_foot_index"]) == (31, 32)


def test_equation_families_are_separate_and_named() -> None:
    """v2 defect: Moore girls equation was labelled Mirwald and mixed with Mirwald boys."""
    equations = load_config("growth/equations.yaml")
    assert equations["primary_family"] == "moore2015"
    assert set(equations["moore2015"]) == {"boys", "girls", "boys_stature"}
    assert set(equations["mirwald2002"]) == {"boys", "girls"}
    assert (
        equations["moore2015"]["girls"]["intercept"]
        != equations["mirwald2002"]["girls"]["intercept"]
    )


def test_ablation_grid_has_no_expected_results() -> None:
    """v2 defect: an 'Expected AUC' column. Config may name ablations, never their results."""
    ablations = load_config("eval/ablations.yaml")
    assert [a["id"] for a in ablations["ablations"]] == list(range(1, 10))
    forbidden = ("auc", "expected", "result", "score")
    assert not [k for k in _walk_keys(ablations) if any(word in k.lower() for word in forbidden)]


def test_integrity_switches() -> None:
    assert load_config("eval/splits.yaml")["group_key"] == "athlete_id"
    assert load_config("eval/splits.yaml")["scalers"] == "fit_on_train_fold_only"
    assert load_config("models/hazard.yaml")["class_imbalance"]["smote"] is False
    assert load_config("sim/cohort.yaml")["synthetic"] is True
    assert load_config("features/mech.yaml")["validated_against_force_plates"] is False
