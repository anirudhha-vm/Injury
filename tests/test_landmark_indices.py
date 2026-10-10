"""Landmark indices (v2 regression): one source of truth, correct lower-limb indices."""

from __future__ import annotations

import pytest

from youthtwin.pose.skeletons import BLAZEPOSE, HALPE26, JOINTS, SIDES, Skeleton, get_skeleton
from youthtwin.utils.config import load_config


def test_blazepose_lower_limb_indices() -> None:
    """v2 used 11/13/15 (shoulder, elbow, wrist) for the knee and 13/14 for the ankle."""
    assert (BLAZEPOSE.index("left", "hip"), BLAZEPOSE.index("right", "hip")) == (23, 24)
    assert (BLAZEPOSE.index("left", "knee"), BLAZEPOSE.index("right", "knee")) == (25, 26)
    assert (BLAZEPOSE.index("left", "ankle"), BLAZEPOSE.index("right", "ankle")) == (27, 28)
    assert (BLAZEPOSE.index("left", "heel"), BLAZEPOSE.index("right", "heel")) == (29, 30)
    assert (BLAZEPOSE.index("left", "foot_index"), BLAZEPOSE.index("right", "foot_index")) == (
        31,
        32,
    )
    lower_limb = [BLAZEPOSE.index(s, j) for s in SIDES for j in JOINTS if j != "shoulder"]
    assert min(lower_limb) == 23 and max(lower_limb) == 32
    assert BLAZEPOSE.n_landmarks == 33


def test_code_and_config_agree_on_indices() -> None:
    assert load_config("pose/skeleton.yaml")["blazepose"]["landmarks"] == BLAZEPOSE.landmarks


def test_skeleton_validation_and_lookup() -> None:
    assert get_skeleton("halpe26") is HALPE26
    with pytest.raises(KeyError, match="Unknown skeleton"):
        get_skeleton("openpose")
    with pytest.raises(KeyError, match="no landmark"):
        BLAZEPOSE.index("left", "elbow")
    good = dict(BLAZEPOSE.landmarks)
    with pytest.raises(ValueError, match="exactly"):
        Skeleton("bad", 33, {k: v for k, v in good.items() if k != "left_knee"})
    with pytest.raises(ValueError, match="duplicate"):
        Skeleton("bad", 33, {**good, "left_knee": good["right_knee"]})
    with pytest.raises(ValueError, match="out of range"):
        Skeleton("bad", 20, good)
