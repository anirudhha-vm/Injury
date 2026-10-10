"""Between-limb asymmetry comes only from event- and window-level features (v2 regression).

v2 subtracted left from right frame by frame, including during single-leg tasks where the two
limbs are doing different things by design.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from youthtwin.pose import kinematics
from youthtwin.pose import task_features as tf
from youthtwin.sim import skeleton_sim as sk
from youthtwin.utils.config import load_config
from youthtwin.utils.seeding import make_rng


@pytest.fixture(scope="module")
def configs() -> dict[str, Any]:
    return {
        "filter_config": load_config("pose/filter.yaml"),
        "event_config": load_config("pose/events.yaml"),
        "feature_config": load_config("features/task_features.yaml"),
    }


def _trial(simulated: sk.SyntheticTrial, number: int, session: str = "s1") -> tf.TrialInput:
    return tf.TrialInput(
        "athlete",
        session,
        simulated.task,
        number,
        simulated.limb,
        simulated.sagittal,
        simulated.frontal,
    )


def test_asymmetry_indices_reference_values() -> None:
    assert tf.symmetry_angle(10.0, 10.0) == pytest.approx(0.0)
    # atan2(left=10, right=20) = 26.565 deg -> (45 - 26.565) / 90 * 100 = 20.48
    assert tf.symmetry_angle(10.0, 20.0) == pytest.approx(20.483, abs=1e-3)
    assert tf.symmetry_angle(20.0, 10.0) == pytest.approx(-20.483, abs=1e-3)
    assert -100.0 <= tf.symmetry_angle(-5.0, 8.0) <= 100.0  # opposite signs stay bounded
    assert tf.limb_symmetry_index(8.0, 10.0) == pytest.approx(80.0)
    assert tf.limb_symmetry_index(10.0, 8.0) == pytest.approx(80.0)
    assert np.isnan(tf.limb_symmetry_index(-2.0, 5.0))  # undefined across zero
    assert tf.asymmetry_value(12.0, 7.0, "difference") == pytest.approx(5.0)
    assert tf.asymmetry_value(8.0, 10.0, "lsi") == pytest.approx(80.0)
    with pytest.raises(ValueError, match="Unknown asymmetry index"):
        tf.asymmetry_value(1.0, 2.0, "ratio")


def test_no_frame_wise_asymmetry_function_exists() -> None:
    """Asymmetry helpers take two numbers, never two time series."""
    for function in (tf.symmetry_angle, tf.limb_symmetry_index, tf.asymmetry_value):
        parameters = inspect.signature(function).parameters
        assert [p.annotation for p in list(parameters.values())[:2]] == ["float", "float"]
    for module in (tf, kinematics):
        public = [name for name in dir(module) if not name.startswith("_")]
        assert not [name for name in public if "asym" in name.lower() and "frame" in name.lower()]


def test_bilateral_task_asymmetry_uses_values_at_the_same_events(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(1, "bi"), peak_fppa=(14.0, 6.0))
    result = tf.trial_features(_trial(simulated, 1), **configs)
    table, excluded = tf.session_features([result], configs["feature_config"])
    assert excluded.empty
    asymmetry = table[table["feature"].str.startswith("asymmetry_")]
    assert (asymmetry["limb"] == "both").all() and asymmetry["trial"].isna().all()
    # Every asymmetry row is built from an event- or window-level feature of each limb.
    suffixes = ("_max", "_rom", "_peak_velocity", *(f"_at_{e}" for e in result.events.events))
    assert asymmetry["feature"].str.endswith(suffixes).all()

    row = asymmetry.set_index("feature")["value"]
    left = result.features[("left", "knee_fppa_at_peak_knee_flexion")]
    right = result.features[("right", "knee_fppa_at_peak_knee_flexion")]
    assert left == pytest.approx(14.0, abs=0.5) and right == pytest.approx(6.0, abs=0.5)
    assert row["asymmetry_knee_fppa_at_peak_knee_flexion"] == pytest.approx(
        tf.symmetry_angle(left, right)
    )
    # Sagittal flexion is identical on both sides in this simulation: no asymmetry.
    assert row["asymmetry_knee_flexion_at_peak_knee_flexion"] == pytest.approx(0.0, abs=0.5)


def test_unilateral_task_compares_left_trials_with_right_trials(configs: dict[str, Any]) -> None:
    left = [
        sk.simulate_single_leg_squat(make_rng(2, f"L{k}"), limb="left", peak_knee_flexion=70.0)
        for k in range(2)
    ]
    right = [
        sk.simulate_single_leg_squat(make_rng(2, f"R{k}"), limb="right", peak_knee_flexion=55.0)
        for k in range(2)
    ]
    results = [tf.trial_features(_trial(s, k + 1), **configs) for k, s in enumerate(left + right)]

    # Within one single-leg trial only the test limb is measured: nothing to subtract.
    for result in results:
        limbs = {limb for limb, _ in result.features}
        assert limbs == {result.trial.limb}
    one_trial, _ = tf.session_features(results[:1], configs["feature_config"])
    assert not one_trial["feature"].str.startswith("asymmetry_").any()

    table, _ = tf.session_features(results, configs["feature_config"])
    row = table[table["feature"] == "asymmetry_knee_flexion_at_deepest_point"]
    assert len(row) == 1
    assert row["value"].iloc[0] == pytest.approx(tf.symmetry_angle(70.0, 55.0), abs=0.5)
    means = table[table["trial"].isna() & (table["feature"] == "knee_flexion_at_deepest_point")]
    assert means.set_index("limb")["value"].to_dict() == pytest.approx(
        {"left": 70.0, "right": 55.0}, abs=0.5
    )


def test_asymmetry_index_is_configurable(configs: dict[str, Any]) -> None:
    simulated = sk.simulate_drop_vertical_jump(make_rng(3, "cfg"), peak_fppa=(14.0, 6.0))
    result = tf.trial_features(_trial(simulated, 1), **configs)
    by_difference = {**configs["feature_config"], "asymmetry": {"index": "difference"}}
    table, _ = tf.session_features([result], by_difference)
    value = table.set_index("feature").loc["asymmetry_knee_fppa_at_peak_knee_flexion", "value"]
    assert value == pytest.approx(8.0, abs=0.7)
