"""Movement encoder: backbones, conditioning modes, session pooling, inputs, pre-training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from youthtwin.models.encoder import stgcn
from youthtwin.models.encoder.conditioning import MODES, ConditionedEncoder, SessionEncoder
from youthtwin.models.encoder.film import FiLM
from youthtwin.models.encoder.sequences import (
    ChannelScaler,
    channel_names,
    stack_sessions,
    trial_channels,
)
from youthtwin.models.pretrain import ssl
from youthtwin.models.risk.hazard_neural import NeuralHazardNet, count_parameters
from youthtwin.pose import task_features as tf
from youthtwin.sim import skeleton_sim as sk
from youthtwin.utils.config import load_config
from youthtwin.utils.seeding import make_rng

SIDES = ("left", "right")
CHANNELS = channel_names(SIDES, frontal=True)
COND = 5
BACKBONES = ("tcn_film", "stgcn_film")


def _config(name: str) -> dict[str, Any]:
    return load_config(f"models/{name}.yaml")


def _encoder(name: str, mode: str, seed: int = 0) -> ConditionedEncoder:
    torch.manual_seed(seed)
    return ConditionedEncoder(_config(name), CHANNELS, SIDES, mode=mode, cond_dim=COND).eval()


def _batch(n: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, len(CHANNELS), 101, generator=generator),
        torch.randn(n, COND, generator=generator),
    )


# ----------------------------------------------------------------------------- shapes and size


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_gives_the_same_embedding_size_within_budget(name: str, mode: str) -> None:
    config = _config(name)
    x, g = _batch()
    assert _encoder(name, mode)(x, g).shape == (8, config["embedding_dim"])
    network = NeuralHazardNet(
        config,
        load_config("models/neural.yaml")["head"],
        CHANNELS,
        SIDES,
        mode=mode,
        cond_dim=COND,
        tabular_dim=4,
    )
    assert count_parameters(network) <= config["max_params"]


def test_unknown_mode_and_backbone_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown conditioning mode"):
        ConditionedEncoder(_config("tcn_film"), CHANNELS, SIDES, mode="hypernetwork", cond_dim=COND)
    with pytest.raises(ValueError, match="Unknown backbone"):
        ConditionedEncoder(
            {**_config("tcn_film"), "backbone": "lstm"}, CHANNELS, SIDES, mode="none", cond_dim=COND
        )
    with pytest.raises(ValueError, match="odd"):
        ConditionedEncoder(
            {**_config("tcn_film"), "kernel_size": 4}, CHANNELS, SIDES, mode="none", cond_dim=COND
        )


# ----------------------------------------------------------------------------- conditioning


def test_film_layer_reference_values() -> None:
    layer = FiLM(cond_dim=2, n_channels=3, hidden=4)
    h, g = torch.ones(1, 3, 5), torch.ones(1, 2)
    torch.testing.assert_close(layer(h, g), h)  # identity at initialisation
    last = layer.generator[-1]
    with torch.no_grad():
        last.bias.copy_(torch.tensor([0.5, 0.0, -1.0, 0.0, 2.0, 0.0]))  # d_gamma then beta
    expected = torch.tensor([1.5, 3.0, 0.0])[None, :, None].expand(1, 3, 5)  # (1 + dg) * 1 + beta
    torch.testing.assert_close(layer(h, g), expected)


@pytest.mark.parametrize("name", BACKBONES)
def test_film_starts_as_the_unconditioned_network(name: str) -> None:
    """With FiLM at its identity initialisation, the network equals the same weights without it."""
    film, plain = _encoder(name, "film", seed=1), _encoder(name, "none", seed=2)
    copied = ssl.load_pretrained(plain.backbone, film.backbone.state_dict())
    assert len(copied) == len(plain.backbone.state_dict())
    x, g = _batch()
    torch.testing.assert_close(film(x, g), plain(x, g), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", BACKBONES)
def test_growth_state_changes_the_output_only_when_it_should(name: str) -> None:
    x, g = _batch()
    other = g + 1.0
    none = _encoder(name, "none")
    torch.testing.assert_close(none(x, g), none(x, other))  # never sees growth
    for mode in ("concat_input", "late_fusion"):
        encoder = _encoder(name, mode)
        assert not torch.allclose(encoder(x, g), encoder(x, other))
    film = _encoder(name, "film")
    for layer in film.backbone.films:
        torch.nn.init.normal_(layer.generator[-1].weight, std=0.5)  # as after some training
    assert not torch.allclose(film(x, g), film(x, other))
    with pytest.raises(ValueError, match="needs a growth vector"):
        film.backbone(x)


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("mode", MODES)
def test_each_conditioning_mode_trains(name: str, mode: str) -> None:
    """A few optimisation steps reduce the loss on a small regression target for every mode."""
    torch.manual_seed(0)
    encoder = ConditionedEncoder(_config(name), CHANNELS, SIDES, mode=mode, cond_dim=COND)
    x, g = _batch(32)
    target = x[:, 0].mean(dim=1, keepdim=True).expand(-1, _config(name)["embedding_dim"])
    optimiser = torch.optim.Adam(encoder.parameters(), lr=0.01)
    losses = []
    for _ in range(25):
        loss = ((encoder(x, g) - target) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0]
    if mode == "film":  # the conditioning layers actually receive gradient
        assert any(
            float(film.generator[-1].weight.abs().sum()) > 0 for film in encoder.backbone.films
        )


# ----------------------------------------------------------------------------- session pooling


def test_session_pooling_ignores_padding_and_trial_order() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(
        _encoder("tcn_film", "film"), _config("tcn_film")["embedding_dim"]
    ).eval()
    generator = torch.Generator().manual_seed(1)
    trials = torch.randn(2, 3, len(CHANNELS), 101, generator=generator)
    g = torch.randn(2, COND, generator=generator)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = encoder(trials, mask, g)
    garbage = trials.clone()
    garbage[0, 2] = 1e6  # a padded slot: must not matter
    torch.testing.assert_close(encoder(garbage, mask, g), out)
    shuffled = trials[:, [1, 0, 2]]
    torch.testing.assert_close(encoder(shuffled, mask, g), out, atol=1e-6, rtol=1e-5)
    empty = encoder(trials, torch.zeros(2, 3), g)
    assert torch.equal(empty, torch.zeros_like(empty))  # never screened: zero embedding


# ----------------------------------------------------------------------------- graph layout


def test_joint_graph_and_node_layout() -> None:
    nodes, adjacency = stgcn.build_graph(SIDES)
    assert nodes == [
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]
    torch.testing.assert_close(adjacency, adjacency.T)
    connected = adjacency > 0
    index = {name: k for k, name in enumerate(nodes)}
    assert connected[index["left_hip"], index["left_knee"]]
    assert connected[index["left_hip"], index["right_hip"]]
    assert not connected[index["left_knee"], index["right_knee"]]
    assert not connected[index["left_hip"], index["left_ankle"]]
    layout = stgcn.node_layout(CHANNELS, nodes)
    knee = layout[index["right_knee"]].tolist()
    assert [CHANNELS[k] for k in knee] == [
        "right_knee_flexion",
        "right_knee_flexion_velocity",
        "right_knee_fppa",
        "right_knee_fppa_velocity",
    ]
    assert layout[index["left_ankle"]].tolist()[2:] == [-1, -1]  # ankle has one angle
    one_side_nodes, one_side = stgcn.build_graph(("left",))
    assert len(one_side_nodes) == 3 and one_side.shape == (3, 3)


# ----------------------------------------------------------------------------- inputs


def _trial_result(fps: float, seed: int = 0, **kwargs: Any) -> tf.TrialResult:
    simulated = sk.simulate_drop_vertical_jump(make_rng(seed, "enc"), fps=fps, **kwargs)
    return tf.trial_features(
        tf.TrialInput("a", "s", simulated.task, 1, "both", simulated.sagittal, simulated.frontal),
        filter_config=load_config("pose/filter.yaml"),
        event_config=load_config("pose/events.yaml"),
        feature_config=load_config("features/task_features.yaml"),
    )


def test_trial_channels_order_and_units() -> None:
    result = _trial_result(120.0)
    array = trial_channels(result, SIDES, frontal=True)
    assert array.shape == (len(CHANNELS), 101) and len(CHANNELS) == 16
    assert CHANNELS[:2] == ["left_knee_flexion", "left_knee_flexion_velocity"]
    np.testing.assert_allclose(array[0], result.curves[("left", "knee_flexion")])
    # Velocity is in degrees per second of real time: same movement at 60 fps gives the same.
    slower_camera = trial_channels(_trial_result(60.0), SIDES, frontal=True)
    assert np.abs(array[1]).max() == pytest.approx(np.abs(slower_camera[1]).max(), rel=0.1)
    assert 200.0 < np.abs(array[1]).max() < 1200.0
    sagittal_only = trial_channels(result, ("left",), frontal=False)
    assert sagittal_only.shape == (6, 101)


def test_sessions_are_padded_and_empty_sessions_dropped() -> None:
    rng = np.random.default_rng(0)
    trial = lambda: rng.normal(size=(len(CHANNELS), 101))  # noqa: E731
    sequences = stack_sessions(
        {"s1": [trial(), trial(), trial()], "s2": [trial()], "s3": []},
        {"s1": "a", "s2": "b", "s3": "c"},
        CHANNELS,
    )
    assert sequences.session_ids == ("s1", "s2") and sequences.athlete_ids == ("a", "b")
    assert sequences.trials.shape == (2, 3, 16, 101)
    assert sequences.mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert np.all(sequences.trials[1, 1:] == 0) and sequences.index_of() == {"s1": 0, "s2": 1}
    with pytest.raises(ValueError, match="No session"):
        stack_sessions({"s": []}, {"s": "a"}, CHANNELS)


def test_channel_scaler_uses_training_athletes_only() -> None:
    """N4: held-out athletes' trials cannot change the normalisation statistics."""
    rng = np.random.default_rng(1)
    sessions = {f"s{k}": [rng.normal(10.0 * (k % 2), 2.0, (len(CHANNELS), 101))] for k in range(6)}
    athletes = {f"s{k}": ("train" if k < 4 else "held_out") + str(k) for k in range(6)}
    sequences = stack_sessions(sessions, athletes, CHANNELS)
    train = {athletes[f"s{k}"] for k in range(4)}
    scaler = ChannelScaler.fit(sequences, train)
    corrupted = {**sessions, "s4": [sessions["s4"][0] + 1e6], "s5": [sessions["s5"][0] * -50]}
    again = ChannelScaler.fit(stack_sessions(corrupted, athletes, CHANNELS), train)
    np.testing.assert_array_equal(scaler.mean, again.mean)
    np.testing.assert_array_equal(scaler.sd, again.sd)
    scaled = scaler.transform(sequences.trials[:4])
    assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled.std(axis=(0, 1, 3)) == pytest.approx(np.ones(len(CHANNELS)), abs=1e-6)
    assert set(scaler.to_dict()) == {"mean", "sd"}
    with pytest.raises(ValueError, match="No training sessions"):
        ChannelScaler.fit(sequences, {"nobody"})


# ----------------------------------------------------------------------------- pre-training


def test_time_warp() -> None:
    x = torch.arange(101, dtype=torch.float32)[None, None, :].repeat(2, 3, 1)
    torch.testing.assert_close(ssl.time_warp(x, 1.0), x)
    faster = ssl.time_warp(x, 1.1)
    assert float(faster[0, 0, 50]) == pytest.approx(55.0, abs=1e-4)
    assert float(faster[0, 0, -1]) == 100.0  # holds the final value
    assert float(ssl.time_warp(x, 0.9)[0, 0, 50]) == pytest.approx(45.0, abs=1e-4)


def test_augmentation_never_swaps_limbs() -> None:
    """Each augmented channel still matches its own original, not the other limb's."""
    config = load_config("models/neural.yaml")["pretraining"]
    generator = torch.Generator().manual_seed(0)
    t = torch.linspace(0, 1, 101)
    x = torch.stack([torch.sin((k + 1) * 3.0 * t) for k in range(len(CHANNELS))])[None].repeat(
        4, 1, 1
    )
    out = ssl.augment(x, generator, config)
    assert out.shape == x.shape and not torch.equal(out, x)
    left, right = CHANNELS.index("left_knee_flexion"), CHANNELS.index("right_knee_flexion")
    for own, other in ((left, right), (right, left)):
        same = torch.corrcoef(torch.stack([out[0, own], x[0, own]]))[0, 1]
        cross = torch.corrcoef(torch.stack([out[0, own], x[0, other]]))[0, 1]
        assert same > 0.95 and same > cross + 0.2
    low, high = config["time_warp_range"]
    assert 0.85 < low < 1.0 < high < 1.15


def test_mask_span() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.ones(16, 5, 101)
    masked, hidden = ssl.mask_span(x, generator, 0.2)
    assert hidden.sum(dim=1).tolist() == [20] * 16  # one contiguous span of 20% per trial
    edges = hidden[:, 1:] != hidden[:, :-1]
    assert edges.sum(dim=1).max() <= 2
    assert torch.equal(masked == 0, hidden[:, None, :].expand_as(masked))


def test_pretraining_learns_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 101)
    amplitude = rng.normal(1.0, 0.3, (200, 1, 1))
    trials = amplitude * np.sin(np.arange(1, 7)[None, :, None] * 2.0 * t) + rng.normal(
        0, 0.05, (200, 6, 101)
    )
    config = {**load_config("models/neural.yaml")["pretraining"], "epochs": 6}
    weights, history = ssl.pretrain(trials, _config("tcn_film"), config, seed=3)
    assert history[-1] < 0.5 * history[0]
    again, history_again = ssl.pretrain(trials, _config("tcn_film"), config, seed=3)
    assert history == history_again
    assert all(torch.equal(weights[k], again[k]) for k in weights)
    different, _ = ssl.pretrain(trials, _config("tcn_film"), config, seed=4)
    assert not torch.equal(weights["stem.weight"], different["stem.weight"])


def test_pretrained_weights_load_where_shapes_match() -> None:
    plain = _encoder("tcn_film", "none", seed=5)
    weights = plain.backbone.state_dict()
    film = _encoder("tcn_film", "film", seed=6)
    matched = ssl.load_pretrained(film.backbone, weights)
    assert set(matched) == set(weights)  # FiLM layers are extra and keep their initial values
    torch.testing.assert_close(film.backbone.stem.weight, plain.backbone.stem.weight)
    widened = ConditionedEncoder(
        {**_config("tcn_film")}, CHANNELS[:6], ("left",), mode="concat_input", cond_dim=COND
    )
    partial = ssl.load_pretrained(widened.backbone, weights)
    assert "stem.weight" not in partial and "blocks.0.body.0.weight" in partial
