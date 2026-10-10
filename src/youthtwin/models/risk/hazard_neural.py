"""Neural discrete-time hazard: movement encoder + growth state + tabular inputs -> risk.

One network per outcome group (cause-specific, N6). For each athlete-week the inputs are

- the trials of the athlete's latest screening session (encoded and attention-pooled),
- the growth-state vector ``g`` (conditions the encoder according to the chosen mode),
- tabular inputs that are not growth quantities (exposure, injury history, sex, ...).

The head is a small MLP on ``[session embedding, tabular inputs]``. Growth enters only through
the encoder's conditioning mode, so the ablation compares like with like; with mode ``none``
the network never sees the growth state.

Everything learned from data is learned on training rows: standardisation of ``g`` and of the
tabular inputs, imputation, the encoder and the head. Early stopping uses validation athletes
drawn from the training athletes. A deep ensemble is several networks that differ only in seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from youthtwin.models.encoder.conditioning import ConditionedEncoder, SessionEncoder
from youthtwin.models.encoder.sequences import ChannelScaler, SessionSequences
from youthtwin.models.pretrain.ssl import load_pretrained

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
_EPS = 1e-8


@dataclass(frozen=True)
class NeuralData:
    """Athlete-week inputs as arrays (rows aligned across fields)."""

    growth: FloatArray  # (N, D) growth-state vector, may contain NaN
    tabular: FloatArray  # (N, P) non-growth inputs, may contain NaN
    session_index: IntArray  # (N,) row of ``SessionSequences``; -1 if never screened
    outcome: FloatArray  # (N,)
    athlete: NDArray[np.str_]  # (N,)


@dataclass(frozen=True)
class Standardiser:
    """Column means and SDs from training rows; missing values become 0 (the training mean)."""

    mean: FloatArray
    sd: FloatArray

    @classmethod
    def fit(cls, values: FloatArray) -> Standardiser:
        if values.shape[1] == 0:
            return cls(np.zeros(0), np.ones(0))
        with np.errstate(invalid="ignore"):
            mean = np.nan_to_num(np.nanmean(values, axis=0))
            sd = np.nan_to_num(np.nanstd(values, axis=0), nan=1.0)
        return cls(mean, np.where(sd > _EPS, sd, 1.0))

    def transform(self, values: FloatArray) -> FloatArray:
        return np.nan_to_num((values - self.mean) / self.sd, nan=0.0)


class NeuralHazardNet(nn.Module):
    def __init__(
        self,
        encoder_config: Mapping[str, Any],
        head_config: Mapping[str, Any],
        channels: Sequence[str],
        sides: Sequence[str],
        *,
        mode: str,
        cond_dim: int,
        tabular_dim: int,
    ) -> None:
        super().__init__()
        embedding_dim = int(encoder_config["embedding_dim"])
        encoder = ConditionedEncoder(encoder_config, channels, sides, mode=mode, cond_dim=cond_dim)
        self.session_encoder = SessionEncoder(encoder, embedding_dim)
        hidden = int(head_config["hidden"])
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + tabular_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(float(head_config["dropout"])),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, trials: torch.Tensor, mask: torch.Tensor, g: torch.Tensor, tabular: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.session_encoder(trials, mask, g)
        screened = (mask.sum(dim=1, keepdim=True) > 0).to(embedding.dtype)
        return self.head(torch.cat([embedding, tabular, screened], dim=1)).squeeze(-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class FittedNeuralHazard:
    """A trained ensemble with the preprocessing it was trained with."""

    networks: list[NeuralHazardNet]
    channel_scaler: ChannelScaler
    growth_scaler: Standardiser
    tabular_scaler: Standardiser
    mode: str
    history: list[list[float]]  # validation loss per epoch, per ensemble member

    def _tensors(
        self, data: NeuralData, sequences: SessionSequences, rows: IntArray, growth: FloatArray
    ) -> tuple[torch.Tensor, ...]:
        index = data.session_index[rows]
        present = index >= 0
        safe = np.where(present, index, 0)
        trials = self.channel_scaler.transform(sequences.trials[safe])
        mask = sequences.mask[safe] * present[:, None]
        return (
            torch.as_tensor(trials * mask[:, :, None, None], dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
            torch.as_tensor(self.growth_scaler.transform(growth), dtype=torch.float32),
            torch.as_tensor(self.tabular_scaler.transform(data.tabular[rows]), dtype=torch.float32),
        )

    def predict_members(
        self,
        data: NeuralData,
        sequences: SessionSequences,
        rows: IntArray | None = None,
        *,
        growth: FloatArray | None = None,
        batch_size: int = 1024,
    ) -> FloatArray:
        """Predicted risk per ensemble member: (members, rows).

        ``growth`` overrides the growth vector of the selected rows (same shape), which is how
        draws from the growth posterior are pushed through the model.
        """
        rows = np.arange(len(data.outcome)) if rows is None else rows
        g_all = data.growth[rows] if growth is None else growth
        out = np.empty((len(self.networks), len(rows)))
        with torch.no_grad():
            for m, network in enumerate(self.networks):
                network.eval()
                for start in range(0, len(rows), batch_size):
                    part = slice(start, start + batch_size)
                    tensors = self._tensors(data, sequences, rows[part], g_all[part])
                    out[m, part] = torch.sigmoid(network(*tensors)).numpy()
        return out

    def predict_proba(
        self, data: NeuralData, sequences: SessionSequences, rows: IntArray | None = None
    ) -> FloatArray:
        """Ensemble-mean predicted risk."""
        return np.asarray(self.predict_members(data, sequences, rows).mean(axis=0))


def _train_one(
    fitted: FittedNeuralHazard,
    network: NeuralHazardNet,
    data: NeuralData,
    sequences: SessionSequences,
    train: IntArray,
    validation: IntArray,
    config: Mapping[str, Any],
    seed: int,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW(
        network.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    batch_size = int(config["batch_size"])
    validation_y = torch.as_tensor(data.outcome[validation], dtype=torch.float32)
    best, best_state, stale = float("inf"), network.state_dict(), 0
    history: list[float] = []
    for _ in range(int(config["max_epochs"])):
        network.train()
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            tensors = fitted._tensors(data, sequences, rows, data.growth[rows])
            loss = loss_fn(
                network(*tensors), torch.as_tensor(data.outcome[rows], dtype=torch.float32)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        network.eval()
        with torch.no_grad():
            logits = torch.cat(
                [
                    network(
                        *fitted._tensors(
                            data,
                            sequences,
                            validation[s : s + 1024],
                            data.growth[validation[s : s + 1024]],
                        )
                    )
                    for s in range(0, len(validation), 1024)
                ]
            )
            current = float(loss_fn(logits, validation_y))
        history.append(current)
        if current < best - float(config["min_delta"]):
            best, stale = current, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    network.load_state_dict(best_state)
    return history


def fit_neural_hazard(
    data: NeuralData,
    sequences: SessionSequences,
    train_rows: BoolArray,
    *,
    mode: str,
    sides: Sequence[str],
    encoder_config: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    pretrained: Mapping[str, torch.Tensor] | None = None,
) -> FittedNeuralHazard:
    """Train an ensemble on the rows where ``train_rows`` is True.

    A fraction of the **training athletes** is set aside for early stopping. Scalers are fitted
    on the remaining training rows; held-out rows are never read.
    """
    rng = np.random.default_rng(seed)
    athletes = np.unique(data.athlete[train_rows])
    rng.shuffle(athletes)
    n_validation = max(1, round(float(config["validation_fraction"]) * len(athletes)))
    validation_athletes = set(athletes[:n_validation])
    in_validation = np.isin(data.athlete, list(validation_athletes))
    train = np.flatnonzero(train_rows & ~in_validation)
    validation = np.flatnonzero(train_rows & in_validation)
    if len(np.unique(data.outcome[train])) < 2:
        raise ValueError("Training rows contain a single outcome class")

    fitted = FittedNeuralHazard(
        networks=[],
        channel_scaler=ChannelScaler.fit(sequences, set(data.athlete[train])),
        growth_scaler=Standardiser.fit(data.growth[train]),
        tabular_scaler=Standardiser.fit(data.tabular[train]),
        mode=mode,
        history=[],
    )
    for member in range(int(config["ensemble_size"])):
        member_seed = seed * 1000 + member
        torch.manual_seed(member_seed)
        network = NeuralHazardNet(
            encoder_config,
            config["head"],
            sequences.channels,
            sides,
            mode=mode,
            cond_dim=data.growth.shape[1],
            tabular_dim=data.tabular.shape[1],
        )
        if pretrained is not None:
            load_pretrained(network.session_encoder.encoder.backbone, pretrained)
        # Start the output bias at the training log-odds so early epochs are not spent on it.
        prevalence = float(np.clip(data.outcome[train].mean(), 1e-4, 1 - 1e-4))
        final = network.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, float(np.log(prevalence / (1 - prevalence))))
        fitted.history.append(
            _train_one(fitted, network, data, sequences, train, validation, config, member_seed)
        )
        fitted.networks.append(network)
    return fitted
