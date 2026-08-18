from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


EXPERIMENT_MODEL_VERSION = "radar_temporal_experiment_v3"
MULTITASK_MODEL_VERSION = "radar_multitask_experiment_v3"
MULTIHORIZON_MODEL_VERSION = "radar_multihorizon_experiment_v4"


class LstmTemporalEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.output_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(features)
        return sequence[:, -1]


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = functional.pad(values, (self.left_padding, 0))
        values = self.convolution(values)
        return self.activation(values + residual)


class CausalTcnTemporalEncoder(nn.Module):
    """Small causal TCN with a receptive field covering the 20-frame window."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.output_size = hidden_size
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            _CausalResidualBlock(hidden_size, dilation)
            for dilation in (1, 2, 4, 8)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        return values[:, :, -1]


class TemporalBinaryModel(nn.Module):
    def __init__(
        self,
        *,
        architecture: str,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = build_temporal_encoder(
            architecture=architecture,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.output = nn.Linear(self.encoder.output_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(features)).squeeze(-1)


class SharedMultiTaskTemporalModel(nn.Module):
    """Shared radar encoder with semantically separate research heads."""

    def __init__(
        self,
        *,
        architecture: str,
        input_size: int,
        hidden_size: int,
        action_class_count: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = build_temporal_encoder(
            architecture=architecture,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        output_size = self.encoder.output_size
        self.prefall_head = nn.Linear(output_size, 1)
        self.fall_sequence_head = nn.Linear(output_size, 1)
        self.action_head = nn.Linear(output_size, action_class_count)

    def forward_prefall(self, features: torch.Tensor) -> torch.Tensor:
        return self.prefall_head(self.encoder(features)).squeeze(-1)

    def forward_iwr6843(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return (
            self.fall_sequence_head(encoded).squeeze(-1),
            self.action_head(encoded),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_prefall(features)


class MultiHorizonTemporalModel(nn.Module):
    """One causal encoder with separate early and imminent pre-fall heads."""

    def __init__(
        self,
        *,
        architecture: str,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = build_temporal_encoder(
            architecture=architecture,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        output_size = self.encoder.output_size
        self.early_head = nn.Linear(output_size, 1)
        self.imminent_head = nn.Linear(output_size, 1)

    def forward_all(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return (
            self.early_head(encoded).squeeze(-1),
            self.imminent_head(encoded).squeeze(-1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        early, _ = self.forward_all(features)
        return early


def build_temporal_encoder(
    *, architecture: str, input_size: int, hidden_size: int
) -> LstmTemporalEncoder | CausalTcnTemporalEncoder:
    if architecture == "lstm":
        return LstmTemporalEncoder(input_size, hidden_size)
    if architecture == "causal_tcn":
        return CausalTcnTemporalEncoder(input_size, hidden_size)
    raise ValueError(f"unsupported temporal architecture: {architecture}")
