from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from radar_module.preprocess.pointcloud_sequence import POINT_FEATURE_NAMES


POINT_TEMPORAL_MODEL_VERSION = "pointnet_gru_pretrain_v1"


class MaskedPointNetFrameEncoder(nn.Module):
    """Small PointNet-style unordered frame encoder with masked pooling."""

    def __init__(self, *, input_size: int = len(POINT_FEATURE_NAMES), hidden_size: int = 64) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_size),
            nn.ReLU(),
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        if points.ndim != 4:
            raise ValueError("points must have shape [batch, time, point, feature]")
        if points.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {points.shape[-1]}")
        if point_mask.shape != points.shape[:3]:
            raise ValueError("point_mask must match batch/time/point dimensions")
        mask = point_mask.to(dtype=torch.bool)
        encoded = self.point_mlp(points)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=2).clamp_min(1)
        mean_pool = (encoded * valid).sum(dim=2) / count
        negative_inf = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~valid, negative_inf).max(dim=2).values
        empty = ~mask.any(dim=2)
        max_pool = max_pool.masked_fill(empty.unsqueeze(-1), 0.0)
        return self.frame_projection(torch.cat((max_pool, mean_pool), dim=-1))


class PointTemporalEncoder(nn.Module):
    """PointNet frame representation followed by a deliberately small GRU."""

    def __init__(
        self,
        *,
        input_size: int = len(POINT_FEATURE_NAMES),
        frame_hidden_size: int = 64,
        temporal_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.frame_encoder = MaskedPointNetFrameEncoder(
            input_size=input_size,
            hidden_size=frame_hidden_size,
        )
        self.temporal_hidden_size = int(temporal_hidden_size)
        self.temporal = nn.GRU(
            input_size=frame_hidden_size,
            hidden_size=self.temporal_hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence, _ = self.forward_sequence(points, point_mask, frame_mask)
        return self.pool_last(sequence, frame_mask)

    def forward_sequence(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return causal temporal states and their PointNet frame targets."""

        if frame_mask.shape != points.shape[:2]:
            raise ValueError("frame_mask must match batch/time dimensions")
        if not torch.all(frame_mask.any(dim=1)):
            raise ValueError("each sequence must contain at least one observed frame")
        frames = self.frame_encoder(points, point_mask)
        frames = frames * frame_mask.unsqueeze(-1).to(frames.dtype)
        sequence, _ = self.temporal(frames)
        return sequence, frames

    @staticmethod
    def pool_last(sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        last_valid = indices.masked_fill(~frame_mask.bool(), -1).max(dim=1).values
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), last_valid]


class NormalDynamicsAuxiliaryHead(nn.Module):
    """Training-only predictor for the next observed PointNet frame embedding."""

    def __init__(self, *, temporal_hidden_size: int, frame_hidden_size: int) -> None:
        super().__init__()
        if temporal_hidden_size <= 0 or frame_hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        self.output = nn.Linear(temporal_hidden_size, frame_hidden_size)

    def forward(self, temporal_states: torch.Tensor) -> torch.Tensor:
        return self.output(temporal_states)


class PointTemporalPretrainingModel(nn.Module):
    """Activity-supervised encoder pretraining; it is not a fall predictor."""

    def __init__(
        self,
        *,
        class_count: int,
        input_size: int = len(POINT_FEATURE_NAMES),
        frame_hidden_size: int = 64,
        temporal_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least two")
        self.class_count = int(class_count)
        self.encoder = PointTemporalEncoder(
            input_size=input_size,
            frame_hidden_size=frame_hidden_size,
            temporal_hidden_size=temporal_hidden_size,
        )
        self.activity_head = nn.Linear(temporal_hidden_size, self.class_count)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        representation = self.encoder(points, point_mask, frame_mask)
        return self.activity_head(representation)


class PointTemporalPredictionHead(nn.Module):
    """Separate learned multi-horizon head for later pre-fall fine-tuning."""

    def __init__(self, encoder: PointTemporalEncoder, *, horizon_count: int = 3) -> None:
        super().__init__()
        if horizon_count <= 0:
            raise ValueError("horizon_count must be positive")
        self.encoder = encoder
        self.horizon_count = int(horizon_count)
        self.output = nn.Linear(encoder.temporal_hidden_size, self.horizon_count)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(self.encoder(points, point_mask, frame_mask))


@dataclass(frozen=True, slots=True)
class PointTemporalCheckpointMetadata:
    model_version: str = POINT_TEMPORAL_MODEL_VERSION
    sequence_version: str = "radar_point_sequence_v1"
    feature_names: tuple[str, ...] = POINT_FEATURE_NAMES
    model_role: str = "representation_pretraining"
    deployment_eligible: bool = False
