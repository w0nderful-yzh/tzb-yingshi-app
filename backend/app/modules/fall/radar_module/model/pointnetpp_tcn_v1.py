from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODEL_VERSION = "radar_pointnetpp_tcn_upgrade_v1"
ARCHITECTURE = "pointnetpp_frame_encoder_causal_tcn"
INPUT_FEATURES = ("x_m", "y_m", "z_m", "radial_velocity_mps", "snr")
TIME_STEPS = 20
MAX_POINTS = 64


def _index(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    view = (values.shape[0],) + (1,) * (indices.ndim - 1)
    return values[batch.view(view).expand_as(indices), indices]


def masked_farthest_point_sample(
    xyz: torch.Tensor, mask: torch.Tensor, sample_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic masked FPS for small 64-point radar frames."""
    if xyz.ndim != 3 or mask.shape != xyz.shape[:2]:
        raise ValueError("xyz/mask shapes are incompatible")
    batch, point_count, _ = xyz.shape
    valid_count = mask.sum(dim=1)
    centroid = (xyz * mask.unsqueeze(-1)).sum(dim=1) / valid_count.clamp_min(1).unsqueeze(-1)
    start_distance = ((xyz - centroid[:, None]) ** 2).sum(dim=-1)
    start_distance = start_distance.masked_fill(~mask, -1.0)
    farthest = start_distance.argmax(dim=1)
    minimum_distance = torch.full(
        (batch, point_count), torch.finfo(xyz.dtype).max,
        device=xyz.device, dtype=xyz.dtype,
    )
    selected: list[torch.Tensor] = []
    for _ in range(sample_count):
        selected.append(farthest)
        center = _index(xyz, farthest)
        distance = ((xyz - center[:, None]) ** 2).sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance = minimum_distance.masked_fill(~mask, -1.0)
        farthest = minimum_distance.argmax(dim=1)
    indices = torch.stack(selected, dim=1)
    sampled_mask = torch.arange(sample_count, device=xyz.device)[None] < valid_count[:, None]
    return indices, sampled_mask


class SetAbstraction(nn.Module):
    def __init__(
        self,
        *,
        input_feature_size: int,
        output_size: int,
        centroid_count: int,
        neighbour_count: int,
    ) -> None:
        super().__init__()
        self.centroid_count = int(centroid_count)
        self.neighbour_count = int(neighbour_count)
        hidden = max(32, output_size // 2)
        self.mlp = nn.Sequential(
            nn.Linear(3 + input_feature_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
            nn.ReLU(),
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid_indices, centroid_mask = masked_farthest_point_sample(
            xyz, mask, self.centroid_count
        )
        centroid_xyz = _index(xyz, centroid_indices)
        distances = torch.cdist(centroid_xyz, xyz)
        distances = distances.masked_fill(~mask[:, None], float("inf"))
        neighbour_count = min(self.neighbour_count, xyz.shape[1])
        neighbour_indices = distances.topk(neighbour_count, dim=-1, largest=False).indices
        neighbour_xyz = _index(xyz, neighbour_indices)
        neighbour_features = _index(features, neighbour_indices)
        neighbour_valid = _index(mask.unsqueeze(-1), neighbour_indices).squeeze(-1)
        local = torch.cat(
            (neighbour_xyz - centroid_xyz[:, :, None], neighbour_features), dim=-1
        )
        encoded = self.mlp(local)
        encoded = encoded.masked_fill(~neighbour_valid.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        pooled = encoded.max(dim=2).values
        pooled = pooled.masked_fill(~centroid_mask.unsqueeze(-1), 0.0)
        centroid_xyz = centroid_xyz.masked_fill(~centroid_mask.unsqueeze(-1), 0.0)
        return centroid_xyz, pooled, centroid_mask


class PointNetPlusPlusFrameEncoder(nn.Module):
    def __init__(self, *, input_size: int = 5, output_size: int = 64) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.sa1 = SetAbstraction(
            input_feature_size=input_size,
            output_size=64,
            centroid_count=16,
            neighbour_count=8,
        )
        self.sa2 = SetAbstraction(
            input_feature_size=64,
            output_size=128,
            centroid_count=4,
            neighbour_count=8,
        )
        self.output = nn.Sequential(
            nn.Linear(256, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        if points.ndim != 4 or points.shape[-1] != self.input_size:
            raise ValueError("points must have shape [batch,time,point,5]")
        if point_mask.shape != points.shape[:3]:
            raise ValueError("point_mask shape mismatch")
        batch, time, point_count, feature_count = points.shape
        flat = points.reshape(batch * time, point_count, feature_count)
        mask = point_mask.reshape(batch * time, point_count).bool()
        xyz = flat[:, :, :3]
        xyz1, feature1, mask1 = self.sa1(xyz, flat, mask)
        _xyz2, feature2, mask2 = self.sa2(xyz1, feature1, mask1)
        valid = mask2.unsqueeze(-1)
        mean_pool = (feature2 * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        max_pool = feature2.masked_fill(~valid, torch.finfo(feature2.dtype).min).max(dim=1).values
        empty = ~mask2.any(dim=1)
        max_pool = max_pool.masked_fill(empty.unsqueeze(-1), 0.0)
        encoded = self.output(torch.cat((max_pool, mean_pool), dim=-1))
        encoded = encoded.masked_fill(empty.unsqueeze(-1), 0.0)
        return encoded.reshape(batch, time, self.output_size)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.convolution(F.pad(values, (self.padding, 0)))
        values = self.normalization(values)
        return F.relu(residual + self.dropout(F.relu(values)))


class PointNetPlusPlusTcnPrefall(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 5,
        frame_size: int = 64,
        temporal_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_encoder = PointNetPlusPlusFrameEncoder(
            input_size=input_size, output_size=frame_size
        )
        self.input_projection = nn.Conv1d(frame_size, temporal_size, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            CausalResidualBlock(temporal_size, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.output = nn.Linear(temporal_size, 1)

    def encode_frames(
        self, points: torch.Tensor, point_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.frame_encoder(points, point_mask)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if frame_mask.shape != points.shape[:2]:
            raise ValueError("frame_mask shape mismatch")
        frames = self.encode_frames(points, point_mask)
        frames = frames * frame_mask.unsqueeze(-1).to(frames.dtype)
        values = self.input_projection(frames.transpose(1, 2))
        for block in self.temporal_blocks:
            values = block(values)
        indices = torch.arange(values.shape[2], device=values.device)[None]
        last_valid = indices.masked_fill(~frame_mask.bool(), -1).max(dim=1).values
        if torch.any(last_valid < 0):
            raise ValueError("each sequence must contain an observed frame")
        representation = values[
            torch.arange(values.shape[0], device=values.device), :, last_valid
        ]
        return self.output(representation).squeeze(-1)


class SpatialActivityPretrainer(nn.Module):
    """Fall-102 activity supervision updates only the spatial frame encoder."""

    def __init__(self, *, class_count: int, frame_size: int = 64) -> None:
        super().__init__()
        self.frame_encoder = PointNetPlusPlusFrameEncoder(output_size=frame_size)
        self.activity_head = nn.Linear(frame_size * 2, class_count)

    def forward(
        self, points: torch.Tensor, point_mask: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        frames = self.frame_encoder(points, point_mask)
        valid = frame_mask.unsqueeze(-1)
        mean_pool = (frames * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        max_pool = frames.masked_fill(~valid, torch.finfo(frames.dtype).min).max(dim=1).values
        return self.activity_head(torch.cat((max_pool, mean_pool), dim=-1))
