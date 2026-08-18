from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from radar_module.contracts import ModelMode


MODEL_VERSION = "radar_lstm_v1"


@dataclass(frozen=True, slots=True)
class LoadedRadarModel:
    model: "RadarLSTM"
    model_mode: ModelMode
    model_version: str
    feature_version: str
    feature_names: tuple[str, ...]
    window_size: int
    input_size: int


class RadarLSTM(nn.Module):
    """MVP风险推理模型。

    输入必须已经由FeatureVector窗口构造成[B, 30, 8]。本模块不知道
    RadarFrame或点云结构，forward只返回risk_logit。
    """

    def __init__(
        self,
        *,
        input_size: int = 8,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"expected input_size={self.input_size}, got {features.shape[-1]}"
            )
        sequence_output, _ = self.lstm(features)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)

    @classmethod
    def create_test_checkpoint(
        cls,
        output_path: str | Path,
        *,
        feature_version: str = "radar_features_v1",
        feature_names: tuple[str, ...],
        window_size: int = 30,
        input_size: int = 8,
        hidden_size: int = 64,
        seed: int = 20260724,
    ) -> Path:
        """创建确定性的系统联调checkpoint，不代表训练结果。

        权重固定为零，输出层bias固定为1，使窗口完成后得到稳定的DEMO
        分数(sigmoid(1)≈0.731)。这便于验证事件链路，也确保不会把随机
        初始化误认为模型推理。
        """

        if len(feature_names) != input_size:
            raise ValueError("feature_names length must match input_size")
        torch.manual_seed(seed)
        model = cls(input_size=input_size, hidden_size=hidden_size)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output.bias.fill_(1.0)

        checkpoint = {
            "model_version": MODEL_VERSION,
            "model_mode": ModelMode.TEST_CHECKPOINT.value,
            "feature_version": feature_version,
            "feature_names": tuple(feature_names),
            "window_size": int(window_size),
            "input_size": int(input_size),
            "hidden_size": int(hidden_size),
            "state_dict": model.state_dict(),
            "seed": int(seed),
            "purpose": "DEMO pipeline validation only",
        }
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_feature_version: str,
        expected_feature_names: tuple[str, ...],
        expected_window_size: int,
        expected_input_size: int,
        device: str | torch.device = "cpu",
    ) -> LoadedRadarModel:
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        payload = _safe_torch_load(path, map_location=device)
        _validate_checkpoint_payload(payload)

        feature_version = str(payload["feature_version"])
        feature_names = tuple(str(name) for name in payload["feature_names"])
        window_size = int(payload["window_size"])
        input_size = int(payload["input_size"])
        if feature_version != expected_feature_version:
            raise ValueError(
                "checkpoint feature_version is incompatible: "
                f"{feature_version} != {expected_feature_version}"
            )
        if feature_names != tuple(expected_feature_names):
            raise ValueError("checkpoint feature_names/order are incompatible")
        if window_size != expected_window_size:
            raise ValueError("checkpoint window_size is incompatible")
        if input_size != expected_input_size:
            raise ValueError("checkpoint input_size is incompatible")

        model_mode = ModelMode(str(payload["model_mode"]))
        hidden_size = int(payload.get("hidden_size", 64))
        model = cls(input_size=input_size, hidden_size=hidden_size)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        model.eval()
        return LoadedRadarModel(
            model=model,
            model_mode=model_mode,
            model_version=str(payload["model_version"]),
            feature_version=feature_version,
            feature_names=feature_names,
            window_size=window_size,
            input_size=input_size,
        )


def _safe_torch_load(
    path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    return payload


def _validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    required = {
        "model_version",
        "model_mode",
        "feature_version",
        "feature_names",
        "window_size",
        "input_size",
        "state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint metadata is incomplete: {missing}")
    if payload["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"unsupported model_version: {payload['model_version']}"
        )
    ModelMode(str(payload["model_mode"]))
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("checkpoint state_dict must be a mapping")
