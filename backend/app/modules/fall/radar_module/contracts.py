from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Room(str, Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"


class SourceMode(str, Enum):
    REAL = "REAL"
    REPLAY = "REPLAY"


class ModelMode(str, Enum):
    TEST_CHECKPOINT = "TEST_CHECKPOINT"
    TRAINED_CHECKPOINT = "TRAINED_CHECKPOINT"


class HumanState(str, Enum):
    NO_PERSON = "NO_PERSON"
    NORMAL = "NORMAL"
    FALL_RISK = "FALL_RISK"


DEMO_DISCLAIMER = "当前为DEMO风险推理框架结果，不能代表真实跌倒预测能力"


@dataclass(frozen=True, slots=True)
class RadarPoint:
    x: float
    y: float
    z: float
    velocity: float
    snr: float | None = None
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class RadarTarget:
    """TI People Tracking target metadata retained for shadow alignment only.

    字段对应 TI Target List TLV（world-frame，米）：
    - posX/posY/posZ → x/y/z
    - velX/velY/velZ → velocity_x/y/z（米/秒）
    - accX/accY/accZ → accel_x/y/z（米/秒²）
    - confidenceLevel → confidence
    """

    track_id: int
    x: float
    y: float
    z: float
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_z: float | None = None
    accel_x: float | None = None
    accel_y: float | None = None
    accel_z: float | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RadarFrame:
    timestamp: datetime
    device_id: str
    room: Room
    source_mode: SourceMode
    points: tuple[RadarPoint, ...]
    # Optional decoded-frame metadata. Existing model consumers only read the
    # fields above; these values are an observational Camera-Radar sidecar.
    frame_number: int | None = None
    source_timestamp: str | None = None
    source_monotonic_ns: int | None = None
    received_at: str | None = None
    targets: tuple[RadarTarget, ...] = ()
    radar_config_name: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("RadarFrame.timestamp must include a timezone offset")
        if not self.device_id.strip():
            raise ValueError("RadarFrame.device_id must not be blank")


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """RadarFrame与窗口/模型之间唯一允许的数据接口。"""

    timestamp: datetime
    device_id: str
    room: Room
    source_mode: SourceMode
    human_present: bool
    version: str
    names: tuple[str, ...]
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("FeatureVector.timestamp must include a timezone offset")
        if not self.device_id.strip():
            raise ValueError("FeatureVector.device_id must not be blank")
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("FeatureVector.values must be one-dimensional")
        if len(self.names) != values.shape[0]:
            raise ValueError("FeatureVector.names and values must have equal length")
        if not np.isfinite(values).all():
            raise ValueError("FeatureVector.values must contain only finite values")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class RadarRiskResult:
    room: Room
    device_id: str
    timestamp: datetime
    source_mode: SourceMode
    human_state: HumanState
    risk_score: float
    model_mode: ModelMode
    disclaimer: str | None
    event_triggered: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "room": self.room.value,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "source_mode": self.source_mode.value,
            "human_state": self.human_state.value,
            "risk_score": float(self.risk_score),
            "model_mode": self.model_mode.value,
            "disclaimer": self.disclaimer,
            "event_triggered": self.event_triggered,
        }
