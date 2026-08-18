"""Tests for sensor-to-world coordinate audit."""

from __future__ import annotations

import numpy as np
import pytest

from radar_module.analysis.sensor_to_world_audit_v1 import (
    euler_rot,
    to_world,
)


def test_euler_rot_no_tilt_identity() -> None:
    x, y, z = euler_rot(1.0, 2.0, 0.5, 0.0, 0.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)
    assert z == pytest.approx(0.5)


def test_euler_rot_positive_elevation_lowers_z() -> None:
    # 传感器向下俯视(正 elev_tilt)，远端点的 z 应降低（相对 x 轴旋转）
    x, y, z = euler_rot(0.0, 2.0, 0.3, 15.0, 0.0)
    # y 沿视线方向，elev 旋转后 z 分量变小
    assert z < 0.3


def test_to_world_adds_sensor_height() -> None:
    pts = [{"x": 0.1, "y": 1.0, "z": 0.2}]
    xs, ys, zs = to_world(pts, sensor_height_m=1.0, elev_tilt_deg=0.0, azi_tilt_deg=0.0)
    assert zs[0] == pytest.approx(1.2)
    assert xs[0] == pytest.approx(0.1)


def test_to_world_with_tilt_changes_z() -> None:
    pts = [{"x": 0.0, "y": 2.0, "z": 0.0}]
    _, _, zs = to_world(pts, sensor_height_m=1.0, elev_tilt_deg=15.0, azi_tilt_deg=0.0)
    # 无 tilt 时 z=1.0；15° tilt 后 z < 1.0（远端下倾）
    assert zs[0] < 1.0


def test_raw_vs_world_standing_z_difference() -> None:
    """验证 raw(0高度) 与 world(1m) 的 centroid_z 差约为 sensorHeight。"""
    pts = [
        {"x": 0.0, "y": 1.0, "z": 0.0},
        {"x": 0.0, "y": 1.0, "z": 0.5},
    ]
    _, _, raw_z = to_world(pts, sensor_height_m=0.0, elev_tilt_deg=0.0, azi_tilt_deg=0.0)
    _, _, world_z = to_world(pts, sensor_height_m=1.0, elev_tilt_deg=0.0, azi_tilt_deg=0.0)
    assert np.mean(world_z) - np.mean(raw_z) == pytest.approx(1.0)
