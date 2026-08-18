# 域校准 TCN shadow 实时启动手册

更新日期：2026-08-10

## 目的

在 Windows 本机启动雷达 FastAPI，并启用"域校准 TCN + 决策门控"shadow 分支，
让真机/回放分数从 1e-11 塌缩恢复到正常 0-1 量级，同时用恢复门控压制坐下/蹲下
等受控动作误报。

不修改冻结 TCN checkpoint、阈值、特征提取器或正式告警链路。
输出固定为 `shadow_only=true`、`alert_suppressed=true`。

## 前置条件

1. 已创建校准归一化文件：
   `reports/domain_calibration_v1_full/calibrated_normalization_real_gaussian.json`
   （由 `domain_calibration_v1.py` 生成）。
2. 冻结 TCN checkpoint 存在且 SHA256 匹配：
   `checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt`
   SHA256 `0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef`。
3. 使用项目训练 Python：`E:\python3.10.9aaa\python.exe`。

## 一键启动（REPLAY 演示）

在 `雷达模块` 目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_calibrated_tcn_demo.ps1
```

该脚本会：
- 关闭 PointNet shadow（README 要求 TCN 不与 PointNet 同时启用）；
- 启用 calibrated TCN shadow；
- 前台运行 uvicorn on `127.0.0.1:8010`。

启动后验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
# 应看到 calibrated_tcn_shadow_enabled: True

# 回放一段真机会话
Copy-Item reports\real_scene_validation_v1\person_walking_single_start_20260809\session.jsonl data\replay\_calib_walk_demo.jsonl
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8010/api/radar/replay `
  -ContentType application/json `
  -Body '{"file_name":"_calib_walk_demo.jsonl","speed":20.0,"loop":false}'

# 查看实时结果
Invoke-RestMethod http://127.0.0.1:8010/api/radar/latest
```

`/api/radar/latest` 返回：

```json
{
  "calibrated_tcn_prediction": {
    "schema_version": "radar_calibrated_tcn_live_v1",
    "pre_fall_score": 0.11,
    "gate_state": "NORMAL",
    "tcn_risk_state": "NORMAL",
    "data_quality": "GOOD",
    "formal_alert": false,
    "shadow_only": true,
    "alert_suppressed": true
  },
  "tcn_baseline": { "...": "原始冻结TCN，分数可能仍接近1e-11（塌缩）" }
}
```

## 手动启动（等价）

```powershell
cd "E:\创新实践\老人摔倒预警\雷达模块"
$env:RADAR_POINTNET_SHADOW_ENABLED = "false"
$env:RADAR_TCN_SHADOW_ENABLED = "true"
$env:RADAR_TCN_CHECKPOINT_PATH = ".\checkpoints\experiments_v5\tcn_hard_negative\tcn_0p5_1p0_specificity_operating_point_v1.pt"
$env:RADAR_TCN_CHECKPOINT_SHA256 = "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"
$env:RADAR_CALIBRATED_TCN_SHADOW_ENABLED = "true"
$env:RADAR_CALIBRATED_TCN_CALIBRATION_PATH = ".\reports\domain_calibration_v1_full\calibrated_normalization_real_gaussian.json"
& "E:\python3.10.9aaa\python.exe" -m uvicorn radar_module.service.radar_api:app --host 127.0.0.1 --port 8010
```

## 真机 REAL 模式

REAL 模式依赖 Windows 串口 COM5/COM6 与 TI 官方桥接进程。在能访问串口的
Windows 前台终端运行：

```powershell
# 先确保 COM5/COM6 未被 UniFlash / Industrial Visualizer 占用
Invoke-RestMethod -Method Post http://127.0.0.1:8010/api/radar/real
```

启动后 `/api/radar/latest` 同样返回 `calibrated_tcn_prediction`。
若板卡 CLI 对 sensorStop 无响应，需冷启动板卡后重试，不能跳过 TI 配置命令。

## 环境变量

| 变量 | 说明 |
|---|---|
| RADAR_CALIBRATED_TCN_SHADOW_ENABLED | 启用 calibrated TCN shadow |
| RADAR_CALIBRATED_TCN_CALIBRATION_PATH | 校准归一化 JSON 路径 |
| RADAR_CALIBRATED_TCN_CALIBRATION_METHOD | 默认 real_gaussian |
| RADAR_CALIBRATED_TCN_CONFIRMATION_WINDOWS | 连续高窗数，默认 3 |
| RADAR_CALIBRATED_TCN_RECOVERY_WINDOWS | 恢复门控连续低窗数，默认 2 |
| RADAR_CALIBRATED_TCN_RECOVERY_WINDOW_SECONDS | 恢复窗口时长，默认 1.5 |

## 边界与诚实声明

- calibrated TCN 是 shadow/demo 分支，不生成正式 `PRE_FALL_RISK` 事件。
- 域校准让"正常 vs 误报"变得干净，但没有让"跌倒"可检出：真机受控跌倒
  （从静止直接倒、无前兆）在校准后仍为 NORMAL。要检出跌倒仍需带前兆的
  数据和 time-to-impact 目标函数。
- 任何校准归一化要用于正式判断，必须在独立留出受试者上验证。
