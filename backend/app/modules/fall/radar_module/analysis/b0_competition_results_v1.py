"""Reproduce the frozen B0 competition metrics and presentation figures.

This module is deliberately evaluation-only.  It never trains, calibrates, or
rewrites a checkpoint and it plots the raw sigmoid score on a fixed [0, 1]
axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402

from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.dguha_event_evaluation_v2 import (
    _build_prefall_model,
    _confirmed_run_end_indices,
    _score_recording,
)
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
)


THRESHOLD = 0.35
CONFIRMATION_WINDOWS = 3
STEP_SECONDS = 0.1
SUCCESS_SOURCE = "Test/5_falling_forward/radar/M_012_A5_005.txt"
MISS_SOURCE = "Test/5_falling_forward/radar/M_012_A5_004.txt"
SIT_SOURCE = "Test/3_Sit_down_and_stand_up/radar/M_012_A3_003.txt"
ONLINE_B0_SHA256 = "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_chinese_font() -> None:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(candidate)
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _score_dguha_recording(
    source_root: Path,
    relative_source: str,
    checkpoint: dict[str, Any],
    *,
    maximum_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    frames = parse_radhar_text(
        source_root / relative_source,
        device_id=f"b0-figure-{Path(relative_source).stem}",
    )
    start = frames[0].timestamp
    duration = (frames[-1].timestamp - start).total_seconds()
    end = min(duration, maximum_end) if maximum_end is not None else duration
    extractor = RadarTemporalFeatureExtractorV2()
    endpoints = np.arange(
        extractor.history_seconds - 1.0 / extractor.target_sample_rate_hz,
        end + STEP_SECONDS * 0.25,
        STEP_SECONDS,
        dtype=np.float64,
    )
    model = _build_prefall_model(checkpoint)
    times, scores, _ = _score_recording(
        frames,
        endpoints,
        start,
        extractor,
        model,
        np.asarray(checkpoint["normalization_mean"], dtype=np.float32),
        np.asarray(checkpoint["normalization_std"], dtype=np.float32),
        use_relative_features_v3=False,
        use_hybrid_features_v4=False,
    )
    return times, scores, duration


def _trigger_times(times: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ends = _confirmed_run_end_indices(
        scores >= THRESHOLD,
        times,
        CONFIRMATION_WINDOWS,
        STEP_SECONDS,
    )
    return times[np.asarray(ends, dtype=np.int64)] if ends else np.asarray([])


def _plot_score_curve(
    times: np.ndarray,
    scores: np.ndarray,
    destination: Path,
    *,
    title: str,
    x_label: str,
    trigger_times: np.ndarray,
    onset_at_zero: bool,
    note: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    axis.plot(times, scores, color="#2457a6", linewidth=2.0, label="B0 原始 score")
    axis.axhline(
        THRESHOLD,
        color="#b42318",
        linestyle="--",
        linewidth=1.5,
        label="固定阈值 0.35",
    )
    if onset_at_zero:
        axis.axvline(0.0, color="#101828", linewidth=1.2, label="下降起点")
        axis.axvspan(-1.0, -0.5, color="#f79009", alpha=0.12, label="目标提前区间")
    for index, trigger in enumerate(trigger_times):
        score = float(np.interp(trigger, times, scores))
        axis.scatter([trigger], [score], color="#b42318", zorder=4, s=36)
        if index == 0:
            trigger_label = (
                "目标区间内三窗口确认"
                if onset_at_zero and -1.0 <= trigger <= -0.5
                else "目标区间外三窗口确认"
                if onset_at_zero
                else "三窗口确认"
            )
            axis.annotate(
                f"{trigger_label}\n{trigger:.2f} s",
                xy=(trigger, score),
                xytext=(10, 18),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#b42318"},
            )
    axis.set(xlabel=x_label, ylabel="TCN 短时预测分数", ylim=(0.0, 1.0), title=title)
    axis.grid(axis="y", color="#d0d5dd", linewidth=0.7, alpha=0.65)
    axis.legend(loc="upper left", frameon=False, ncol=2)
    axis.text(
        0.995,
        0.02,
        note,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#475467",
        fontsize=8.5,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _score_external_squat(
    dataset_path: Path,
    source_root: Path,
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        origin = np.asarray(dataset["dataset_origin"]).astype(str)
        actions = np.asarray(dataset["action"]).astype(str)
        sources = np.asarray(dataset["source_files"]).astype(str)
        features = np.asarray(dataset["features"], dtype=np.float32)
    selected = (origin == "radhar") & np.isin(actions, ["squat", "squats"])
    if not np.any(selected):
        raise ValueError("RadHAR squat samples are unavailable")
    model = _build_prefall_model(checkpoint)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    with torch.inference_mode():
        scores = torch.sigmoid(
            model(torch.from_numpy((features[selected] - mean) / std))
        ).numpy()
    selected_sources = sources[selected]
    best_source = max(
        np.unique(selected_sources),
        key=lambda source: float(np.max(scores[selected_sources == source])),
    )
    raw_times, raw_scores, _ = _score_dguha_recording(
        source_root,
        str(best_source),
        checkpoint,
    )
    return raw_times, raw_scores, str(best_source)


def _write_hard_action_figure(
    sit_times: np.ndarray,
    sit_scores: np.ndarray,
    squat_times: np.ndarray,
    squat_scores: np.ndarray,
    squat_source: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, times, scores, title in (
        (axes[0], sit_times, sit_scores, "困难负样本：正常坐下/起身（DGUHA）"),
        (axes[1], squat_times, squat_scores, "困难负样本：下蹲（RadHAR 外部正常动作）"),
    ):
        axis.plot(times, scores, color="#2457a6", linewidth=1.8)
        axis.axhline(THRESHOLD, color="#b42318", linestyle="--", linewidth=1.4)
        triggers = _trigger_times(times, scores)
        if len(triggers):
            axis.scatter(
                triggers,
                np.interp(triggers, times, scores),
                color="#b42318",
                s=28,
                zorder=4,
                label="三窗口确认",
            )
        axis.set(title=title, xlabel="录像时间（s）", ylabel="TCN 短时预测分数", ylim=(0, 1))
        axis.grid(axis="y", color="#d0d5dd", linewidth=0.7, alpha=0.65)
        axis.legend(["B0 原始 score", "固定阈值 0.35", "三窗口确认"], loc="upper right", frameon=False)
    axes[1].text(
        0.995,
        0.02,
        f"RadHAR source: {squat_source}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="#475467",
        fontsize=8,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def build_report(root: Path, output: Path) -> dict[str, Any]:
    _configure_chinese_font()
    checkpoint_path = root / "checkpoints/experiments_v5/tcn_hard_negative/tcn_0p5_1p0_specificity_operating_point_v1.pt"
    event_report_path = root / "reports/tcn_hard_negative_v1/baseline_specificity_operating_point_0p35.json"
    audit_path = root / "reports/multisource_training_v1/B0_audit.json"
    multisource_report = root / "reports/multisource_training_v1/MULTISOURCE_TRAINING_REPORT.md"
    events_path = root / "data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json"
    source_root = root / "data/external/dguha/raw"
    dataset_path = root / "data/processed/experiments_v7/tcn_multisource_m1_clean_v1.npz"
    output.mkdir(parents=True, exist_ok=True)

    actual_hash = _sha256(checkpoint_path)
    if actual_hash != ONLINE_B0_SHA256:
        raise ValueError(f"B0 SHA256 changed: {actual_hash}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if float(checkpoint["decision_threshold"]) != THRESHOLD:
        raise ValueError("B0 threshold contract changed")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    by_source = {str(event["source_file"]): event for event in events}
    curves: dict[str, dict[str, Any]] = {}
    for key, source in (("success", SUCCESS_SOURCE), ("miss", MISS_SOURCE)):
        event = by_source[source]
        frames = parse_radhar_text(source_root / source)
        onset_seconds = (
            datetime.fromisoformat(str(event["descent_onset"])) - frames[0].timestamp
        ).total_seconds()
        times, scores, _ = _score_dguha_recording(
            source_root,
            source,
            checkpoint,
            maximum_end=onset_seconds - 0.1,
        )
        relative_times = times - onset_seconds
        triggers = _trigger_times(relative_times, scores)
        curves[key] = {
            "source": source,
            "times": relative_times,
            "scores": scores,
            "triggers": triggers,
        }

    _plot_score_curve(
        curves["success"]["times"],
        curves["success"]["scores"],
        output / "01_successful_fall_prediction.png",
        title="B0 成功短时跌倒预测案例",
        x_label="相对下降起点时间（s）",
        trigger_times=curves["success"]["triggers"],
        onset_at_zero=True,
        note=f"source: {SUCCESS_SOURCE}",
    )
    _plot_score_curve(
        curves["miss"]["times"],
        curves["miss"]["scores"],
        output / "02_missed_fall_prediction.png",
        title="B0 漏检案例：目标区间未形成三窗口确认",
        x_label="相对下降起点时间（s）",
        trigger_times=curves["miss"]["triggers"],
        onset_at_zero=True,
        note=f"source: {MISS_SOURCE}",
    )
    sit_times, sit_scores, _ = _score_dguha_recording(
        source_root, SIT_SOURCE, checkpoint
    )
    squat_times, squat_scores, squat_source = _score_external_squat(
        dataset_path,
        root / "data/external/radhar/Data",
        checkpoint,
    )
    _write_hard_action_figure(
        sit_times,
        sit_scores,
        squat_times,
        squat_scores,
        squat_source,
        output / "03_hard_normal_actions.png",
    )

    event_report = json.loads(event_report_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "threshold": THRESHOLD,
        "confirmation_windows": CONFIRMATION_WINDOWS,
        "dguha_validation": {
            "event_recall": event_report["prediction_corridor_event_recall"],
            "event_count": event_report["eligible_fall_recording_count"],
            "detected_event_count": event_report["prediction_corridor_detected_event_count"],
            "median_lead_seconds": event_report["corridor_confirmation_lead_seconds"]["median"],
            "window_auroc": audit["dguha_validation"]["auroc"],
            "window_f1": audit["dguha_validation"]["f1"],
            "same_recording_early_false_positive_rate": event_report["same_recording_early_negative_false_positive_rate"],
            "normal_confirmed_false_alarms_per_hour": event_report["normal_confirmed_runs_per_hour"],
            "normal_recordings_with_confirmed_run": event_report["normal_recordings_with_confirmed_run"],
            "normal_recording_count": event_report["normal_recording_count"],
        },
        "external_normal": audit["external_normal"]["aggregate"],
        "multisource_report": str(multisource_report),
        "figure_cases": {
            "success": SUCCESS_SOURCE,
            "miss": MISS_SOURCE,
            "sit": SIT_SOURCE,
            "squat": squat_source,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = f"""# B0 毫米波雷达短时运动风险感知模块：比赛展示摘要

## 冻结契约

- 模型：B0 causal TCN（20 帧 × 19 特征，hidden size 24）
- checkpoint SHA256：`{actual_hash}`
- 固定阈值：`0.35`
- 触发策略：连续 3 个有效窗口达到阈值
- 状态：`shadow_only=true`；不触发正式告警，不输出最终跌倒风险

## 严格验证结果

| 指标 | B0 结果 |
|---|---:|
| DGUHA 事件级召回 | 8/14（57.1%） |
| 命中事件提前量中位数 | 0.722 s |
| DGUHA 窗口 AUROC | 0.7552 |
| DGUHA 窗口 F1 | 0.1019 |
| 同录像早期负样本误报 | 75.7% |
| DGUHA normal 连续误报 | 170.0 次/小时 |
| DGUHA normal 发生确认触发的录像 | 27/108 |
| 外部 normal 连续误报 | 19.83 次/小时 |

上述误报指标揭示模型仍会把坐下、下蹲及跌倒录像较早阶段的“下降状态”误认为跌倒前演化。因此比赛中只将其定义为**短时运动风险证据**，不宣称单雷达可完成高可靠跌倒判断。

## 困难负样本与多源实验结论

困难负样本训练未改变结构，加入了同录像早期窗口、坐下/快速坐下代理、下蹲、弯腰、跪地/坐地等正常下降动作，并采用标签可信度权重；在 0.35 特异性优先操作点仍存在显著误报。

M1（多源干净）和 M2（多源+轻量域增强）均失败：事件召回从 B0 的 8/14 降到 6/14，同录像早期误报升至 85.8%/87.0%，DGUHA normal 连续误报升至 190.4/273.7 次/小时。它们虽然压低了外部 normal 分数，却同时压低真实 IWR6843 受控跌倒响应，属于分数塌缩而非泛化改善，因此未替换 B0。

## 最终算法表述

毫米波分支持续输出身份无关的 `radar_score`、`risk_state` 和质量信息，用于感知约 0.5–1.0 秒尺度的运动状态变化。它是后续摄像头姿态/行为风险与环境风险融合时的一项独立证据，不直接生成系统最终跌倒风险。

```json
{{
  "radar_score": 0.287,
  "risk_state": "WATCH",
  "timestamp": "2026-08-09T15:27:17.394+08:00",
  "room": "bathroom",
  "device_id": "iwr6843_revd_bathroom",
  "quality": "GOOD",
  "model_version": "radar_temporal_experiment_v3"
}}
```

当 `risk_state=UNKNOWN` 时，`radar_score=null`，不能把质量不足显示成 0 分。接口同时保留冻结的 `tcn_prediction` 载荷以兼容现有实时链路，但不再把它映射为通用 `risk_score`。

比赛页面明确区分 `DGUHA Offline Replay` 与 `IWR6843 Real Sensor`，同时显示原始 score 趋势、0.35 阈值、状态、质量、帧率、点数和模型版本。score 图固定使用 0–1 纵轴，不进行概率换算、平滑或局部放大。

比赛中的预测效果演示使用公开数据成功事件 `M_012_A5_005` 导出的标准 JSONL，而不使用自采 IWR6843 受控跌倒录像。该回放仍经过完整实时推理链路，实测 score 为 `0.7320 → 0.6402 → 0.6008`，第三个连续高分窗口进入 `IMMINENT`；实时推理与独立批推理的最大绝对差异为 `5.96e-8`。

雷达服务启动后，在雷达模块目录运行：

```powershell
.\start_b0_offline_replay.ps1
```

脚本使用 `speed=0.1, loop=false`，一次回放约持续 25 秒，避免循环回放时源时间戳倒退。页面必须显示 `DGUHA Offline Replay`，不得把它描述为 IWR6843 真机预测。

真机 IWR6843 中极小 score 是已确认的 DGUHA→IWR6843 正样本域偏移表现，不是百分比显示错误，也不能通过放大数值修复。公开数据离线回放用于展示算法能力；真机模式用于展示实时采集、预处理、推理和质量闭环。
"""
    (output / "B0_ALGORITHM_SUMMARY.md").write_text(summary, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "reports/competition_b0_v1",
    )
    args = parser.parse_args()
    metrics = build_report(args.root.resolve(), args.output.resolve())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
