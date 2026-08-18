"""Summarize a continuous live API recording for the final radar demo."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    by_timestamp: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                payload = row.get("payload") or {}
                prediction = payload.get("calibrated_tcn_prediction")
                if not isinstance(prediction, dict):
                    continue
                timestamp = prediction.get("timestamp")
                if not isinstance(timestamp, str):
                    continue
                by_timestamp[timestamp] = {
                    "sampled_at": row.get("sampled_at"),
                    "prediction": prediction,
                    "sensor_metrics": payload.get("sensor_metrics") or {},
                }
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return sorted(
        by_timestamp.values(),
        key=lambda item: _parse_time(item["prediction"]["timestamp"]),
    )


def _count_imminent_entries(rows: list[dict[str, Any]]) -> int:
    count = 0
    previous = None
    for row in rows:
        state = row["prediction"].get("gate_state")
        if state == "IMMINENT" and previous != "IMMINENT":
            count += 1
        previous = state
    return count


def build_report(
    input_path: Path,
    *,
    fall_start: datetime,
    fall_end: datetime,
) -> dict[str, Any]:
    rows = _read_predictions(input_path)
    fall_rows = [
        row
        for row in rows
        if fall_start
        <= _parse_time(row["prediction"]["timestamp"])
        <= fall_end
    ]
    normal_rows = [
        row
        for row in rows
        if not (
            fall_start
            <= _parse_time(row["prediction"]["timestamp"])
            <= fall_end
        )
    ]
    valid_normal = [
        row for row in normal_rows if bool(row["prediction"].get("score_valid"))
    ]
    normal_peak = max(
        valid_normal,
        key=lambda row: float(row["prediction"].get("pre_fall_score", 0.0)),
        default=None,
    )
    valid_fall = [
        row for row in fall_rows if bool(row["prediction"].get("score_valid"))
    ]
    fall_peak = max(
        valid_fall,
        key=lambda row: float(row["prediction"].get("pre_fall_score", 0.0)),
        default=None,
    )
    confirmation_rows = sorted(
        [row for row in fall_rows if row["prediction"].get("confirmed_at")],
        key=lambda row: _parse_time(str(row["prediction"]["confirmed_at"])),
    )
    first_confirmation = confirmation_rows[0]["prediction"] if confirmation_rows else None
    crossed = sorted(
        {
            str(row["prediction"]["threshold_crossed_at"])
            for row in fall_rows
            if row["prediction"].get("threshold_crossed_at")
        },
        key=_parse_time,
    )
    qualities = Counter(
        str(row["prediction"].get("data_quality", "UNKNOWN")) for row in rows
    )
    total = len(rows)
    return {
        "schema_version": "radar_live_action_validation_report_v1",
        "source_file": str(input_path.resolve()),
        "deduplication_key": "calibrated_tcn_prediction.timestamp",
        "recording": {
            "first_timestamp": rows[0]["prediction"]["timestamp"] if rows else None,
            "last_timestamp": rows[-1]["prediction"]["timestamp"] if rows else None,
            "unique_prediction_windows": total,
        },
        "fall_interval": {
            "start": fall_start.isoformat(),
            "end": fall_end.isoformat(),
            "alignment": "operator-command timeline; no external impact ground truth",
        },
        "normal_actions": {
            "maximum_score": (
                float(normal_peak["prediction"]["pre_fall_score"])
                if normal_peak
                else None
            ),
            "maximum_score_at": (
                normal_peak["prediction"]["timestamp"] if normal_peak else None
            ),
            "imminent_trigger_count": _count_imminent_entries(normal_rows),
        },
        "controlled_fall": {
            "maximum_score": (
                float(fall_peak["prediction"]["pre_fall_score"])
                if fall_peak
                else None
            ),
            "maximum_score_at": (
                fall_peak["prediction"]["timestamp"] if fall_peak else None
            ),
            "imminent_trigger_count": _count_imminent_entries(fall_rows),
            "first_threshold_crossed_at": crossed[0] if crossed else None,
            "confirmation_sequence_started_at": (
                first_confirmation.get("threshold_crossed_at")
                if first_confirmation
                else None
            ),
            "first_confirmed_at": (
                first_confirmation.get("confirmed_at")
                if first_confirmation
                else None
            ),
            "confirmation_latency_seconds": (
                (
                    _parse_time(str(first_confirmation["confirmed_at"]))
                    - _parse_time(str(first_confirmation["threshold_crossed_at"]))
                ).total_seconds()
                if first_confirmation
                and first_confirmation.get("threshold_crossed_at")
                else None
            ),
        },
        "data_quality": {
            "counts": dict(sorted(qualities.items())),
            "percentages": {
                key: value / total if total else 0.0
                for key, value in sorted(qualities.items())
            },
        },
        "final_output_contract": {
            "risk_source": "frozen calibrated TCN score + 3-window decision gate",
            "descent_branch": "debug_only",
            "rule_risk_branch": "debug_only",
            "score_is_probability": False,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    normal = report["normal_actions"]
    fall = report["controlled_fall"]
    quality = report["data_quality"]
    lines = [
        "# IWR6843连续动作验证报告",
        "",
        "本报告对API轮询中的重复窗口按模型时间戳去重。跌倒区间来自操作指令时间线，未使用视频或接触垫作为撞击真值。",
        "",
        "## 结果",
        "",
        f"- 本次连续记录正常动作最高TCN score：`{normal['maximum_score']:.6f}`（{normal['maximum_score_at']}）。",
        f"- 正常动作IMMINENT触发次数：`{normal['imminent_trigger_count']}`。",
        f"- 受控跌倒最高TCN score：`{fall['maximum_score']:.6f}`（{fall['maximum_score_at']}）。",
        f"- 受控跌倒IMMINENT触发次数：`{fall['imminent_trigger_count']}`。",
        f"- 首次越过阈值：`{fall['first_threshold_crossed_at']}`。",
        f"- 连续确认序列起点：`{fall['confirmation_sequence_started_at']}`。",
        f"- 首次连续确认：`{fall['first_confirmed_at']}`。",
        f"- 连续确认附加延迟：`{fall['confirmation_latency_seconds']:.3f} s`。",
        "",
        "## 数据质量",
        "",
    ]
    for key, count in quality["counts"].items():
        lines.append(
            f"- {key}: `{count}` 个唯一窗口（{quality['percentages'][key] * 100:.1f}%）。"
        )
    lines.extend(
        [
            "",
            "## 最终展示契约",
            "",
            "- 最终雷达风险证据仅来自冻结校准TCN score与连续3窗口门控。",
            "- 下降检测和旧规则风险仅保留为debug，不改变risk_state且不参与告警。",
            "- score是模型连续输出，不表述为跌倒概率。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fall-start", required=True)
    parser.add_argument("--fall-end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.input,
        fall_start=_parse_time(args.fall_start),
        fall_end=_parse_time(args.fall_end),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
