from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.stats import linregress, mannwhitneyu, spearmanr, wilcoxon

from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT
from radar_module.dataset.iwr6843_fall_v1 import parse_iwr6843_fall_csv
from radar_module.dataset.radhar_converter import parse_radhar_text


ANALYSIS_VERSION = "dguha_motion_evolution_v1"
METRICS = (
    "height_proxy_m",
    "centroid_speed_mps",
    "spatial_dispersion_m",
    "doppler_abs_mean_mps",
    "doppler_std_mps",
    "point_count",
)
METRIC_LABELS = {
    "height_proxy_m": "Upper-height proxy, $z_{90}$ (m)",
    "centroid_speed_mps": "3D centroid speed (m/s)",
    "spatial_dispersion_m": "Point-cloud RMS dispersion (m)",
    "doppler_abs_mean_mps": "Mean |Doppler| (m/s)",
    "doppler_std_mps": "Doppler SD (m/s)",
    "point_count": "Detected point count",
}
TIME_BINS = (
    (-2.0, -1.5, "fall_early_2p0_1p5"),
    (-1.5, -1.2, "fall_early_1p5_1p2"),
    (-1.0, -0.5, "prediction_1p0_0p5"),
    (-0.5, 0.0, "prediction_0p5_0p0"),
    (0.0, 0.5, "descent_0p0_0p5"),
    (0.5, 1.0, "descent_0p5_1p0"),
    (1.0, 2.0, "descent_1p0_2p0"),
)


def analyze_motion_evolution(
    data_root: str | Path,
    events_path: str | Path,
    output_directory: str | Path,
    *,
    iwr6843_root: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(data_root).resolve()
    event_file = Path(events_path).resolve()
    destination = Path(output_directory).resolve()
    if not source_root.is_dir() or not event_file.is_file():
        raise FileNotFoundError("DGUHA data root and event metadata must exist")
    destination.mkdir(parents=True, exist_ok=True)

    events = json.loads(event_file.read_text(encoding="utf-8"))
    development_events = [
        event
        for event in events
        if bool(event["eligible_for_prediction_windows"])
        and event["project_split"] in {"train", "validation"}
    ]
    heldout_events = [
        event
        for event in events
        if bool(event["eligible_for_prediction_windows"])
        and event["project_split"] == "test"
    ]
    if not development_events:
        raise ValueError("no eligible development fall events")

    fall_frames: list[pd.DataFrame] = []
    anchor_rows: list[dict[str, Any]] = []
    for event in development_events:
        relative = str(event["source_file"])
        frames = parse_radhar_text(
            source_root / Path(relative),
            device_id=f"evolution-{Path(relative).stem}",
        )
        series = _recording_series(frames)
        radar_start = datetime.fromisoformat(str(event["radar_start"]))
        onset_seconds = (
            datetime.fromisoformat(str(event["descent_onset"])) - radar_start
        ).total_seconds()
        near_floor_seconds = (
            datetime.fromisoformat(str(event["near_floor_level_reached"]))
            - radar_start
        ).total_seconds()
        near_floor_relative = near_floor_seconds - onset_seconds
        selected = _align_series(
            series,
            anchor_seconds=onset_seconds,
            start_relative=-2.0,
            end_relative=near_floor_relative,
        )
        selected["recording_id"] = relative
        selected["subject_id"] = str(event["subject_id"])
        selected["split"] = str(event["project_split"])
        selected["group"] = "dguha_fall"
        selected["action"] = "falling_forward"
        selected["anchor_definition"] = "skeleton_derived_descent_onset"
        selected["near_floor_relative_seconds"] = near_floor_relative
        selected["descent_phase"] = np.where(
            selected["relative_seconds"] >= 0,
            selected["relative_seconds"] / max(near_floor_relative, 1e-6),
            np.nan,
        )
        fall_frames.append(_add_baseline_deltas(selected))
        anchor_rows.append(
            {
                "recording_id": relative,
                "subject_id": str(event["subject_id"]),
                "split": str(event["project_split"]),
                "descent_onset_seconds": onset_seconds,
                "near_floor_seconds": near_floor_seconds,
                "onset_to_near_floor_seconds": near_floor_relative,
                "near_floor_is_impact_ground_truth": False,
            }
        )
    fall = pd.concat(fall_frames, ignore_index=True)

    normal_frames: list[pd.DataFrame] = []
    radar_files = sorted(source_root.glob("*/*/radar/*.txt"))
    for radar_path in radar_files:
        action = radar_path.parent.parent.name
        if action == "5_falling_forward":
            continue
        subject = _subject_id(radar_path.name)
        split = DGUHA_SPLIT_BY_SUBJECT[subject]
        if split == "test":
            continue
        frames = parse_radhar_text(
            radar_path, device_id=f"normal-evolution-{radar_path.stem}"
        )
        series = _recording_series(frames)
        anchor_seconds = _normal_motion_anchor(series, action)
        selected = _align_series(
            series,
            anchor_seconds=anchor_seconds,
            start_relative=-2.0,
            end_relative=2.0,
        )
        if len(selected) < 20:
            continue
        relative = radar_path.relative_to(source_root).as_posix()
        selected["recording_id"] = relative
        selected["subject_id"] = subject
        selected["split"] = split
        selected["group"] = "dguha_normal"
        selected["action"] = action
        selected["anchor_definition"] = (
            "radar_maximum_downward_centroid_velocity"
            if action == "3_Sit_down_and_stand_up"
            else "radar_maximum_centroid_speed"
        )
        selected["near_floor_relative_seconds"] = np.nan
        selected["descent_phase"] = np.nan
        normal_frames.append(_add_baseline_deltas(selected))
    normal = pd.concat(normal_frames, ignore_index=True)

    iwr = _iwr_normal_comparison(iwr6843_root) if iwr6843_root else pd.DataFrame()
    all_rows = pd.concat((fall, normal, iwr), ignore_index=True, sort=False)
    all_rows.to_csv(destination / "motion_evolution_timeseries.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(
        destination / "development_event_anchors.csv", index=False
    )

    event_statistics = _event_statistics(fall)
    event_statistics.to_csv(destination / "fall_event_trend_statistics.csv", index=False)
    group_statistics = _group_statistics(fall, normal)
    group_statistics.to_csv(destination / "group_comparison_statistics.csv", index=False)
    time_bin_summary = _time_bin_summary(all_rows)
    time_bin_summary.to_csv(destination / "time_bin_feature_summary.csv", index=False)

    _plot_aggregate_trends(fall, normal, destination)
    _plot_event_heatmaps(fall, destination)
    _plot_group_comparison(fall, normal, iwr, destination)
    _plot_individual_events(fall, destination)

    verdict = _evolution_verdict(event_statistics, group_statistics)
    near_floor_durations = np.asarray(
        [row["onset_to_near_floor_seconds"] for row in anchor_rows], dtype=np.float64
    )
    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "data_root": str(source_root),
        "events_file": str(event_file),
        "output_directory": str(destination),
        "development_split_policy": "train+validation only",
        "eligible_development_fall_event_count": len(development_events),
        "heldout_test_event_count_not_analyzed": len(heldout_events),
        "normal_recording_count": int(normal["recording_id"].nunique()),
        "normal_action_counts": {
            str(key): int(value)
            for key, value in normal.groupby("action")["recording_id"].nunique().items()
        },
        "iwr6843_supplemental_recording_count": (
            int(iwr["recording_id"].nunique()) if len(iwr) else 0
        ),
        "feature_definitions": {
            "height_proxy_m": "90th percentile radar point z; not anatomical height",
            "centroid_speed_mps": "3D speed of smoothed radar point centroid",
            "spatial_dispersion_m": "root-mean-square point distance from frame centroid",
            "doppler_abs_mean_mps": "mean absolute measured radial velocity",
            "doppler_std_mps": "standard deviation of measured radial velocity",
            "point_count": "number of detected points in the aligned radar frame",
        },
        "fall_alignment": "skeleton-derived descent_onset",
        "normal_sit_alignment": "radar-derived maximum downward centroid velocity",
        "other_normal_alignment": "radar-derived maximum 3D centroid speed",
        "near_floor_anchor_warning": (
            "near_floor_level_reached is a skeleton-derived state proxy, not an "
            "annotated impact time"
        ),
        "onset_to_near_floor_seconds": _describe(near_floor_durations),
        "verdict": verdict,
        "deployment_eligible": False,
        "model_training_performed": False,
        "test_split_inspected": False,
    }
    (destination / "motion_evolution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _recording_series(frames) -> pd.DataFrame:
    start = frames[0].timestamp
    source_seconds = np.asarray(
        [(frame.timestamp - start).total_seconds() for frame in frames],
        dtype=np.float64,
    )
    target_seconds = np.arange(0.0, source_seconds[-1] + 0.025, 0.1)
    nearest = np.searchsorted(source_seconds, target_seconds)
    rows: list[dict[str, float]] = []
    for target, right in zip(target_seconds, nearest):
        candidates = [index for index in (right - 1, right) if 0 <= index < len(frames)]
        source_index = min(candidates, key=lambda index: abs(source_seconds[index] - target))
        if abs(source_seconds[source_index] - target) > 0.06:
            rows.append({name: np.nan for name in _raw_feature_names()})
            continue
        rows.append(_frame_features(frames[source_index].points))
    table = pd.DataFrame(rows)
    table.insert(0, "recording_seconds", target_seconds)
    for name in _raw_feature_names():
        table[name] = table[name].interpolate(limit=2, limit_direction="both")
    centroid = table[["centroid_x_m", "centroid_y_m", "centroid_z_m"]].to_numpy()
    if len(table) >= 5 and np.isfinite(centroid).all():
        centroid = savgol_filter(centroid, window_length=5, polyorder=2, axis=0)
        speed = np.linalg.norm(np.gradient(centroid, 0.1, axis=0), axis=1)
        vertical_velocity = np.gradient(centroid[:, 2], 0.1)
    else:
        speed = np.full(len(table), np.nan)
        vertical_velocity = np.full(len(table), np.nan)
    table["centroid_speed_mps"] = speed
    table["centroid_vertical_velocity_mps"] = vertical_velocity
    return table


def _frame_features(points) -> dict[str, float]:
    if not points:
        return {name: (0.0 if name == "point_count" else np.nan) for name in _raw_feature_names()}
    coordinates = np.asarray([(point.x, point.y, point.z) for point in points], dtype=np.float64)
    velocities = np.asarray([point.velocity for point in points], dtype=np.float64)
    centroid = coordinates.mean(axis=0)
    dispersion = np.sqrt(np.mean(np.sum((coordinates - centroid) ** 2, axis=1)))
    return {
        "centroid_x_m": float(centroid[0]),
        "centroid_y_m": float(centroid[1]),
        "centroid_z_m": float(centroid[2]),
        "height_proxy_m": float(np.quantile(coordinates[:, 2], 0.90)),
        "vertical_extent_m": float(
            np.quantile(coordinates[:, 2], 0.90)
            - np.quantile(coordinates[:, 2], 0.10)
        ),
        "spatial_dispersion_m": float(dispersion),
        "doppler_mean_mps": float(velocities.mean()),
        "doppler_abs_mean_mps": float(np.abs(velocities).mean()),
        "doppler_std_mps": float(velocities.std()),
        "doppler_abs_p90_mps": float(np.quantile(np.abs(velocities), 0.90)),
        "point_count": float(len(points)),
    }


def _raw_feature_names() -> tuple[str, ...]:
    return (
        "centroid_x_m",
        "centroid_y_m",
        "centroid_z_m",
        "height_proxy_m",
        "vertical_extent_m",
        "spatial_dispersion_m",
        "doppler_mean_mps",
        "doppler_abs_mean_mps",
        "doppler_std_mps",
        "doppler_abs_p90_mps",
        "point_count",
    )


def _align_series(
    series: pd.DataFrame,
    *,
    anchor_seconds: float,
    start_relative: float,
    end_relative: float,
) -> pd.DataFrame:
    selected = series[
        (series["recording_seconds"] >= anchor_seconds + start_relative - 0.051)
        & (series["recording_seconds"] <= anchor_seconds + end_relative + 0.051)
    ].copy()
    selected["relative_seconds"] = selected["recording_seconds"] - anchor_seconds
    selected["relative_seconds"] = np.round(selected["relative_seconds"], 3)
    return selected


def _add_baseline_deltas(table: pd.DataFrame) -> pd.DataFrame:
    baseline = table[
        (table["relative_seconds"] >= -2.0)
        & (table["relative_seconds"] <= -1.5)
    ]
    result = table.copy()
    for metric in METRICS:
        center = float(baseline[metric].median()) if len(baseline) else np.nan
        result[f"{metric}_delta"] = result[metric] - center
    return result


def _normal_motion_anchor(series: pd.DataFrame, action: str) -> float:
    eligible = series[
        (series["recording_seconds"] >= 2.0)
        & (series["recording_seconds"] <= series["recording_seconds"].max() - 2.0)
    ]
    if not len(eligible):
        eligible = series
    if action == "3_Sit_down_and_stand_up":
        index = eligible["centroid_vertical_velocity_mps"].idxmin()
    else:
        index = eligible["centroid_speed_mps"].idxmax()
    return float(series.loc[index, "recording_seconds"])


def _iwr_normal_comparison(root: str | Path) -> pd.DataFrame:
    source = Path(root).resolve()
    gathered = source / "GatheredData" if (source / "GatheredData").is_dir() else source
    frames_out: list[pd.DataFrame] = []
    for path in sorted((gathered / "Not").glob("*.csv")):
        match = re.fullmatch(r"(Areeb|Raffay|Towsif)_(bow|squat|walk)_\d+\.csv", path.name)
        if match is None or match.group(1) == "Towsif":
            continue
        frames, _ = parse_iwr6843_fall_csv(path)
        series = _recording_series(frames)
        anchor = _normal_motion_anchor(series, "3_Sit_down_and_stand_up")
        selected = _align_series(
            series, anchor_seconds=anchor, start_relative=-1.5, end_relative=1.0
        )
        selected["recording_id"] = path.name
        selected["subject_id"] = match.group(1)
        selected["split"] = "external_development"
        selected["group"] = "iwr6843_normal_supplement"
        selected["action"] = f"iwr_{match.group(2)}"
        selected["anchor_definition"] = "radar_maximum_downward_centroid_velocity"
        selected["near_floor_relative_seconds"] = np.nan
        selected["descent_phase"] = np.nan
        frames_out.append(_add_baseline_deltas(selected))
    return pd.concat(frames_out, ignore_index=True) if frames_out else pd.DataFrame()


def _event_statistics(fall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for recording_id, recording in fall.groupby("recording_id"):
        pre = recording[
            (recording["relative_seconds"] >= -2.0)
            & (recording["relative_seconds"] < 0.0)
        ]
        early = recording[
            (recording["relative_seconds"] >= -2.0)
            & (recording["relative_seconds"] <= -1.2)
        ]
        target = recording[
            (recording["relative_seconds"] >= -1.0)
            & (recording["relative_seconds"] <= -0.5)
        ]
        for metric in METRICS:
            valid = pre[["relative_seconds", metric]].dropna()
            rho = float(spearmanr(valid["relative_seconds"], valid[metric]).statistic) if len(valid) >= 5 else np.nan
            slope = float(linregress(valid["relative_seconds"], valid[metric]).slope) if len(valid) >= 5 else np.nan
            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": str(recording["subject_id"].iloc[0]),
                    "split": str(recording["split"].iloc[0]),
                    "metric": metric,
                    "pre_onset_spearman_rho": rho,
                    "pre_onset_slope_per_second": slope,
                    "early_median": float(early[metric].median()),
                    "target_median": float(target[metric].median()),
                    "target_minus_early": float(target[metric].median() - early[metric].median()),
                }
            )
    return pd.DataFrame(rows)


def _group_statistics(fall: pd.DataFrame, normal: pd.DataFrame) -> pd.DataFrame:
    sitting = normal[normal["action"] == "3_Sit_down_and_stand_up"]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        fall_values = _recording_window_values(fall, metric, -1.0, -0.5)
        early_values = _recording_window_values(fall, metric, -2.0, -1.2)
        sit_values = _recording_window_values(sitting, metric, -1.0, -0.5)
        differences = fall_values.align(early_values, join="inner")[0] - fall_values.align(early_values, join="inner")[1]
        nonzero = differences[np.abs(differences) > 1e-12]
        try:
            paired_p = float(wilcoxon(differences).pvalue) if len(nonzero) else 1.0
        except ValueError:
            paired_p = 1.0
        rank_biserial = (
            float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
            if len(nonzero)
            else 0.0
        )
        try:
            normal_p = float(mannwhitneyu(fall_values, sit_values, alternative="two-sided").pvalue)
        except ValueError:
            normal_p = 1.0
        rows.append(
            {
                "metric": metric,
                "fall_event_count": int(len(fall_values)),
                "normal_sit_recording_count": int(len(sit_values)),
                "fall_target_median": float(fall_values.median()),
                "fall_early_median": float(early_values.median()),
                "normal_sit_pre_anchor_median": float(sit_values.median()),
                "paired_target_vs_early_p": paired_p,
                "paired_rank_biserial": rank_biserial,
                "target_vs_normal_sit_p": normal_p,
                "target_vs_normal_sit_cliffs_delta": _cliffs_delta(
                    fall_values.to_numpy(), sit_values.to_numpy()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["paired_target_vs_early_q"] = _benjamini_hochberg(
        result["paired_target_vs_early_p"].to_numpy()
    )
    result["target_vs_normal_sit_q"] = _benjamini_hochberg(
        result["target_vs_normal_sit_p"].to_numpy()
    )
    return result


def _recording_window_values(
    table: pd.DataFrame, metric: str, start: float, end: float
) -> pd.Series:
    selected = table[
        (table["relative_seconds"] >= start)
        & (table["relative_seconds"] <= end)
    ]
    return selected.groupby("recording_id")[metric].median().dropna()


def _time_bin_summary(table: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for start, end, label in TIME_BINS:
        selected = table[
            (table["relative_seconds"] >= start)
            & (table["relative_seconds"] < end)
        ]
        if not len(selected):
            continue
        for metric in METRICS:
            per_recording = (
                selected.groupby(["group", "action", "recording_id"])[metric]
                .median()
                .rename("value")
                .reset_index()
            )
            summary = (
                per_recording.groupby(["group", "action"])["value"]
                .agg(
                    median="median",
                    q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75),
                    recording_count="count",
                )
                .reset_index()
            )
            summary["metric"] = metric
            summary["time_bin"] = label
            summary["start_relative_seconds"] = start
            summary["end_relative_seconds"] = end
            frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _plot_aggregate_trends(
    fall: pd.DataFrame, normal: pd.DataFrame, destination: Path
) -> None:
    _configure_matplotlib()
    sitting = normal[normal["action"] == "3_Sit_down_and_stand_up"]
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 122 / 25.4), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        delta = f"{metric}_delta"
        for _, recording in fall.groupby("recording_id"):
            visible = recording[recording["relative_seconds"] <= 2.0]
            ax.plot(visible["relative_seconds"], visible[delta], color="#bc5a55", alpha=0.09, lw=0.55)
        _median_band(ax, fall[fall["relative_seconds"] <= 2.0], delta, "#9f3b36", "Fall events")
        _median_band(ax, sitting, delta, "#386cb0", "Sit/stand normal")
        ax.axvline(0.0, color="#333333", lw=0.8, ls="--")
        ax.axvspan(-1.0, -0.5, color="#e6b65c", alpha=0.12, lw=0)
        ax.set_title(METRIC_LABELS[metric], fontsize=7)
        ax.set_xlim(-2.0, 2.0)
        ax.grid(axis="y", color="#dddddd", lw=0.4)
    axes[1, 0].set_xlabel("Time relative to alignment anchor (s)")
    axes[1, 1].set_xlabel("Time relative to alignment anchor (s)")
    axes[1, 2].set_xlabel("Time relative to alignment anchor (s)")
    axes[0, 0].set_ylabel("Change from −2.0 to −1.5 s baseline")
    axes[1, 0].set_ylabel("Change from −2.0 to −1.5 s baseline")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        "Radar feature evolution: fall descent onset vs radar-aligned normal sitting",
        y=0.995,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, destination / "figure1_aggregate_feature_trends")
    plt.close(fig)


def _median_band(
    ax, table: pd.DataFrame, value: str, color: str, label: str
) -> None:
    rounded = table.assign(time_bin=np.round(table["relative_seconds"] * 10) / 10)
    per_recording = rounded.groupby(["recording_id", "time_bin"])[value].median().reset_index()
    summary = per_recording.groupby("time_bin")[value].agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    ax.plot(summary.index, summary["median"], color=color, lw=1.5, label=label)
    ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.18, lw=0)


def _plot_event_heatmaps(fall: pd.DataFrame, destination: Path) -> None:
    _configure_matplotlib()
    heat_metrics = (
        "height_proxy_m",
        "centroid_speed_mps",
        "spatial_dispersion_m",
        "doppler_abs_mean_mps",
        "point_count",
    )
    event_order = (
        fall.groupby("recording_id")["near_floor_relative_seconds"]
        .first()
        .sort_values()
        .index
    )
    time_grid = np.round(np.arange(-2.0, 2.01, 0.1), 1)
    fig, axes = plt.subplots(
        len(heat_metrics), 1, figsize=(183 / 25.4, 175 / 25.4), sharex=True
    )
    image = None
    for ax, metric in zip(axes, heat_metrics):
        matrix = []
        for event in event_order:
            recording = fall[fall["recording_id"] == event]
            values = np.interp(
                time_grid,
                recording["relative_seconds"],
                recording[f"{metric}_delta"],
                left=np.nan,
                right=np.nan,
            )
            baseline = recording[
                (recording["relative_seconds"] >= -2.0)
                & (recording["relative_seconds"] <= -1.2)
            ][metric]
            scale = float(baseline.std())
            if not np.isfinite(scale) or scale < 1e-6:
                scale = float(fall[metric].std()) or 1.0
            matrix.append(values / scale)
        image = ax.imshow(
            np.clip(np.asarray(matrix), -3, 3),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-3,
            vmax=3,
            extent=(time_grid[0], time_grid[-1], len(event_order), 0),
        )
        ax.axvline(0.0, color="black", lw=0.8, ls="--")
        ax.set_ylabel(METRIC_LABELS[metric].split(" (")[0], fontsize=6)
    axes[-1].set_xlabel("Seconds relative to skeleton-derived descent onset")
    fig.suptitle("Per-event feature evolution (32 development events)", fontsize=8)
    fig.subplots_adjust(left=0.18, right=0.82, top=0.94, bottom=0.07, hspace=0.18)
    if image is not None:
        colorbar_axis = fig.add_axes((0.86, 0.24, 0.018, 0.53))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Baseline-standardized change (clipped)", fontsize=6)
    _save_figure(fig, destination / "figure2_per_event_heatmaps")
    plt.close(fig)


def _plot_group_comparison(
    fall: pd.DataFrame, normal: pd.DataFrame, iwr: pd.DataFrame, destination: Path
) -> None:
    _configure_matplotlib()
    sitting = normal[normal["action"] == "3_Sit_down_and_stand_up"]
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 122 / 25.4))
    labels = ["Fall early", "Fall 0.5–1.0 s", "DGUHA sit"]
    colors = ["#aaaaaa", "#b64b45", "#4f79b5"]
    for ax, metric in zip(axes.flat, METRICS):
        groups = [
            _recording_window_values(fall, f"{metric}_delta", -2.0, -1.2).to_numpy(),
            _recording_window_values(fall, f"{metric}_delta", -1.0, -0.5).to_numpy(),
            _recording_window_values(sitting, f"{metric}_delta", -1.0, -0.5).to_numpy(),
        ]
        if len(iwr):
            for action in ("iwr_bow", "iwr_squat"):
                selected = iwr[iwr["action"] == action]
                groups.append(
                    _recording_window_values(
                        selected, f"{metric}_delta", -1.0, -0.5
                    ).to_numpy()
                )
            panel_labels = labels + ["IWR bow", "IWR squat"]
            panel_colors = colors + ["#7f6aa2", "#7a9f55"]
        else:
            panel_labels, panel_colors = labels, colors
        box = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.62)
        for patch, color in zip(box["boxes"], panel_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axhline(0.0, color="#777777", lw=0.6)
        ax.set_title(METRIC_LABELS[metric], fontsize=7)
        ax.set_xticks(range(1, len(panel_labels) + 1), panel_labels, rotation=28, ha="right", fontsize=5.5)
        ax.grid(axis="y", color="#dddddd", lw=0.4)
    fig.suptitle("Baseline-relative features before the alignment anchor", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, destination / "figure3_group_comparison")
    plt.close(fig)


def _plot_individual_events(fall: pd.DataFrame, destination: Path) -> None:
    _configure_matplotlib()
    with PdfPages(destination / "individual_fall_event_trends.pdf") as pdf:
        for recording_id, recording in fall.groupby("recording_id"):
            fig, axes = plt.subplots(2, 3, figsize=(8.27, 5.8), sharex=True)
            for ax, metric in zip(axes.flat, METRICS):
                ax.plot(recording["relative_seconds"], recording[metric], color="#9f3b36", lw=1.1)
                ax.axvline(0.0, color="#333333", lw=0.8, ls="--")
                floor_time = float(recording["near_floor_relative_seconds"].iloc[0])
                ax.axvline(floor_time, color="#777777", lw=0.7, ls=":")
                ax.axvspan(-1.0, -0.5, color="#e6b65c", alpha=0.14, lw=0)
                ax.set_title(METRIC_LABELS[metric], fontsize=7)
                ax.grid(axis="y", color="#dddddd", lw=0.4)
            axes[1, 0].set_xlabel("Seconds from descent onset")
            axes[1, 1].set_xlabel("Seconds from descent onset")
            axes[1, 2].set_xlabel("Seconds from descent onset")
            fig.suptitle(recording_id, fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _evolution_verdict(
    event_statistics: pd.DataFrame, group_statistics: pd.DataFrame
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    stable_count = 0
    specific_count = 0
    for metric in METRICS:
        event = event_statistics[event_statistics["metric"] == metric]
        group = group_statistics[group_statistics["metric"] == metric].iloc[0]
        rho_values = event["pre_onset_spearman_rho"].dropna().to_numpy()
        median_rho = float(np.median(rho_values)) if len(rho_values) else 0.0
        sign = np.sign(median_rho)
        sign_consistency = float(np.mean(np.sign(rho_values) == sign)) if sign else 0.0
        rho_ci = _bootstrap_median_ci(rho_values)
        stable = (
            abs(median_rho) >= 0.30
            and sign_consistency >= 0.65
            and float(group["paired_target_vs_early_q"]) < 0.05
            and abs(float(group["paired_rank_biserial"])) >= 0.33
        )
        specific = (
            stable
            and float(group["target_vs_normal_sit_q"]) < 0.05
            and abs(float(group["target_vs_normal_sit_cliffs_delta"])) >= 0.33
        )
        stable_count += int(stable)
        specific_count += int(specific)
        features.append(
            {
                "metric": metric,
                "median_pre_onset_spearman_rho": median_rho,
                "bootstrap_95_ci": rho_ci,
                "same_direction_event_fraction": sign_consistency,
                "paired_target_vs_early_q": float(group["paired_target_vs_early_q"]),
                "paired_rank_biserial": float(group["paired_rank_biserial"]),
                "target_vs_normal_sit_q": float(group["target_vs_normal_sit_q"]),
                "target_vs_normal_sit_cliffs_delta": float(group["target_vs_normal_sit_cliffs_delta"]),
                "stable_pre_onset_evolution": stable,
                "specific_against_normal_sitting": specific,
            }
        )
    if specific_count >= 2:
        conclusion = "stable_and_partly_specific_evolution_signal"
    elif stable_count >= 1:
        conclusion = "temporal_signal_present_but_not_specific_to_normal_descent"
    else:
        conclusion = "no_stable_cross_event_pre_onset_evolution_signal"
    return {
        "conclusion": conclusion,
        "stable_temporal_feature_count": stable_count,
        "normal_descent_specific_feature_count": specific_count,
        "exploratory_criteria": {
            "absolute_median_spearman_at_least": 0.30,
            "same_direction_event_fraction_at_least": 0.65,
            "paired_fdr_q_below": 0.05,
            "absolute_paired_rank_biserial_at_least": 0.33,
            "normal_specific_absolute_cliffs_delta_at_least": 0.33,
        },
        "features": features,
    }


def _bootstrap_median_ci(values: np.ndarray) -> list[float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [None, None]
    generator = np.random.default_rng(20260809)
    samples = generator.choice(values, size=(4000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return 0.0
    differences = left[:, None] - right[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(array),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _subject_id(file_name: str) -> str:
    match = re.fullmatch(r"([FM]_\d{3})_A\d+_\d+\.txt", file_name)
    if match is None or match.group(1) not in DGUHA_SPLIT_BY_SUBJECT:
        raise ValueError(f"unexpected DGUHA file name: {file_name}")
    return match.group(1)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DGUHA pre-fall motion evolution without model training.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--iwr6843-root", type=Path)
    args = parser.parse_args()
    result = analyze_motion_evolution(
        args.data_root,
        args.events,
        args.output_directory,
        iwr6843_root=args.iwr6843_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
