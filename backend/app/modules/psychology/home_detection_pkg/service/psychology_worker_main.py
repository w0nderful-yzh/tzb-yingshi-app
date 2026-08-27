# -*- coding: utf-8 -*-
"""流式心理评估 worker：攒满 7 个 clip（每个严格 1800 帧）判断一次。

编排链路（与雷达 worker 同模式，独立可选进程）：
    [OpenSDK 取流(本设备默认) / 萤石 live URL] → 捕获一个窗口 → OpenFace → CSV
    → 1800帧 clip → 内存累计 → 7 clips → infer_7clips → LatestAssessmentStore 原子写快照

说明与约束
----------
- 只做编排：不改 OpenFace 特征提取逻辑、openface_to_mccl.py、mccl_home_inference.py、
  MCCL/XGBoost/checkpoint、Backend API、Android。算法代码仅被 import/子进程复用。
- 模型输入语义不变：1 clip = 1800 帧；7 clips = 1 次评估。抓流时长只由帧率换算
  （window_seconds = 1800/fps + 余量），最终送入 MCCL 的每个 clip 仍是严格 1800 帧
  （尾部取帧、缺失零填充）。
- 取流链路：本设备 C6c 经 v2/live/address/get 只返回单片段占位 HLS（ErrCode 9053），
  标准 openlive 流（仅 H.264）不可用；OpenSDK（ezopen://，app/浏览器同款）能取到真实
  HEVC 画面（实测 ~13fps），经 ffmpeg 转临时 H.264 文件喂 OpenFace。--capture-mode auto
  在配好 --sdk-root 时优先 opensdk，否则回退 url。
- 视频不长期保存：临时视频/CSV 位于 --runtime-dir（gitignored），单窗口处理完即删；
  启动时清扫超时残留。
- 常驻不退出：取流失败有限重试；OpenFace 无输出 / 有效帧不足 → 该窗口失败；
  一轮凑不满 7 clips → 写 insufficient_data；模型/checkpoint/OpenFace exe 缺失 → 写
  failed 并在日志记录原因；--loop 继续下一轮。
- subject_key 必须与 Backend 该 elder 的 external_subject 一致，否则 LocalPsychologySource
  读不到快照。

启动（在 home_detection_pkg 目录下，需 PSYCH_YS7_SDK_ROOT 指向 EZVIZ OpenSDK 根目录，
Python 环境需含 numpy/pandas/torch/xgboost）：
    python -m service.psychology_worker_main \\
        --subject-key elder-001 --loop --capture-mode opensdk
普通项目启动仍是 docker compose up；本 worker 不跑时 Backend 照常读已有
latest snapshot / unavailable，不影响 Fraud/Fall/Backend。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
for _path in (str(PACKAGE_ROOT), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from service.result_store import LatestAssessmentStore  # noqa: E402
from service.schemas import PsychologyAssessmentSnapshot  # noqa: E402

WINDOW_FRAMES = 1800  # 1 clip 的严格帧数（模型输入约束，勿改）
CLIP_NAMES_ORDER = ("kps", "gaze", "pose", "AUs")  # 与 mccl_home_inference.CLIP_NAMES 一致

TOKEN_URL = "https://open.ys7.com/api/lapp/token/get"
LIVE_ADDRESS_URL = "https://open.ys7.com/api/lapp/v2/live/address/get"
PROTOCOL_CODES = {"hls": 2, "rtmp": 3}
OPENFACE_ARGS = ["-multi_view", "1", "-track", "1"]
DEFAULT_OPENFACE_TIMEOUT_SECONDS = 300.0
OPENFACE_STOP_TIMEOUT_SECONDS = 15.0


def log(message: str) -> None:
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


# ---------- 萤石 live URL（stdlib，无 httpx 依赖） ----------

def _post_form(url: str, data: dict[str, object]) -> dict:
    body = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _ys7_token(app_key: str, app_secret: str) -> str:
    payload = _post_form(TOKEN_URL, {"appKey": app_key, "appSecret": app_secret})
    if str(payload.get("code")) != "200":
        raise RuntimeError(
            f"YS7 token failed: code={payload.get('code')} msg={payload.get('msg')}"
        )
    token = payload.get("data", {}).get("accessToken")
    if not isinstance(token, str) or not token:
        raise RuntimeError("YS7 token response missing accessToken")
    return token


def _ys7_live_address(
    token: str, device_serial: str, channel_no: int, protocol: str, quality: int
) -> str:
    payload = _post_form(
        LIVE_ADDRESS_URL,
        {
            "accessToken": token,
            "deviceSerial": device_serial,
            "channelNo": channel_no,
            "protocol": PROTOCOL_CODES[protocol],
            "quality": quality,
        },
    )
    if str(payload.get("code")) != "200":
        raise RuntimeError(
            f"YS7 live address failed: code={payload.get('code')} msg={payload.get('msg')}"
        )
    url = payload.get("data", {}).get("url")
    if not isinstance(url, str) or not url:
        raise RuntimeError("YS7 live address response missing data.url")
    return url


def fetch_live_url(
    *,
    app_key: str,
    app_secret: str,
    device_serial: str,
    channel_no: int,
    protocol: str,
    quality: int,
) -> str:
    token = _ys7_token(app_key, app_secret)
    try:
        return _ys7_live_address(token, device_serial, channel_no, protocol, quality)
    except RuntimeError as exc:
        # 10002 = token 过期，刷新一次再试（镜像后端 Ys7ApiClient 行为）
        if "10002" not in str(exc):
            raise
        token = _ys7_token(app_key, app_secret)
        return _ys7_live_address(token, device_serial, channel_no, protocol, quality)


# ---------- FPS 探测（只决定抓流时长） ----------

def probe_stream_fps(url: str, ffprobe: str | None) -> float | None:
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "json", url,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            return None
        info = json.loads(completed.stdout)
        streams = info.get("streams") or []
        if not streams:
            return None
        rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
        num, _, den = str(rate).partition("/")
        num, den = float(num), float(den or 1)
        if num <= 0 or den <= 0:
            return None
        return num / den
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


def resolve_window_seconds(args: argparse.Namespace, url: str) -> float:
    if args.window_seconds is not None:
        return args.window_seconds
    fps = probe_stream_fps(url, args.ffprobe)
    if fps is None:
        fps = 30.0
    return WINDOW_FRAMES / fps + args.window_margin_seconds


# ---------- 窗口捕获（模式 url / tempfile） ----------

def _run_openface(openface_exe: Path, source: str, out_dir: Path) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "openface.log"
    log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [str(openface_exe), "-f", source, "-out_dir", str(out_dir), *OPENFACE_ARGS],
        cwd=str(openface_exe.parent),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    process._of_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def _stop_openface(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=OPENFACE_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=OPENFACE_STOP_TIMEOUT_SECONDS)


def _finish_openface(
    process: subprocess.Popen,
    *,
    terminate: bool,
    timeout_seconds: float = DEFAULT_OPENFACE_TIMEOUT_SECONDS,
) -> int:
    try:
        if terminate:
            _stop_openface(process)
        else:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _stop_openface(process)
                raise
        return process.returncode or 0
    finally:
        handle = getattr(process, "_of_log_handle", None)
        if handle is not None:
            handle.close()


def capture_window_url(
    openface_exe: Path, url: str, out_dir: Path, window_seconds: float
) -> Path | None:
    process = _run_openface(openface_exe, url, out_dir)
    deadline = time.monotonic() + window_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None and not list(out_dir.glob("*.csv")):
            break  # OpenFace 提前退出且无输出 -> 尽早判该窗口失败
        time.sleep(2.0)
    _finish_openface(process, terminate=True)
    csvs = sorted(out_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def capture_window_tempfile(
    openface_exe: Path, ffmpeg: str, url: str, out_dir: Path, window_seconds: float,
    openface_timeout_seconds: float = DEFAULT_OPENFACE_TIMEOUT_SECONDS,
) -> Path | None:
    chunk = out_dir / "chunk.ts"
    record = subprocess.run(
        [
            ffmpeg, "-y", "-i", url, "-t", str(window_seconds),
            "-c", "copy", "-f", "mpegts", str(chunk),
        ],
        capture_output=True,
        text=True,
        timeout=window_seconds + 30,
    )
    if record.returncode != 0 or not chunk.is_file() or chunk.stat().st_size == 0:
        return None
    process = _run_openface(openface_exe, str(chunk), out_dir)
    _finish_openface(process, terminate=False, timeout_seconds=openface_timeout_seconds)
    csvs = sorted(out_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def resolve_window_seconds_opensdk(args: argparse.Namespace) -> float:
    """OpenSDK 模式：无 URL 可探测，按 source_fps 换算抓流时长。"""
    if args.window_seconds is not None:
        return args.window_seconds
    return WINDOW_FRAMES / max(args.source_fps, 1.0) + args.window_margin_seconds


def capture_window_opensdk(
    openface_exe: Path, args: argparse.Namespace, out_dir: Path, window_seconds: float
) -> Path | None:
    """OpenSDK（ezopen:// 链路，app/浏览器同款）→ 临时 H.264 mp4 → OpenFace。

    本机 C6c 经 v2/live/address/get 只返回单片段占位 HLS（ErrCode 9053），
    标准 openlive 流不可用；OpenSDK 是这条设备唯一可用的取流链路。
    """
    from service.ezviz_opensdk_capture import EzvizOpenSdkRecorder

    window_label = f"{out_dir.parent.name}/{out_dir.name}"
    log(f"  [{window_label}] token.begin")
    token = _ys7_token(args.ys7_app_key, args.ys7_app_secret)
    log(f"  [{window_label}] token.end")
    recorder = EzvizOpenSdkRecorder(
        sdk_root=args.sdk_root,
        app_key=args.ys7_app_key,
        access_token=token,
        device_serial=args.device_serial,
        verify_code=args.ys7_verify_code or "",
        stream_type=args.stream_type,
        out_dir=out_dir,
    )
    log(f"  [{window_label}] capture.begin seconds={window_seconds:.1f}")
    video = recorder.record(seconds=window_seconds)
    log(f"  [{window_label}] capture.end file={video.name} bytes={video.stat().st_size}")
    log(f"  [{window_label}] openface.begin timeout_seconds={args.openface_timeout_seconds:.1f}")
    process = _run_openface(openface_exe, str(video), out_dir)
    _finish_openface(
        process, terminate=False, timeout_seconds=args.openface_timeout_seconds
    )
    log(f"  [{window_label}] openface.end returncode={process.returncode}")
    csvs = sorted(out_dir.glob("*.csv"))
    log(f"  [{window_label}] csv.count={len(csvs)}")
    # OpenFace 2.2 默认还会输出可视化 AVI、HOG 与对齐帧（几十~上百 MB/窗口），
    # 均非模型输入且关闭参数不生效，这里取到 CSV 后立即删除，窗口目录只留
    # mp4(OpenFace 必需输入) + csv(特征)，中断残留也仅此二者。
    for pattern in ("*.avi", "*.hog", "*_of_details.txt"):
        for extra in out_dir.glob(pattern):
            extra.unlink(missing_ok=True)
    for aligned in out_dir.glob("*_aligned"):
        shutil.rmtree(aligned, ignore_errors=True)
    return csvs[0] if csvs else None


# ---------- CSV → 1800 帧 clip ----------

def _tail_1800(matrix):
    """取尾部 1800 帧，不足零填充（模型输入语义：clip 恒为 1800 帧）。"""
    import numpy as np

    count = matrix.shape[0]
    clip = np.zeros((WINDOW_FRAMES,) + matrix.shape[1:], dtype=matrix.dtype)
    take = min(count, WINDOW_FRAMES)
    clip[:take] = matrix[-take:]
    return clip


def _max_faces_per_frame(csv_path: Path) -> int:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    if "face_id" not in frame.columns:
        return 1
    if frame.empty or "frame" not in frame.columns:
        return 1
    return int(frame.groupby("frame")["face_id"].nunique().max())


def build_clip_from_csv(
    csv_path: Path, *, min_valid_frames: int, work_dir: Path
) -> tuple | None:
    """CSV → (kps, gaze, pose, AUs) 各 (1800, ...) 的 clip；失败返回 None。"""
    from openface_to_mccl import build_feature_matrix, load_elderly_csv

    if _max_faces_per_frame(csv_path) > 1:
        elder_csv = work_dir / "elderly.csv"
        completed = subprocess.run(
            [
                sys.executable, str(SCRIPTS_DIR / "extract_elderly.py"),
                str(csv_path), str(elder_csv),
            ],
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not elder_csv.is_file():
            return None
        csv_path = elder_csv

    df, frame_min, frame_max = load_elderly_csv(csv_path)
    if df is None or df.empty or int(frame_max) < 1:
        return None
    num_total = int(frame_max)
    if num_total < min_valid_frames:
        return None
    kps, gaze, pose, aus = build_feature_matrix(df, frame_min, frame_max, num_total)
    return tuple(_tail_1800(matrix) for matrix in (kps, gaze, pose, aus))


# ---------- MCCL 推理（lazy import，避免无 torch 环境 import 本模块失败） ----------

_mccl_models = None
_mccl_requested_device = None


def _ensure_mccl_models(requested_device: str = "cpu"):
    """加载 MCCL 模型（缓存一次）。OpenSDK 模式必须在取流前调用：
    先载 torch，避免 OpenSDK 已载入的 DLL 与 torch 的 c10.dll 冲突。"""
    global _mccl_models, _mccl_requested_device
    if _mccl_models is not None:
        if requested_device != _mccl_requested_device:
            raise RuntimeError(
                "MCCL models are already loaded on "
                f"{_mccl_requested_device}; cannot switch to {requested_device} in-process"
            )
        return _mccl_models
    import mccl_home_inference as mi

    args = mi.build_args(requested_device)
    model1, model2, regressor, device = mi.load_models(args, mi.CKPT_DIR)
    _mccl_models = (args, model1, model2, regressor, device)
    _mccl_requested_device = requested_device
    log(
        "MCCL device resolved: "
        f"requested_device={requested_device} effective_device={device} "
        f"cuda_available={mi.torch.cuda.is_available()} XGB_device=cpu"
    )
    return _mccl_models


def run_mccl(accumulated: list[tuple], requested_device: str = "cpu") -> float:
    args, model1, model2, regressor, device = _ensure_mccl_models(requested_device)
    import mccl_home_inference as mi
    import torch

    clip_feats = [
        [
            torch.from_numpy(clip[index][mi.SAMPLE_INDEX]).float()
            for index in range(len(CLIP_NAMES_ORDER))
        ]
        for clip in accumulated
    ]
    return float(mi.infer_7clips(model1, model2, regressor, clip_feats, args, device))


# ---------- 快照 ----------

def build_snapshot(
    *,
    assessment_id: str,
    subject_key: str,
    status: str,
    window_started_at: _dt.datetime,
    window_ended_at: _dt.datetime,
    score: float | None = None,
    clip_count: int = 0,
    completed_at: _dt.datetime | None = None,
) -> PsychologyAssessmentSnapshot:
    return PsychologyAssessmentSnapshot(
        schema_version="psychology_assessment_v1",
        assessment_id=assessment_id,
        subject_key=subject_key,
        status=status,  # type: ignore[arg-type]
        window_started_at=window_started_at,
        window_ended_at=window_ended_at,
        estimated_phq8_score=score,
        segment_scores=[score] if score is not None else [],
        clip_count=clip_count,
        completed_at=completed_at,
    )


# ---------- 运行时清理 ----------

def sweep_runtime(runtime_dir: Path, max_age_seconds: float = 3600.0) -> None:
    if not runtime_dir.is_dir():
        return
    for child in runtime_dir.glob("psych-*"):
        try:
            if time.time() - child.stat().st_mtime > max_age_seconds:
                shutil.rmtree(child, ignore_errors=True)
                log(f"清理超时残留: {child}")
        except OSError:
            continue


# ---------- 主流程 ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="流式心理评估 worker：攒满 7 个 1800 帧 clip 判断一次"
    )
    parser.add_argument("--subject-key", required=True,
                        help="必须与 Backend 该 elder 的 external_subject 一致")
    parser.add_argument("--device-serial", default=os.environ.get("APP_YS7_DEVICE_SERIAL"))
    parser.add_argument("--ys7-app-key", default=os.environ.get("APP_YS7_APP_KEY"))
    parser.add_argument("--ys7-app-secret", default=os.environ.get("APP_YS7_APP_SECRET"))
    parser.add_argument("--channel-no", type=int, default=1)
    parser.add_argument("--live-protocol", choices=("rtmp", "hls"), default="hls")
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=None,
                        help="抓流时长；默认按探测 fps 换算 1800/fps + margin")
    parser.add_argument("--window-margin-seconds", type=float, default=8.0)
    parser.add_argument("--clips-per-assessment", type=int, default=7)
    parser.add_argument("--min-valid-frames", type=int, default=1500)
    parser.add_argument("--openface-exe",
                        default=os.environ.get("OPENFACE_EXE") or os.environ.get("PSYCH_OPENFACE_EXE"))
    parser.add_argument("--openface-timeout-seconds", type=float,
                        default=DEFAULT_OPENFACE_TIMEOUT_SECONDS,
                        help="单个离线视频允许 OpenFace 处理的最长时间；默认 300 秒")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--store-root", type=Path,
                        default=PACKAGE_ROOT / "home_out" / "latest")
    parser.add_argument("--runtime-dir", type=Path, default=PACKAGE_ROOT / ".runtime")
    parser.add_argument("--capture-mode", choices=("auto", "url", "tempfile", "opensdk"), default="auto",
                        help="auto = 配好 sdk-root 时优先 opensdk，否则 url；本设备标准 openlive 流不可用，"
                             "实际只有 opensdk 能取到真实画面")
    parser.add_argument("--sdk-root", default=os.environ.get("PSYCH_YS7_SDK_ROOT"),
                        help="EZVIZ OpenSDK 根目录（含 lib/win64/OpenNetStream.dll）")
    parser.add_argument("--stream-type", type=int, default=2,
                        help="OpenSDK 码流：1 主码流 / 2 子码流（C6c 实测两者均约 13fps）")
    parser.add_argument("--source-fps", type=float, default=15.0,
                        help="OpenSDK 模式抓流帧率（转码后恒定帧率），仅决定抓流时长")
    parser.add_argument("--ys7-verify-code", default=os.environ.get("APP_YS7_DEVICE_VERIFY_CODE", ""),
                        help="设备验证码；未加密设备可留空（SDK 端用占位值）")
    parser.add_argument("--stream-retry-attempts", type=int, default=3)
    parser.add_argument("--mccl-device", default=os.environ.get("PSYCH_MCCL_DEVICE", "cpu"),
                        help="MCCL device，默认 cpu；GPU 必须显式指定 cuda:0")
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> int:
    args.mccl_device = args.mccl_device.strip().lower()
    if args.clips_per_assessment != 7:
        raise SystemExit("--clips-per-assessment 必须是 7（MCCL 每样本恰好 7 个 clip）")
    if args.openface_timeout_seconds <= 0:
        raise SystemExit("--openface-timeout-seconds 必须大于 0")
    if not args.subject_key:
        raise SystemExit("--subject-key 必填")
    if not args.device_serial or not args.ys7_app_key or not args.ys7_app_secret:
        raise SystemExit("缺少 YS7 配置：--device-serial/--ys7-app-key/--ys7-app-secret")
    if not args.openface_exe or not Path(args.openface_exe).is_file():
        raise SystemExit(
            f"OpenFace exe 不存在: {args.openface_exe}（用 --openface-exe 或环境变量指定）"
        )
    openface_exe = Path(args.openface_exe)

    mode = args.capture_mode
    if mode == "auto":
        mode = "opensdk" if args.sdk_root else "url"
    if mode == "opensdk":
        if not args.sdk_root or not Path(args.sdk_root).joinpath("lib", "win64", "OpenNetStream.dll").is_file():
            raise SystemExit(
                f"OpenNetStream.dll 不存在: {args.sdk_root}（用 --sdk-root 或 PSYCH_YS7_SDK_ROOT 指定）"
            )

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sweep_runtime(runtime_dir)

    store = LatestAssessmentStore(Path(args.store_root))
    log(f"worker 启动: subject_key={args.subject_key} capture_mode={mode} "
        f"store={Path(args.store_root)} runtime={runtime_dir}")

    if mode == "opensdk":
        # 必须先于 OpenSDK 加载 torch，否则 OpenSDK 的 DLL 会让 torch c10.dll 加载失败
        try:
            _ensure_mccl_models(args.mccl_device)
            log("MCCL 模型已预热（先于 OpenSDK 加载，避免 DLL 冲突）")
        except Exception as exc:  # noqa: BLE001
            log(f"MCCL 模型预热失败，评估时会记入 failed 快照: {type(exc).__name__}: {exc}")

    while True:
        assessment_id = f"psy-{uuid.uuid4().hex}"
        cycle_started = _dt.datetime.now(_dt.timezone.utc)
        accumulated: list[tuple] = []
        window_failures = 0
        log(f"== 开始一轮评估 {assessment_id} ==")

        for clip_index in range(args.clips_per_assessment):
            url: str | None = None
            if mode != "opensdk":
                for attempt in range(args.stream_retry_attempts):
                    try:
                        url = fetch_live_url(
                            app_key=args.ys7_app_key,
                            app_secret=args.ys7_app_secret,
                            device_serial=args.device_serial,
                            channel_no=args.channel_no,
                            protocol=args.live_protocol,
                            quality=args.quality,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        log(f"  取 live URL 失败(尝试 {attempt + 1}/{args.stream_retry_attempts}): {exc}")
                        time.sleep(3)
                if url is None:
                    window_failures += 1
                    log(f"  窗口 {clip_index + 1} 失败：网络流不可用")
                    continue

            window_started = _dt.datetime.now(_dt.timezone.utc)
            if mode == "opensdk":
                window_seconds = resolve_window_seconds_opensdk(args)
            else:
                window_seconds = resolve_window_seconds(args, url)
            work_dir = runtime_dir / f"psych-{assessment_id}" / f"win-{clip_index}"
            try:
                if mode == "opensdk":
                    csv_path = capture_window_opensdk(
                        openface_exe, args, work_dir, window_seconds
                    )
                elif mode == "tempfile":
                    csv_path = capture_window_tempfile(
                        openface_exe, args.ffmpeg, url, work_dir, window_seconds,
                        args.openface_timeout_seconds,
                    )
                else:
                    csv_path = capture_window_url(openface_exe, url, work_dir, window_seconds)
                if csv_path is None:
                    raise RuntimeError("OpenFace 未产出 CSV")
                clip = build_clip_from_csv(
                    csv_path, min_valid_frames=args.min_valid_frames, work_dir=work_dir
                )
            except Exception as exc:  # noqa: BLE001
                log(f"  窗口 {clip_index + 1} 失败：{exc}")
                window_failures += 1
                continue
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

            if clip is None:
                log(f"  窗口 {clip_index + 1} 失败：有效帧不足（<{args.min_valid_frames}）")
                window_failures += 1
                continue

            accumulated.append(clip)
            window_ended = _dt.datetime.now(_dt.timezone.utc)
            store.write(
                build_snapshot(
                    assessment_id=assessment_id,
                    subject_key=args.subject_key,
                    status="processing",
                    window_started_at=cycle_started,
                    window_ended_at=window_ended,
                    clip_count=len(accumulated),
                )
            )
            log(f"  窗口 {clip_index + 1}/{args.clips_per_assessment} 完成，clip 数={len(accumulated)}"
                f"（失败窗口 {window_failures}）")

        if len(accumulated) < args.clips_per_assessment:
            log(f"== 本轮凑不满 {args.clips_per_assessment} 个 clip -> insufficient_data ==")
            store.write(
                build_snapshot(
                    assessment_id=assessment_id,
                    subject_key=args.subject_key,
                    status="insufficient_data",
                    window_started_at=cycle_started,
                    window_ended_at=_dt.datetime.now(_dt.timezone.utc),
                    clip_count=len(accumulated),
                )
            )
        else:
            try:
                score = run_mccl(accumulated, args.mccl_device)
            except Exception as exc:  # noqa: BLE001
                log(f"== 模型推理失败 -> failed: {type(exc).__name__}: {exc} ==")
                store.write(
                    build_snapshot(
                        assessment_id=assessment_id,
                        subject_key=args.subject_key,
                        status="failed",
                        window_started_at=cycle_started,
                        window_ended_at=_dt.datetime.now(_dt.timezone.utc),
                        clip_count=len(accumulated),
                    )
                )
            else:
                if not (0.0 <= score <= 24.0):
                    log(f"== 分数越界 {score:.2f} -> failed ==")
                    store.write(
                        build_snapshot(
                            assessment_id=assessment_id,
                            subject_key=args.subject_key,
                            status="failed",
                            window_started_at=cycle_started,
                            window_ended_at=_dt.datetime.now(_dt.timezone.utc),
                            clip_count=len(accumulated),
                        )
                    )
                else:
                    log(f"== 评估完成: PHQ-8 = {score:.2f} ==")
                    store.write(
                        build_snapshot(
                            assessment_id=assessment_id,
                            subject_key=args.subject_key,
                            status="completed",
                            window_started_at=cycle_started,
                            window_ended_at=_dt.datetime.now(_dt.timezone.utc),
                            score=score,
                            clip_count=len(accumulated),
                            completed_at=_dt.datetime.now(_dt.timezone.utc),
                        )
                    )

        if not args.loop:
            break
    return 0


def main() -> int:
    return run_worker(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
