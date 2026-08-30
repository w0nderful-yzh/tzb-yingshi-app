# 一键启动雷达链路：8010 Radar FastAPI + 真机 UART bridge + 出分验证。
# 本机实测默认值：CLI=COM5(Enhanced)、Data=COM6(Standard)，与常规接法相反；不同机器用参数覆盖。
# 依赖：Python 3.10 环境（torch/transformers/pyserial）、TI Radar Toolbox 2.20.00.05、板卡上电。
param(
    [string]$PythonPath = "E:\python3.10.9aaa\python.exe",
    [string]$CliPort = "COM5",
    [string]$DataPort = "COM6",
    [string]$Room = "living_room",
    [string]$DeviceId = "radar-living-room-01",
    [int]$Port = 8010,
    [string]$ToolboxRoot = "E:\创新实践\老人摔倒预警\雷达模块\radar_toolbox_2_20_00_05",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)
$ErrorActionPreference = "Stop"

$FallDir     = Join-Path $RepoRoot "backend\app\modules\fall"
$Checkpoint  = Join-Path $RepoRoot "backend\app\modules\fall\radar_module\checkpoints\experiments_v5\tcn_hard_negative\tcn_0p5_1p0_specificity_operating_point_v1.pt"
$Calibration = Join-Path $RepoRoot "backend\app\modules\fall\radar_module\reports\domain_calibration_v1_full\calibrated_normalization_real_gaussian.json"
# TI Toolbox 与 cfg 在仓库外（不随 Git 分发）；路径不同用 -ToolboxRoot 覆盖。
$Cfg       = Join-Path $ToolboxRoot "source\ti\examples\People_Tracking\3D_People_Tracking\chirp_configs\ISK_6m_default.cfg"
$CommonDir = Join-Path $ToolboxRoot "tools\visualizers\Applications_Visualizer\common"

function Assert-File([string]$Path, [string]$Hint) {
    if (-not (Test-Path $Path)) {
        Write-Host "[失败] 缺少 $Path" -ForegroundColor Red
        Write-Host "       $Hint"
        exit 1
    }
}
Assert-File $PythonPath  "请用 -PythonPath 指定 Python 3.10 解释器。"
Assert-File $Checkpoint  "雷达 TCN checkpoint 缺失，检查仓库完整性。"
Assert-File $Calibration "雷达校准文件缺失，检查仓库完整性。"
Assert-File $Cfg         "TI cfg 缺失，用 -ToolboxRoot 指定 Radar Toolbox 根目录。"
Assert-File $CommonDir   "TI gui_parser 目录缺失，用 -ToolboxRoot 指定 Radar Toolbox 根目录。"

# 环境变量：注意两个 shadow 开关必须同时为 true，否则引擎校验 health/latest 的 model_mode 会 mismatch。
$env:RADAR_ROOM = $Room
$env:RADAR_DEVICE_ID = $DeviceId
$env:RADAR_TORCH_DEVICE = "cpu"
$env:RADAR_TCN_SHADOW_ENABLED = "true"
$env:RADAR_CALIBRATED_TCN_SHADOW_ENABLED = "true"
$env:RADAR_TCN_CHECKPOINT_PATH = $Checkpoint
$env:RADAR_CALIBRATED_TCN_CALIBRATION_PATH = $Calibration
$env:TI_OFFICIAL_OUTPUT_CWD = $FallDir
# bridge 命令必须显式传 --config 与 --official-common-dir（代码默认路径指向仓库内，不存在）。
$env:TI_OFFICIAL_OUTPUT_COMMAND_JSON = ConvertTo-Json @(
    $PythonPath,
    "-m", "radar_module.acquisition.ti_official_bridge",
    "--cli-port", $CliPort,
    "--data-port", $DataPort,
    "--config", $Cfg,
    "--official-common-dir", $CommonDir
) -Compress

Write-Host "[1/4] 在新窗口启动 Radar FastAPI :$Port ..." -ForegroundColor Cyan
$cmd = "& '$PythonPath' -m uvicorn radar_module.service.radar_api:app --host 127.0.0.1 --port $Port"
Start-Process powershell -WorkingDirectory $FallDir -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd

# 等待模型加载
$ready = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($h.model_loaded) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host "[失败] 30 s 内 /health 未就绪，请检查新窗口的报错。" -ForegroundColor Red
    exit 1
}
Write-Host "[2/4] 模型已加载（model_loaded=true）。" -ForegroundColor Green

# 连接真机：板卡两次会话间常需喘息，失败重试 3 次
Write-Host "[3/4] 连接板卡（CLI=$CliPort, Data=$DataPort）..." -ForegroundColor Cyan
$connected = $false
for ($try = 1; $try -le 3; $try++) {
    try {
        $r = Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/api/radar/real" -TimeoutSec 25
        Write-Host ("  响应: " + ($r | ConvertTo-Json -Compress))
        $connected = $true
        break
    } catch {
        Write-Host "  第 $try 次失败：$($_.Exception.Message)" -ForegroundColor Yellow
        if ($try -lt 3) { Write-Host "  5 s 后重试（板卡需要喘息；持续失败请给板卡断电重上电）..." }
        Start-Sleep -Seconds 5
    }
}
if (-not $connected) {
    Write-Host "[失败] 无法进入 REAL 模式，见新窗口/雷达窗口日志。" -ForegroundColor Red
    exit 1
}

# 等待第一个有效窗口（warmup 约 2 s，留足余量）
Write-Host "[4/4] 等待首个有效预测窗口..." -ForegroundColor Cyan
$valid = $false
for ($i = 0; $i -lt 10; $i++) {
    Start-Sleep -Seconds 3
    try {
        $l = Invoke-RestMethod "http://127.0.0.1:$Port/api/radar/latest" -TimeoutSec 3
        $c = $l.calibrated_tcn_prediction
        if ($null -ne $c -and $c.score_valid) {
            $valid = $true
            Write-Host ("  score_valid=true, state=" + $c.tcn_risk_state + ", quality=" + $c.data_quality) -ForegroundColor Green
            break
        }
        $reason = if ($null -ne $c) { $c.unknown_reason } else { "503" }
        Write-Host ("  等待中: $reason")
    } catch {
        Write-Host "  latest 暂不可用（bridge 重连或预热中）..."
    }
}
Write-Host ""
if ($valid) {
    Write-Host "[完成] 雷达链路就绪（8010 运行中，请勿关闭新窗口）。App 点开始守护会自动参与融合。" -ForegroundColor Green
} else {
    Write-Host "[警告] REAL 已连接但暂无有效窗口；若持续 INSUFFICIENT_DATA，检查人员是否在 8 m 内与 cfg 帧率。" -ForegroundColor Yellow
}
