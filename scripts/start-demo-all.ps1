# 拍摄/演示日一键汇总：按依赖顺序拉起雷达 -> 认知 Worker，并检查 Docker 后端与引擎。
# 各步骤可独立重跑；单步失败会提示对应排查位置，已就绪的步骤自动跳过。
param(
    [string]$CliPort = "COM5",
    [string]$DataPort = "COM6",
    [switch]$SkipRadar,
    [switch]$SkipCognitive
)
$ErrorActionPreference = "Continue"

Write-Host "==== 演示环境一键检查/启动 ====" -ForegroundColor Cyan

# [1] Docker 后端
Write-Host "`n[1] 主后端 :8000" -ForegroundColor Cyan
try {
    $h = Invoke-RestMethod "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 5
    Write-Host "  OK: $($h.data.status)" -ForegroundColor Green
} catch {
    Write-Host "  不可达。请先执行: docker compose up -d postgres backend" -ForegroundColor Red
}

# [2] ASR 就绪
Write-Host "`n[2] 诈骗 ASR" -ForegroundColor Cyan
try {
    $m = Invoke-RestMethod "http://127.0.0.1:8000/api/v1/integrations/ys7/media/status" -TimeoutSec 5
    Write-Host ("  models_ready=" + $m.data.models_ready + ", source=" + $m.data.source) -ForegroundColor Green
} catch {
    Write-Host "  查询失败（后端未就绪？）" -ForegroundColor Yellow
}

# [3] 引擎 8001
Write-Host "`n[3] 多模态引擎 :8001" -ForegroundColor Cyan
try {
    Invoke-RestMethod "http://127.0.0.1:8001/api/health" -TimeoutSec 5 | Out-Null
    Write-Host "  OK: 已运行（若刚才是手动启动，请勿关闭其窗口）" -ForegroundColor Green
} catch {
    Write-Host "  未运行。请用 scripts/start-multimodal-engine.ps1 -PythonPath <PY3.12> -HostAddress 0.0.0.0 -Port 8001 启动" -ForegroundColor Yellow
}

# [4] 雷达全链
if ($SkipRadar) {
    Write-Host "`n[4] 雷达：已跳过（-SkipRadar）"
} else {
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:8010/health" -TimeoutSec 3
        if ($h.radar_connected) {
            Write-Host "`n[4] 雷达：已连接（REAL），跳过启动" -ForegroundColor Green
        } else {
            throw "connected=false"
        }
    } catch {
        Write-Host "`n[4] 雷达：未就绪，调用 start-radar.ps1 ..." -ForegroundColor Cyan
        & (Join-Path $PSScriptRoot "start-radar.ps1") -CliPort $CliPort -DataPort $DataPort
    }
}

# [5] 认知 MMSE Worker
if ($SkipCognitive) {
    Write-Host "`n[5] 认知 Worker：已跳过（-SkipCognitive）"
} else {
    $alive = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*cognitive.worker*" }
    if ($null -ne $alive) {
        Write-Host "`n[5] 认知 Worker：已在运行，跳过" -ForegroundColor Green
    } else {
        Write-Host "`n[5] 认知 Worker：未运行，调用 start-cognitive-worker.ps1 ..." -ForegroundColor Cyan
        & (Join-Path $PSScriptRoot "start-cognitive-worker.ps1")
    }
}

# [6] 心理 worker（沿用你自己的脚本；存在才调用）
Write-Host "`n[6] 心理 worker" -ForegroundColor Cyan
$psychScript = Join-Path $PSScriptRoot "start-psychology-demo-monitor.ps1"
if (Test-Path $psychScript) {
    & $psychScript
} else {
    Write-Host "  未找到 start-psychology-demo-monitor.ps1；请按部署文档 §7.8 手动启动（--subject-key u-elder-001 --loop）" -ForegroundColor Yellow
}

Write-Host "`n==== 完成。剩余手动动作：App 登录 -> 首页播放 -> 个人页点开始守护 ====" -ForegroundColor Cyan
