param(
    [string]$SubjectKey = "",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$repoRoot = Split-Path -Parent $PSScriptRoot
$packageRoot = Join-Path $repoRoot "backend\app\modules\psychology\home_detection_pkg"

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $name, $value = $line -split '=', 2
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ($name) { [Environment]::SetEnvironmentVariable($name, $value, "Process") }
    }
}

Import-DotEnv (Join-Path $repoRoot ".env")
if (-not $SubjectKey) {
    $SubjectKey = if ($env:PSYCH_SUBJECT_KEY) { $env:PSYCH_SUBJECT_KEY } else { "u-elder-001" }
}
$python = if ($env:PSYCH_PYTHON) { $env:PSYCH_PYTHON } else { "python" }
if (-not $LogPath) {
    $logDir = Join-Path $repoRoot "backend\runtime\demo-logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $LogPath = Join-Path $logDir ("psychology-demo-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
}

Write-Host "PSYCHOLOGY VISUAL-BEHAVIOR MONITOR" -ForegroundColor Cyan
Write-Host "subject: $SubjectKey"
Write-Host "full log: $LogPath"
Write-Host "basis: 3D facial landmarks + gaze + head pose + Action Units"
Write-Host "Ctrl+C to stop."
Write-Host ""

Push-Location $packageRoot
try {
    & $python -X utf8 -m service.psychology_worker_main `
        --subject-key $SubjectKey `
        --capture-mode opensdk `
        --loop 2>&1 |
        ForEach-Object {
            $line = [string]$_
            Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
            $stamp = Get-Date -Format "HH:mm:ss"
            if ($line -match 'MCCL 模型已预热') {
                Write-Host "[$stamp] MCCL READY" -ForegroundColor Green
            } elseif ($line -match 'capture\.begin') {
                Write-Host "[$stamp] capture.begin"
            } elseif ($line -match 'capture\.end') {
                Write-Host "[$stamp] capture.end"
            } elseif ($line -match 'openface\.begin') {
                Write-Host "[$stamp] openface.begin"
            } elseif ($line -match 'openface\.end') {
                Write-Host "[$stamp] openface.end"
            } elseif ($line -match '窗口\s+(\d+)/7\s+完成') {
                Write-Host "[$stamp] Clip $($Matches[1])/7 completed" -ForegroundColor Cyan
            } elseif ($line -match 'inference\.begin') {
                Write-Host "[$stamp] inference" -ForegroundColor Yellow
            } elseif ($line -match '评估完成:\s*PHQ-8\s*=\s*([0-9.]+)') {
                $score = [double]$Matches[1]
                $risk = if ($score -lt 10) { "no_risk" } elseif ($score -lt 15) { "mild" } elseif ($score -lt 20) { "moderate" } else { "severe" }
                Write-Host "[$stamp] PHQ-8 = $score" -ForegroundColor Magenta
                Write-Host "[$stamp] risk level = $risk" -ForegroundColor Magenta
                Write-Host "[$stamp] data quality = limited" -ForegroundColor Magenta
            } elseif ($line -match '失败|insufficient_data|分数越界') {
                Write-Host "[$stamp] $line" -ForegroundColor Red
            }
        }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
