# Unified dev launcher:
#   docker compose (App Backend :8000) + optional Radar workers (Windows host)
#   + adb reverse. Backend is always started; Radar is optional per room.
#
# Configuration is read from the repo-root .env (falls back to .env.example):
#   RADAR_PYTHON, RADAR_BATHROOM_MODE / RADAR_BEDROOM_MODE,
#   RADAR_BATHROOM_REPLAY_FILE / RADAR_BEDROOM_REPLAY_FILE,
#   RADAR_WORKER_CHECKPOINT, RADAR_WORKER_CALIBRATION,
#   RADAR_RUNTIME_STATE_DIR, RADAR_WORKER_CHECKPOINT_SHA256
param([switch]$SkipBackend)
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Read-EnvFile {
    param([string]$Path)
    $map = @{}
    if (Test-Path -LiteralPath $Path) {
        Get-Content -LiteralPath $Path -Encoding UTF8 | Where-Object {
            $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*='
        } | ForEach-Object {
            $kv = $_ -split '=', 2
            if ($kv.Count -eq 2) { $map[$kv[0].Trim()] = $kv[1].Trim().Trim('"') }
        }
    }
    return $map
}

$cfg = Read-EnvFile (Join-Path $Root '.env')
if ($cfg.Count -eq 0) { $cfg = Read-EnvFile (Join-Path $Root '.env.example') }

function Resolve-RepoPath {
    param([string]$p)
    if (-not $p) { return $p }
    if ($p -match '^[A-Za-z]:[\\/]') { return $p }  # absolute Windows path
    return Join-Path $Root $p
}

# ---- 1. Backend (always) ----
if (-not $SkipBackend) {
    Write-Host "== Starting App Backend via docker compose =="
    docker compose up -d --no-deps backend postgres 2>&1 | Out-Host
}
Write-Host "Backend: http://localhost:8000"

# ---- 2. Radar workers (optional, per room) ----
$python  = if ($cfg['RADAR_PYTHON']) { $cfg['RADAR_PYTHON'] } else { 'python' }
$ckpt    = Resolve-RepoPath $cfg['RADAR_WORKER_CHECKPOINT']
$cal     = Resolve-RepoPath $cfg['RADAR_WORKER_CALIBRATION']
$state   = Resolve-RepoPath $cfg['RADAR_RUNTIME_STATE_DIR']
$worker  = Join-Path $Root 'backend\app\modules\fall\radar_module\service\radar_worker_main.py'
$workdir = Join-Path $Root 'backend\app\modules\fall'
$pidDir  = Join-Path $Root '.runtime'
New-Item -ItemType Directory -Force -Path $pidDir | Out-Null

foreach ($room in @('bathroom', 'bedroom')) {
    $modeKey = "RADAR_${room}_MODE"
    $mode = if ($cfg.ContainsKey($modeKey)) { $cfg[$modeKey] } else { 'disabled' }
    if ($mode -eq 'disabled') { Write-Host "[$room] mode=disabled, skipped"; continue }

    $replayKey = "RADAR_${room}_REPLAY_FILE"
    $replayRel = if ($cfg.ContainsKey($replayKey)) { $cfg[$replayKey] } else { '' }
    if (-not $replayRel) {
        Write-Host "[$room] mode=$mode but no REPLAY_FILE configured; skipped"
        continue
    }
    if ($mode -eq 'real') {
        Write-Host "[$room] mode=real requires worker TI-bridge support (not yet implemented); skipped"
        continue
    }
    $replayPath = Resolve-RepoPath $replayRel
    if (-not (Test-Path -LiteralPath $replayPath)) {
        Write-Host "[$room] replay file not found: $replayPath"
        continue
    }
    if (-not (Test-Path -LiteralPath $ckpt) -or -not (Test-Path -LiteralPath $cal)) {
        Write-Host "[$room] checkpoint/calibration not found; skipped"
        continue
    }

    # Avoid double-start: if a recorded PID is still alive, skip.
    $pidFile = Join-Path $pidDir "worker_${room}.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $oldPid = (Get-Content -LiteralPath $pidFile).Trim()
        if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Write-Host "[$room] already running (PID $oldPid); skipped"
            continue
        }
    }

    $workerArgs = @(
        '-m', 'radar_module.service.radar_worker_main',
        '--room', $room,
        '--replay-file', $replayPath,
        '--checkpoint', $ckpt,
        '--calibration', $cal,
        '--runtime-state-dir', $state,
        '--loop'
    )
    if ($cfg['RADAR_WORKER_CHECKPOINT_SHA256']) {
        $workerArgs += @('--checkpoint-sha256', $cfg['RADAR_WORKER_CHECKPOINT_SHA256'])
    }

    try {
        # NOTE: avoid -RedirectStandardOutput/Error on PS 5.1; it makes
        # Start-Process wait for the long-running worker and hangs the script.
        $proc = Start-Process -FilePath $python -ArgumentList $workerArgs `
            -WorkingDirectory $workdir -PassThru -WindowStyle Hidden
        Set-Content -LiteralPath $pidFile -Value $proc.Id
        Write-Host "[$room] worker started (PID $($proc.Id)) mode=$mode"
    } catch {
        # Radar failure must not break the Backend.
        Write-Host "[$room] worker failed to start (Backend unaffected): $($_.Exception.Message)"
    }
}

# ---- 3. adb reverse (if adb available) ----
$adbCandidates = @(
    'D:/android/sdk/platform-tools/adb.exe',
    "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
    'adb'
)
$adb = $adbCandidates | Where-Object {
    $_ -eq 'adb' -or (Test-Path -LiteralPath $_)
} | Select-Object -First 1
if ($adb) {
    try {
        & $adb reverse tcp:8000 tcp:8000 2>&1 | Out-Null
        Write-Host "adb reverse tcp:8000: OK"
    } catch {
        Write-Host "adb reverse failed: $($_.Exception.Message)"
    }
} else {
    Write-Host "adb not found; if using an emulator run: adb reverse tcp:8000 tcp:8000"
}

Write-Host "== Dev stack started. Stop workers with: .\stop-dev.ps1 =="
