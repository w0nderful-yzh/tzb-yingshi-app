# Stop only the Radar workers started by start-dev.ps1 (by PID file).
# Does not touch other Python processes; Backend is stopped via docker compose.
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidDir = Join-Path $Root '.runtime'

$stopped = $false
Get-ChildItem -LiteralPath $pidDir -Filter 'worker_*.pid' -ErrorAction SilentlyContinue |
    ForEach-Object {
        $id = (Get-Content -LiteralPath $_.FullName).Trim()
        if ($id) {
            try {
                Stop-Process -Id $id -Force -ErrorAction Stop
                Write-Host "Stopped radar worker PID $id ($($_.BaseName))"
                $stopped = $true
            } catch {
                Write-Host "PID $id already gone ($($_.BaseName))"
            }
        }
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }

if (-not $stopped) { Write-Host "No radar workers were running (no PID files in .runtime/)." }
Write-Host "To stop the Backend: docker compose down"
