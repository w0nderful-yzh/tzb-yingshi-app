param(
    [string]$BaseUrl = "http://127.0.0.1:8000/api/v1",
    [string]$DeviceId = "",
    [string]$SessionId = "",
    [int]$PollMilliseconds = 500
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

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

function Get-EventStatus([string]$Token) {
    if (-not $Token) { return "AUTH_UNAVAILABLE" }
    try {
        $headers = @{ Authorization = "Bearer $Token" }
        $response = Invoke-RestMethod -Uri "$BaseUrl/events?limit=20" -Headers $headers
        $event = $response.data.events |
            Where-Object { $_.type -eq "fraud_suspected" } |
            Sort-Object occurred_at -Descending |
            Select-Object -First 1
        if ($null -eq $event) { return "NOT_CREATED" }
        return "$($event.status)  event=$($event.event_id)"
    } catch {
        return "QUERY_ERROR"
    }
}

Import-DotEnv (Join-Path $repoRoot ".env")
if (-not $DeviceId) { $DeviceId = $env:APP_YS7_DEVICE_SERIAL }
if (-not $DeviceId) { throw "DeviceId is required (or set APP_YS7_DEVICE_SERIAL)." }

$token = ""
try {
    $loginName = if ($env:APP_DEMO_GUARDIAN_LOGIN) { $env:APP_DEMO_GUARDIAN_LOGIN } else { "guardian" }
    $password = if ($env:APP_DEMO_GUARDIAN_PASSWORD) { $env:APP_DEMO_GUARDIAN_PASSWORD } else { "guardian123" }
    $loginBody = @{ login_name = $loginName; password = $password } | ConvertTo-Json
    $login = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/login" -ContentType "application/json" -Body $loginBody
    $token = $login.data.access_token
} catch {
    $token = ""
}

while ($true) {
    try {
        $media = Invoke-RestMethod -Uri "$BaseUrl/integrations/ys7/media/status"
        $activeSession = if ($SessionId) { $SessionId } else { $media.data.session_id }
        if (-not $activeSession) { throw "Fraud media session is not ready." }
        $escapedDevice = [Uri]::EscapeDataString($DeviceId)
        $risk = Invoke-RestMethod -Uri "$BaseUrl/fraud/sessions/$activeSession`?device_id=$escapedDevice"
        $snapshot = $risk.data
        $evidence = @($snapshot.evidence_chain)
        $latest = $evidence |
            Where-Object { $_.source -eq "speech" -and $_.text } |
            Sort-Object end_ms |
            Select-Object -Last 1
        $eventStatus = Get-EventStatus $token

        Clear-Host
        Write-Host "FRAUD EVIDENCE MONITOR" -ForegroundColor Cyan
        Write-Host "session: $activeSession"
        Write-Host "ASR: $($latest.transcript_status)" -ForegroundColor Yellow
        Write-Host "text: $($latest.text)"
        Write-Host ""
        Write-Host "Evidence" -ForegroundColor Cyan
        $evidence |
            Where-Object { $_.source -eq "speech" } |
            Sort-Object end_ms, kind -Unique |
            Select-Object -Last 8 |
            ForEach-Object {
                $used = if ($_.used_for_transition -eq $false) { "observed" } else { "active" }
                Write-Host ("  {0,-30} {1,-10} {2}" -f $_.kind, $_.strength, $used)
            }
        Write-Host ""
        Write-Host ("state: {0}  ({1})" -f $snapshot.state, $snapshot.state_label) -ForegroundColor Magenta
        Write-Host ("risk:  {0}  score={1}" -f $snapshot.risk_level, $snapshot.score) -ForegroundColor Red
        Write-Host "transition: $($snapshot.transition_reason)"
        Write-Host "RiskEvent: $eventStatus" -ForegroundColor Green
        Write-Host ""
        Write-Host "Ctrl+C to stop. Refresh: $PollMilliseconds ms" -ForegroundColor DarkGray
    } catch {
        Clear-Host
        Write-Host "FRAUD EVIDENCE MONITOR" -ForegroundColor Cyan
        Write-Host "WAITING: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Ctrl+C to stop."
    }
    Start-Sleep -Milliseconds $PollMilliseconds
}
