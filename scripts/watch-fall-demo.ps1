param(
    [string]$EngineUrl = "http://127.0.0.1:8001",
    [int]$PollMilliseconds = 500
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

function Format-Value($Value, [string]$Format = "0.000") {
    if ($null -eq $Value) { return "--" }
    if ($Value -is [ValueType]) {
        try { return ([double]$Value).ToString($Format) } catch { return [string]$Value }
    }
    return [string]$Value
}

while ($true) {
    try {
        $live = Invoke-RestMethod -Uri "$EngineUrl/api/fall-live/status" -TimeoutSec 2
        $multimodal = Invoke-RestMethod -Uri "$EngineUrl/api/multimodal/latest" -TimeoutSec 2
        $augmentation = $multimodal.associated_risk_augmentation
        $radarFeatures = $augmentation.radar_motion_features
        $fusionV2 = $multimodal.camera_led_evidence_fusion_v2

        Clear-Host
        Write-Host "FALL RISK FACTOR MONITOR" -ForegroundColor Cyan
        Write-Host ("source: {0}  input: {1}  fps: {2}" -f `
            $multimodal.data_source, $live.input_state, (Format-Value $live.effective_sample_fps "0.0"))
        Write-Host ""
        Write-Host "Camera / BioSTGCN" -ForegroundColor Cyan
        Write-Host ("  risk score             {0}" -f (Format-Value $live.risk_score)) -ForegroundColor Red
        Write-Host ("  risk level             {0}" -f $live.risk_level)
        Write-Host ("  positive votes         {0}" -f (Format-Value $live.positive_votes "0"))
        Write-Host ("  torso inclination      {0} deg" -f (Format-Value $live.torso_inclination_deg "0.0"))
        Write-Host ("  COM proxy rel. change  {0}" -f (Format-Value $live.com_proxy_relative_change))
        Write-Host ("  torso yaw change       {0} deg" -f (Format-Value $live.yaw_delta_deg "0.0"))
        Write-Host ("  pose quality           {0}" -f (Format-Value $live.pose_quality))
        Write-Host ""
        Write-Host "Associated Radar motion evidence" -ForegroundColor Cyan
        Write-Host ("  height delta           {0} m" -f (Format-Value $radarFeatures.height_delta_m))
        Write-Host ("  vertical velocity      {0} m/s" -f (Format-Value $radarFeatures.vertical_velocity_mps))
        Write-Host ("  point count            {0}" -f (Format-Value $radarFeatures.point_count "0"))
        Write-Host ("  track stability        {0}" -f (Format-Value $augmentation.track_stability))
        Write-Host ("  evidence strength      {0}" -f $augmentation.radar_motion_evidence_strength)
        Write-Host ("  association            {0}" -f $augmentation.association_state)
        Write-Host ("  sync delta             {0} ms" -f (Format-Value $augmentation.sync_delta_ms "0.0"))
        Write-Host ""
        Write-Host ("Fusion v2: {0}  radar eligible={1}" -f $fusionV2.fusion_mode, $fusionV2.radar_eligible) -ForegroundColor Magenta
        Write-Host ("updated: {0}" -f $multimodal.timestamp)
        Write-Host "Ctrl+C to stop. Refresh: $PollMilliseconds ms" -ForegroundColor DarkGray
    } catch {
        Clear-Host
        Write-Host "FALL RISK FACTOR MONITOR" -ForegroundColor Cyan
        Write-Host "WAITING: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Ctrl+C to stop."
    }
    Start-Sleep -Milliseconds $PollMilliseconds
}
