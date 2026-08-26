package com.tzb.safeguard.data.fall.model

import kotlinx.serialization.Serializable

@Serializable
data class FallRiskOverview(
    val overall_risk_level: String = "unknown",
    val rooms: List<RoomFallRisk> = emptyList(),
    val camera_monitoring: CameraMonitoringStatus = CameraMonitoringStatus(),
    val generated_at: String = "",
)

@Serializable
data class CameraMonitoringStatus(
    val camera_stream_status: String = "unavailable",
    val camera_algorithm_status: String = "unavailable",
    val detail: String = "摄像头跌倒预测服务未配置",
    val updated_at: String = "",
)

@Serializable
data class GuardianCapabilityStatus(
    val state: String = "unavailable",
    val enabled: Boolean = false,
    val detail: String = "暂不可用",
)

@Serializable
data class GuardianSessionStatus(
    val session_id: String? = null,
    val active: Boolean = false,
    val state: String = "stopped",
    val camera_analysis: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val fraud_monitoring: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val psychology_observation: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val radar_worker: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val radar_participation: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val fusion: GuardianCapabilityStatus = GuardianCapabilityStatus(),
    val camera_preview_managed_by_guard: Boolean = false,
    val reason_codes: List<String> = emptyList(),
    val started_at: String? = null,
    val updated_at: String = "",
)

@Serializable
data class RoomFallRisk(
    val room_id: String,
    val room_name: String,
    val decision_path: String,
    val risk_level: String,
    val risk_score: Double? = null,
    val prediction_state: String,
    val fall_event_status: String,
    val camera_status: String,
    val radar_status: String,
    val association_status: String,
    val joint_assessment: String,
    val evidence_summary: String,
    val updated_at: String? = null,
)
