package com.tzb.safeguard.data.fall.model

import kotlinx.serialization.Serializable

@Serializable
data class FallRiskOverview(
    val overall_risk_level: String = "unknown",
    val rooms: List<RoomFallRisk> = emptyList(),
    val generated_at: String = "",
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
