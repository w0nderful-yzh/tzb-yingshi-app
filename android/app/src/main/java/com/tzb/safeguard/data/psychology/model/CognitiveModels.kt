package com.tzb.safeguard.data.psychology.model

import kotlinx.serialization.Serializable

@Serializable
data class CognitiveAssessmentWindow(
    val started_at: String = "",
    val ended_at: String? = null,
)

@Serializable
data class CognitiveCompletedReference(
    val assessment_window: CognitiveAssessmentWindow? = null,
    val estimated_mmse_score: Double? = null,
    val attention_level: String? = null,
    val data_quality: String = "limited",
    val source_modality: String = "voice_acoustic",
    val evidence_summary: String = "上一轮语音声学特征辅助分析已完成",
    val updated_at: String? = null,
    val disclaimer: String = "AI辅助认知状态评估仅供日常关怀参考，不构成认知障碍或医疗诊断。",
)

@Serializable
data class CognitiveOverview(
    val source_status: String = "unavailable",
    val assessment_state: String = "unavailable",
    val data_quality: String = "insufficient",
    val source_modality: String = "voice_acoustic",
    val assessment_window: CognitiveAssessmentWindow? = null,
    val estimated_mmse_score: Double? = null,
    val attention_level: String? = null,
    val evidence_summary: String = "认知状态辅助评估服务暂不可用",
    val guidance: String = "建议结合日常沟通、生活表现和专业人员意见进行综合关注",
    val updated_at: String? = null,
    val disclaimer: String = "AI辅助认知状态评估仅供日常关怀参考，不构成认知障碍或医疗诊断。",
    val latest_completed: CognitiveCompletedReference? = null,
)
