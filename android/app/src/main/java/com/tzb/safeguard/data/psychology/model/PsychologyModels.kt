package com.tzb.safeguard.data.psychology.model

import kotlinx.serialization.Serializable

@Serializable
data class PsychologyAssessmentWindow(
    val started_at: String = "",
    val ended_at: String = "",
)

@Serializable
data class PsychologyCompletedReference(
    val assessment_window: PsychologyAssessmentWindow? = null,
    val data_quality: String = "limited",
    val review_status: String = "required",
    val estimated_phq8_score: Double? = null,
    val evidence_summary: String = "上一轮参考分析已完成",
    val guidance: String = "结果仅供日常关怀参考",
    val updated_at: String? = null,
    val disclaimer: String = "该结果不构成心理或医疗诊断",
)

@Serializable
data class PsychologyOverview(
    val source_status: String = "unavailable",
    val operating_mode: String = "shadow",
    val assessment_state: String = "unavailable",
    val attention_level: String = "unknown",
    val trend_state: String = "insufficient_history",
    val data_quality: String = "insufficient",
    val source_modality: String = "camera_behavior",
    val review_status: String = "not_available",
    val assessment_window: PsychologyAssessmentWindow? = null,
    // 研究原型回归输出的原始参考值：非诊断、非分级
    val estimated_phq8_score: Double? = null,
    val segment_scores: List<Double> = emptyList(),
    val evidence_summary: String = "心理健康评估服务暂不可用",
    val guidance: String = "结果仅供日常关怀参考",
    val updated_at: String? = null,
    val disclaimer: String = "该结果不构成心理或医疗诊断",
    val latest_completed: PsychologyCompletedReference? = null,
)

