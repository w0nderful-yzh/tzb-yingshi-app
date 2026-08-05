package com.tzb.safeguard.data.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * 数据模型层。
 * 字段命名与 docs/api/app-client-api.md 保持一致（snake_case 由后端契约决定），
 * 心理关怀相关模型本期按文档约定暂缓。
 */

/** 后端统一响应包装，见 docs/api/README.md「统一响应」 */
@Serializable
data class ApiResponse<T>(
    val code: Int,
    val message: String = "",
    val data: T? = null,
    val request_id: String = ""
)

// ---------- 用户 / 首页 ----------

@Serializable
data class UserInfo(
    val user_id: String = "",
    val role: String = "elder",           // elder | family
    val name: String = "",
    val bound_family_count: Int = 0,
    val font_size: String = "extra_large",
    val voice_assist_enabled: Boolean = true
)

@Serializable
data class TodayStats(
    val event_count: Int = 0,
    val active_hours: Double = 0.0,
    val call_screened: Int = 0
)

/** GET /safety/status 首页聚合 */
@Serializable
data class SafetyStatus(
    val overall: String = "safe",         // safe | attention | danger
    val overall_label: String = "一切正常",
    val active_event_count: Int = 0,
    val highest_active_level: String? = null,
    val devices_online: Int = 0,
    val devices_total: Int = 0,
    val checked_at: String = "",
    val today: TodayStats = TodayStats()
)

// ---------- 设备 ----------

@Serializable
data class Device(
    val device_id: String,
    val name: String,
    val room: String = "",
    val online: Boolean = false,
    val signal: String = "good",          // good | weak | offline
    val last_seen_at: String = ""
)

@Serializable
data class DeviceListData(val devices: List<Device> = emptyList())

@Serializable
data class LiveUrl(
    val url: String = "",
    val protocol: String = "flv",
    val expires_in: Int = 0
)

@Serializable
data class LiveSdkSession(
    val app_key: String,
    val access_token: String,
    val device_serial: String,
    val channel_no: Int = 1,
    val expires_in: Int = 300
)

// ---------- 风险事件 ----------

/** 告警级别：reminder 提醒 / warning 警告 / emergency 紧急 */
@Serializable
data class RiskEvent(
    val event_id: String,
    val type: String,        // fall_suspected | fraud_suspected | stranger | inactivity | sos | device_offline | night_leave_bed | sedentary
    val level: String,
    val title: String,
    val summary: String = "",
    val device_id: String = "",
    val occurred_at: String = "",
    val status: String = "open",   // open | acknowledged | resolved | false_alarm
    val evidence_image_url: String? = null
)

@Serializable
data class EventListData(
    val events: List<RiskEvent> = emptyList(),
    val next_cursor: String? = null
)

@Serializable
data class Reason(
    val key: String = "",
    val label: String = "",
    val value: String = ""
)

/** AI 可解释判断依据，复用防诈 evidence_chain 生成 */
@Serializable
data class Analysis(
    val confidence: Double = 0.0,
    val reasons: List<Reason> = emptyList(),
    val disclaimer: String = ""
)

@Serializable
data class NotificationRecord(
    val target: String = "",
    val channel: String = "",
    val sent_at: String = "",
    val ack: Boolean = false
)

@Serializable
data class Escalation(
    val auto_call_at: String = "",
    val status: String = "pending"   // pending | done | cancelled
)

@Serializable
data class EventDetail(
    val event_id: String,
    val type: String,
    val level: String,
    val status: String,
    val device_id: String = "",
    val occurred_at: String = "",
    val evidence_image_url: String? = null,
    val analysis: Analysis = Analysis(),
    val notifications: List<NotificationRecord> = emptyList(),
    val escalation: Escalation = Escalation()
)

// ---------- 请求体 ----------

@Serializable
data class SosRequest(
    val trigger: String = "long_press",
    val occurred_at: String
)

@Serializable
data class SosResult(
    val event_id: String = "",
    val status: String = "",
    val notified_contacts: Int = 0
)

/** 老人端告警确认：im_ok 我没事 / need_help 我需要帮助 */
@Serializable
data class ConfirmRequest(val action: String)

/** 家属端处置：acknowledged | resolved | false_alarm */
@Serializable
data class StatusPatch(val status: String, val note: String = "")

/** 用于无业务返回体的接口（confirm / patch 等） */
typealias EmptyData = JsonObject

// ---------- 家属端 ----------

@Serializable
data class Contact(
    val order: Int,
    val name: String,
    val relation: String = "",
    val phone: String = "",               // 已脱敏
    val channels: List<String> = emptyList()
)

@Serializable
data class ContactsData(val contacts: List<Contact> = emptyList())

@Serializable
data class ElderInfo(
    val elder_id: String,
    val name: String,
    val relation: String = "",
    val overall: String = "safe",
    val last_active_at: String = "",
    val pending_event_count: Int = 0
)

@Serializable
data class EldersData(val elders: List<ElderInfo> = emptyList())

// ---------- 数据看板统计 ----------

@Serializable
data class StatsBucket(
    val period: String,
    val reminder: Int = 0,
    val warning: Int = 0,
    val emergency: Int = 0
)

@Serializable
data class EventsStatsData(val buckets: List<StatsBucket> = emptyList())

/** 24 小时活动热力，0-1 归一化 */
@Serializable
data class ActivityData(val hours: List<Double> = emptyList())
