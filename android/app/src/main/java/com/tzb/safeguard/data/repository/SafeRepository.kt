package com.tzb.safeguard.data.repository

import com.tzb.safeguard.data.model.*
import com.tzb.safeguard.data.network.ApiService
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** 业务异常：code 沿用后端错误码（10001/10002/10003），-1 表示本地网络失败 */
class ApiException(val code: Int, override val message: String) : Exception(message)

/**
 * 数据层仓库：UI 层唯一数据来源。
 * 统一把 ApiResponse 转成 Result<T>，屏蔽 code != 0、data 为空、网络异常三种失败。
 */
class SafeRepository(private val api: ApiService) {

    private suspend fun <T> call(block: suspend () -> ApiResponse<T>): Result<T> = try {
        val resp = block()
        when {
            resp.code == 0 && resp.data != null -> Result.success(resp.data)
            else -> Result.failure(ApiException(resp.code, resp.message.ifBlank { "服务返回异常，请稍后重试" }))
        }
    } catch (e: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (e: Exception) {
        Result.failure(ApiException(-1, "请求异常：${e.message ?: "未知错误"}"))
    }

    // ---------- 通用 ----------

    suspend fun getMe(): Result<UserInfo> = call { api.getMe() }

    suspend fun getSafetyStatus(): Result<SafetyStatus> = call { api.getSafetyStatus() }

    suspend fun sendSos(): Result<SosResult> {
        // ISO 8601 带时区时间戳，与后端契约一致
        val now = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).apply {
            timeZone = TimeZone.getDefault()
        }.format(Date())
        return call { api.postSos(SosRequest(occurred_at = now)) }
    }

    // ---------- 事件 ----------

    suspend fun getEvents(
        elderId: String? = null,
        level: String? = null,
        status: String? = null
    ): Result<EventListData> = call { api.getEvents(elderId, level, status) }

    suspend fun getEventDetail(eventId: String): Result<EventDetail> =
        call { api.getEventDetail(eventId) }

    /** 老人端：我没事 / 我需要帮助 */
    suspend fun confirmEvent(eventId: String, action: String): Result<Unit> =
        call { api.confirmEvent(eventId, ConfirmRequest(action)) }.map { }

    /** 家属端：acknowledged / resolved / false_alarm */
    suspend fun patchEventStatus(eventId: String, status: String, note: String = ""): Result<Unit> =
        call { api.patchEventStatus(eventId, StatusPatch(status, note)) }.map { }

    suspend fun callElder(eventId: String): Result<Unit> =
        call { api.callElder(eventId) }.map { }

    // ---------- 设备 / 家属 ----------

    suspend fun getDevices(elderId: String? = null): Result<DeviceListData> =
        call { api.getDevices(elderId) }

    suspend fun getLiveUrl(deviceId: String): Result<LiveUrl> = call { api.getLiveUrl(deviceId) }

    suspend fun getLiveSdkSession(deviceId: String): Result<LiveSdkSession> =
        call { api.getLiveSdkSession(deviceId) }

    suspend fun getContacts(elderId: String? = null): Result<ContactsData> =
        call { api.getContacts(elderId) }

    suspend fun getElders(): Result<EldersData> = call { api.getElders() }

    suspend fun getEventsStats(elderId: String? = null, days: Int = 30): Result<EventsStatsData> =
        call { api.getEventsStats(elderId, days) }

    suspend fun getActivityStats(elderId: String? = null, days: Int = 7): Result<ActivityData> =
        call { api.getActivityStats(elderId, days) }
}
