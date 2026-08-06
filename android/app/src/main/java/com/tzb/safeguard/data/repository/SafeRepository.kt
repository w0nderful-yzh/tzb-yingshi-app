package com.tzb.safeguard.data.repository

import com.tzb.safeguard.data.model.*
import com.tzb.safeguard.data.network.ApiService
import java.io.IOException

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

    // ---------- 事件 ----------

    suspend fun getEvents(
        elderId: String? = null,
        level: String? = null,
        status: String? = null
    ): Result<EventListData> = call { api.getEvents(elderId, level, status) }

    suspend fun getEventDetail(eventId: String): Result<EventDetail> =
        call { api.getEventDetail(eventId) }

    /** 家属端：acknowledged / resolved / false_alarm */
    suspend fun patchEventStatus(eventId: String, status: String, note: String = ""): Result<Unit> =
        call { api.patchEventStatus(eventId, StatusPatch(status, note)) }.map { }

    suspend fun sendInterventionReminder(eventId: String): Result<Unit> =
        call { api.sendInterventionReminder(eventId, InterventionReminder()) }.map { }

    // ---------- 设备 / 家属 ----------

    suspend fun getDevices(elderId: String? = null): Result<DeviceListData> =
        call { api.getDevices(elderId) }

    suspend fun getLiveUrl(deviceId: String): Result<LiveUrl> = call { api.getLiveUrl(deviceId) }

    suspend fun getLiveSdkSession(deviceId: String): Result<LiveSdkSession> =
        call { api.getLiveSdkSession(deviceId) }

    suspend fun getHistoryPlayback(
        deviceId: String,
        elderId: String? = null,
        at: String? = null
    ): Result<HistoryPlayback> = call { api.getHistoryPlayback(deviceId, elderId, at) }

    suspend fun getContacts(elderId: String? = null): Result<ContactsData> =
        call { api.getContacts(elderId) }

    suspend fun getElders(): Result<EldersData> = call { api.getElders() }

}
