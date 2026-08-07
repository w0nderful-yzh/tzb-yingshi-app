package com.tzb.safeguard.data.repository

import com.tzb.safeguard.data.model.*
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.auth.AuthStore
import com.tzb.safeguard.data.network.ApiService
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.HttpException

/** 业务异常：code 沿用后端错误码（10001/10002/10003），-1 表示本地网络失败 */
class ApiException(val code: Int, override val message: String) : Exception(message)

/**
 * 数据层仓库：UI 层唯一数据来源。
 * 统一把 ApiResponse 转成 Result<T>，屏蔽 code != 0、data 为空、网络异常三种失败。
 */
class SafeRepository(
    private val api: ApiService,
    private val authStore: AuthStore,
) {

    private suspend fun <T> call(
        unauthorizedMessage: String = "登录已失效，请重新登录",
        block: suspend () -> ApiResponse<T>,
    ): Result<T> = try {
        val resp = block()
        when {
            resp.code == 0 && resp.data != null -> Result.success(resp.data)
            else -> Result.failure(ApiException(resp.code, resp.message.ifBlank { "服务返回异常，请稍后重试" }))
        }
    } catch (e: HttpException) {
        Result.failure(
            ApiException(
                e.code(),
                if (e.code() == 401) unauthorizedMessage else "服务请求失败",
            )
        )
    } catch (e: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (e: Exception) {
        Result.failure(ApiException(-1, "请求异常：${e.message ?: "未知错误"}"))
    }

    // ---------- 通用 ----------

    suspend fun login(loginName: String, password: String): Result<LoginData> {
        val result = call(unauthorizedMessage = "账号或密码错误") {
            api.login(LoginRequest(loginName, password))
        }
        val data = result.getOrElse { return Result.failure(it) }
        if (data.user.role != "family") {
            authStore.save(data.access_token)
            runCatching { api.logout() }
            authStore.clear()
            return Result.failure(ApiException(403, "当前 App 仅支持家属账号"))
        }
        authStore.save(data.access_token)
        return Result.success(data)
    }

    suspend fun logout() {
        runCatching { api.logout() }
        Session.currentElderId = null
        authStore.clear()
    }

    suspend fun getMe(): Result<UserInfo> = call { api.getMe() }

    suspend fun createWebSocketTicket(): Result<WebSocketTicketData> =
        call { api.createWebSocketTicket() }

    suspend fun getCurrentElderId(): Result<String> {
        Session.currentElderId?.let { return Result.success(it) }
        return getElders().mapCatching { data ->
            data.elders.firstOrNull()?.elder_id
                ?: throw ApiException(404, "当前账号还没有绑定老人")
        }.onSuccess { Session.currentElderId = it }
    }

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

    suspend fun relayCameraAudioPcm(
        deviceId: String,
        pcm: ByteArray,
        sampleRate: Int,
    ): Result<Unit> = call {
        api.relayCameraAudioPcm(
            deviceId,
            sampleRate,
            pcm.toRequestBody("application/octet-stream".toMediaType()),
        )
    }.map { }

    suspend fun getHistoryPlayback(
        deviceId: String,
        elderId: String? = null,
        at: String? = null
    ): Result<HistoryPlayback> = call { api.getHistoryPlayback(deviceId, elderId, at) }

    suspend fun getContacts(elderId: String? = null): Result<ContactsData> =
        call { api.getContacts(elderId) }

    suspend fun getElders(): Result<EldersData> = call { api.getElders() }

}
