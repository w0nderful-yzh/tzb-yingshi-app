package com.tzb.safeguard.data.network

import com.tzb.safeguard.data.model.*
import okhttp3.RequestBody
import retrofit2.http.*

/**
 * 网络层接口定义。
 * 路径与 docs/api/app-client-api.md 一一对应；已实现的后端接口（如防诈分析）
 * 供联调工具使用，App 侧不直接调用。
 */
interface ApiService {

    // ---------- 通用 ----------

    @POST("api/v1/auth/login")
    suspend fun login(@Body body: LoginRequest): ApiResponse<LoginData>

    @POST("api/v1/auth/logout")
    suspend fun logout(): ApiResponse<EmptyData>

    @POST("api/v1/ws/tickets")
    suspend fun createWebSocketTicket(): ApiResponse<WebSocketTicketData>

    /** 当前用户信息（老人端/家属端通用） */
    @GET("api/v1/users/me")
    suspend fun getMe(): ApiResponse<UserInfo>

    // ---------- 风险事件 ----------

    /**
     * 事件列表。
     * 老人端不传 elder_id（固定本人）；家属端必须显式传，服务端校验绑定关系。
     */
    @GET("api/v1/events")
    suspend fun getEvents(
        @Query("elder_id") elderId: String? = null,
        @Query("level") level: String? = null,
        @Query("status") status: String? = null,
        @Query("limit") limit: Int = 50,
        @Query("cursor") cursor: String? = null
    ): ApiResponse<EventListData>

    @GET("api/v1/events/{event_id}")
    suspend fun getEventDetail(@Path("event_id") eventId: String): ApiResponse<EventDetail>

    /** 家属端处置：acknowledged | resolved | false_alarm */
    @PATCH("api/v1/events/{event_id}/status")
    suspend fun patchEventStatus(
        @Path("event_id") eventId: String,
        @Body body: StatusPatch
    ): ApiResponse<EmptyData>

    /** TODO 后端：向设备播报提醒或发起家属外呼，当前返回 501。 */
    @POST("api/v1/events/{event_id}/intervention-reminder")
    suspend fun sendInterventionReminder(
        @Path("event_id") eventId: String,
        @Body body: InterventionReminder
    ): ApiResponse<EmptyData>

    // ---------- 设备 ----------

    @GET("api/v1/devices")
    suspend fun getDevices(@Query("elder_id") elderId: String? = null): ApiResponse<DeviceListData>

    /** 短时有效的萤石播放地址；AppSecret 不下发客户端 */
    @GET("api/v1/devices/{device_id}/live-url")
    suspend fun getLiveUrl(@Path("device_id") deviceId: String): ApiResponse<LiveUrl>

    /** H.265 设备使用原生萤石 SDK；AppSecret 始终留在后端。 */
    @GET("api/v1/devices/{device_id}/live-sdk-session")
    suspend fun getLiveSdkSession(
        @Path("device_id") deviceId: String
    ): ApiResponse<LiveSdkSession>

    /** 将 EZOpenSDK 解码后的 16 kHz 单声道 PCM 转发给后端实时 VAD。 */
    @Headers("Content-Type: application/octet-stream")
    @POST("api/v1/devices/{device_id}/audio-pcm")
    suspend fun relayCameraAudioPcm(
        @Path("device_id") deviceId: String,
        @Header("X-Audio-Sample-Rate") sampleRate: Int,
        @Body pcm: RequestBody,
    ): ApiResponse<EmptyData>

    /** TODO 后端：获取事件时间点附近的萤石历史回放地址，当前返回 501。 */
    @GET("api/v1/devices/{device_id}/history-playback")
    suspend fun getHistoryPlayback(
        @Path("device_id") deviceId: String,
        @Query("elder_id") elderId: String? = null,
        @Query("at") at: String? = null,
        @Query("duration_seconds") durationSeconds: Int = 30
    ): ApiResponse<HistoryPlayback>

    // ---------- 家属端 ----------

    @GET("api/v1/family/elders")
    suspend fun getElders(): ApiResponse<EldersData>

    @GET("api/v1/contacts")
    suspend fun getContacts(@Query("elder_id") elderId: String? = null): ApiResponse<ContactsData>

}
