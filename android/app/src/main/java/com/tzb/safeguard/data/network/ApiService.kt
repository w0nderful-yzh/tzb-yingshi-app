package com.tzb.safeguard.data.network

import com.tzb.safeguard.data.model.*
import retrofit2.http.*

/**
 * 网络层接口定义。
 * 路径与 docs/api/app-client-api.md 一一对应；已实现的后端接口（如防诈分析）
 * 供联调工具使用，App 侧不直接调用。
 */
interface ApiService {

    // ---------- 通用 ----------

    /** 当前用户信息（老人端/家属端通用） */
    @GET("api/v1/users/me")
    suspend fun getMe(): ApiResponse<UserInfo>

    /** 首页安全状态聚合（老人端） */
    @GET("api/v1/safety/status")
    suspend fun getSafetyStatus(): ApiResponse<SafetyStatus>

    /** 一键紧急求助（老人端） */
    @POST("api/v1/sos")
    suspend fun postSos(@Body body: SosRequest): ApiResponse<SosResult>

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

    /** 老人端告警确认：im_ok / need_help */
    @POST("api/v1/events/{event_id}/confirm")
    suspend fun confirmEvent(
        @Path("event_id") eventId: String,
        @Body body: ConfirmRequest
    ): ApiResponse<EmptyData>

    /** 家属端处置：acknowledged | resolved | false_alarm */
    @PATCH("api/v1/events/{event_id}/status")
    suspend fun patchEventStatus(
        @Path("event_id") eventId: String,
        @Body body: StatusPatch
    ): ApiResponse<EmptyData>

    /** 家属端一键回呼老人 */
    @POST("api/v1/events/{event_id}/call")
    suspend fun callElder(@Path("event_id") eventId: String): ApiResponse<EmptyData>

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

    // ---------- 家属端 ----------

    @GET("api/v1/family/elders")
    suspend fun getElders(): ApiResponse<EldersData>

    @GET("api/v1/contacts")
    suspend fun getContacts(@Query("elder_id") elderId: String? = null): ApiResponse<ContactsData>

    /** 事件分级别计数（近 N 天） */
    @GET("api/v1/stats/events")
    suspend fun getEventsStats(
        @Query("elder_id") elderId: String? = null,
        @Query("days") days: Int = 30
    ): ApiResponse<EventsStatsData>

    /** 24 小时活动热力（近 N 天平均） */
    @GET("api/v1/stats/activity")
    suspend fun getActivityStats(
        @Query("elder_id") elderId: String? = null,
        @Query("days") days: Int = 7
    ): ApiResponse<ActivityData>
}
