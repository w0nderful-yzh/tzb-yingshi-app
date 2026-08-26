package com.tzb.safeguard.data.fall.network

import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.model.CameraMonitoringStatus
import com.tzb.safeguard.data.fall.model.GuardianSessionStatus
import com.tzb.safeguard.data.model.ApiResponse
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

interface FallRiskApi {
    @GET("api/v1/fall-risk/overview")
    suspend fun getOverview(
        @Query("elder_id") elderId: String? = null,
    ): ApiResponse<FallRiskOverview>

    @POST("api/v1/fall-risk/camera-monitoring/start")
    suspend fun startCameraMonitoring(): ApiResponse<CameraMonitoringStatus>

    @POST("api/v1/fall-risk/camera-monitoring/stop")
    suspend fun stopCameraMonitoring(): ApiResponse<CameraMonitoringStatus>

    @GET("api/v1/fall-risk/camera-monitoring/status")
    suspend fun getCameraMonitoringStatus(): ApiResponse<CameraMonitoringStatus>

    @POST("api/v1/guard-session/start")
    suspend fun startGuardSession(): ApiResponse<GuardianSessionStatus>

    @POST("api/v1/guard-session/stop")
    suspend fun stopGuardSession(): ApiResponse<GuardianSessionStatus>

    @GET("api/v1/guard-session/status")
    suspend fun getGuardSessionStatus(): ApiResponse<GuardianSessionStatus>
}
