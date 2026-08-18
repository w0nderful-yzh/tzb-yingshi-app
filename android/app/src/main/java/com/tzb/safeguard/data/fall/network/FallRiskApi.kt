package com.tzb.safeguard.data.fall.network

import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.model.ApiResponse
import retrofit2.http.GET
import retrofit2.http.Query

interface FallRiskApi {
    @GET("api/v1/fall-risk/overview")
    suspend fun getOverview(
        @Query("elder_id") elderId: String? = null,
    ): ApiResponse<FallRiskOverview>
}
