package com.tzb.safeguard.data.psychology.network

import com.tzb.safeguard.data.model.ApiResponse
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import retrofit2.http.GET
import retrofit2.http.Query

interface PsychologyApi {
    @GET("api/v1/psychology/overview")
    suspend fun getOverview(
        @Query("elder_id") elderId: String? = null,
    ): ApiResponse<PsychologyOverview>
}

