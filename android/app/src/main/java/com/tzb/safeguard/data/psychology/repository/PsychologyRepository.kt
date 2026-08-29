package com.tzb.safeguard.data.psychology.repository

import com.tzb.safeguard.data.psychology.model.CognitiveOverview
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import com.tzb.safeguard.data.psychology.network.PsychologyApi
import com.tzb.safeguard.data.repository.ApiException
import java.io.IOException
import retrofit2.HttpException

class PsychologyRepository(private val api: PsychologyApi) {
    suspend fun getOverview(elderId: String? = null): Result<PsychologyOverview> = try {
        val response = api.getOverview(elderId)
        when {
            response.code == 0 && response.data != null -> Result.success(response.data)
            else -> Result.failure(
                ApiException(
                    response.code,
                    response.message.ifBlank { "心理健康评估数据暂不可用" },
                )
            )
        }
    } catch (error: HttpException) {
        Result.failure(ApiException(error.code(), "心理健康评估服务请求失败"))
    } catch (error: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (error: Exception) {
        Result.failure(ApiException(-1, "心理健康评估请求异常"))
    }

    suspend fun getCognitiveOverview(elderId: String? = null): Result<CognitiveOverview> = try {
        val response = api.getCognitiveOverview(elderId)
        when {
            response.code == 0 && response.data != null -> Result.success(response.data)
            else -> Result.failure(
                ApiException(
                    response.code,
                    response.message.ifBlank { "认知状态辅助评估数据暂不可用" },
                )
            )
        }
    } catch (error: HttpException) {
        Result.failure(ApiException(error.code(), "认知状态辅助评估服务请求失败"))
    } catch (error: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (error: Exception) {
        Result.failure(ApiException(-1, "认知状态辅助评估请求异常"))
    }
}

