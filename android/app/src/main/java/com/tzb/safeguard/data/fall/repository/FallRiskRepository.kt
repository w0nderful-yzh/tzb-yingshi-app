package com.tzb.safeguard.data.fall.repository

import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.model.CameraMonitoringStatus
import com.tzb.safeguard.data.fall.network.FallRiskApi
import com.tzb.safeguard.data.repository.ApiException
import java.io.IOException
import retrofit2.HttpException

class FallRiskRepository(private val api: FallRiskApi) {
    suspend fun getOverview(elderId: String? = null): Result<FallRiskOverview> = try {
        val response = api.getOverview(elderId)
        when {
            response.code == 0 && response.data != null -> Result.success(response.data)
            else -> Result.failure(
                ApiException(
                    response.code,
                    response.message.ifBlank { "跌倒风险数据暂不可用" },
                )
            )
        }
    } catch (error: HttpException) {
        Result.failure(ApiException(error.code(), "跌倒风险服务请求失败"))
    } catch (error: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (error: Exception) {
        Result.failure(ApiException(-1, "跌倒风险请求异常"))
    }

    suspend fun startCameraMonitoring(): Result<CameraMonitoringStatus> =
        cameraMonitoringRequest { api.startCameraMonitoring() }

    suspend fun stopCameraMonitoring(): Result<CameraMonitoringStatus> =
        cameraMonitoringRequest { api.stopCameraMonitoring() }

    suspend fun getCameraMonitoringStatus(): Result<CameraMonitoringStatus> =
        cameraMonitoringRequest { api.getCameraMonitoringStatus() }

    private suspend fun cameraMonitoringRequest(
        request: suspend () -> com.tzb.safeguard.data.model.ApiResponse<CameraMonitoringStatus>,
    ): Result<CameraMonitoringStatus> = try {
        val response = request()
        when {
            response.code == 0 && response.data != null -> Result.success(response.data)
            else -> Result.failure(
                ApiException(
                    response.code,
                    response.message.ifBlank { "摄像头跌倒预测服务暂不可用" },
                )
            )
        }
    } catch (error: HttpException) {
        Result.failure(ApiException(error.code(), "摄像头跌倒预测服务请求失败"))
    } catch (error: IOException) {
        Result.failure(ApiException(-1, "网络连接失败，请检查网络后重试"))
    } catch (error: Exception) {
        Result.failure(ApiException(-1, "摄像头跌倒预测服务请求异常"))
    }
}
