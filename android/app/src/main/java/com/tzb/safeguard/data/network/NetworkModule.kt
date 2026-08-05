package com.tzb.safeguard.data.network

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import com.tzb.safeguard.BuildConfig
import com.tzb.safeguard.Session
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit
import java.util.UUID

/**
 * 网络层装配：OkHttp + Retrofit + kotlinx.serialization。
 * MOCK_MODE=true 时挂 MockInterceptor，所有请求在本地按契约返回演示数据。
 */
object NetworkModule {

    private val json = Json {
        ignoreUnknownKeys = true      // 后端新增字段不崩溃，向前兼容
        coerceInputValues = true      // null -> 默认值
        explicitNulls = false
    }

    private val okHttpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .apply {
                addInterceptor { chain ->
                    val original = chain.request()
                    val request = original.newBuilder()
                        .header("X-Demo-Role", Session.role)
                        .apply {
                            if (original.method != "GET" && original.header("Idempotency-Key") == null) {
                                header("Idempotency-Key", UUID.randomUUID().toString())
                            }
                        }
                        .build()
                    chain.proceed(request)
                }
                if (BuildConfig.MOCK_MODE) {
                    addInterceptor(MockInterceptor())
                }
                if (BuildConfig.DEBUG) {
                    addInterceptor(
                        HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC }
                    )
                }
                // TODO(正式鉴权): 用 Authorization Bearer 替换 X-Demo-Role。
            }
            .build()
    }

    val apiService: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
            .build()
            .create(ApiService::class.java)
    }
}
