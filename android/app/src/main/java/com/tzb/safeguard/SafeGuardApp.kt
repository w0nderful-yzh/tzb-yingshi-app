package com.tzb.safeguard

import android.app.Application
import com.tzb.safeguard.data.auth.AuthStore
import com.tzb.safeguard.data.network.NetworkModule
import com.tzb.safeguard.data.repository.SafeRepository
import okhttp3.OkHttpClient

/**
 * 轻量服务定位器：不引入 Hilt，Demo 阶段手动装配，降低接入成本。
 */
object ServiceLocator {
    lateinit var authStore: AuthStore
        private set
    lateinit var repository: SafeRepository
        private set
    lateinit var httpClient: OkHttpClient
        private set

    fun initialize(application: Application) {
        authStore = AuthStore(application)
        httpClient = NetworkModule.createHttpClient(authStore)
        repository = SafeRepository(
            NetworkModule.createApiService(httpClient),
            authStore,
        )
    }
}

/** 当前为家属端单角色；登录后从绑定关系选择首位老人。 */
object Session {
    var currentElderId: String? = null
}

class SafeGuardApp : Application() {
    override fun onCreate() {
        super.onCreate()
        ServiceLocator.initialize(this)
    }
}
