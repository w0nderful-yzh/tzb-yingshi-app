package com.tzb.safeguard

import android.app.Application
import com.tzb.safeguard.data.network.NetworkModule
import com.tzb.safeguard.data.repository.SafeRepository

/**
 * 轻量服务定位器：不引入 Hilt，Demo 阶段手动装配，降低接入成本。
 */
object ServiceLocator {
    val repository: SafeRepository by lazy { SafeRepository(NetworkModule.apiService) }
}

/** 当前会话角色：elder 老人端 / family 家属端；角色选择页写入 */
object Session {
    var role: String = "elder"
    var currentElderId: String = "u-elder-001"   // 家属端当前查看的老人（联调期固定 1 位）
}

class SafeGuardApp : Application()
