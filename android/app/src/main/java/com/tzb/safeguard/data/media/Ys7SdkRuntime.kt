package com.tzb.safeguard.data.media

import android.app.Application
import com.tzb.safeguard.data.model.LiveSdkSession
import com.videogo.openapi.EZOpenSDK

/** Process-wide EZOpenSDK initialization shared by the monitor service and visible player. */
object Ys7SdkRuntime {
    private var initializedAppKey: String? = null

    @Synchronized
    fun configure(application: Application, session: LiveSdkSession): EZOpenSDK {
        if (initializedAppKey != session.app_key) {
            if (initializedAppKey != null) EZOpenSDK.finiLib()
            // SDK logs may contain AccessToken. They stay disabled in every build type.
            EZOpenSDK.showSDKLog(false)
            EZOpenSDK.enableP2P(false)
            check(EZOpenSDK.initLib(application, session.app_key)) {
                "萤石直播组件初始化失败"
            }
            initializedAppKey = session.app_key
        }
        return EZOpenSDK.getInstance().also { it.setAccessToken(session.access_token) }
    }
}
