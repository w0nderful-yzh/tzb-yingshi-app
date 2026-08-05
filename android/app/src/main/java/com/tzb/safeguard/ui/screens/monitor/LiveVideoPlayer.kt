package com.tzb.safeguard.ui.screens.monitor

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.ViewGroup
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.tzb.safeguard.data.model.LiveSdkSession
import com.tzb.safeguard.ui.theme.WarnRed
import com.videogo.errorlayer.ErrorInfo
import com.videogo.exception.ErrorCode
import com.videogo.openapi.EZConstants.EZRealPlayConstants
import com.videogo.openapi.EZOpenSDK
import com.videogo.openapi.EZPlayer

private object Ys7SdkRuntime {
    private var initializedAppKey: String? = null

    @Synchronized
    fun configure(application: Application, session: LiveSdkSession): EZOpenSDK {
        if (initializedAppKey != session.app_key) {
            if (initializedAppKey != null) {
                EZOpenSDK.finiLib()
            }
            // SDK 调试日志会包含 AccessToken，所有构建均必须关闭。
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

@Composable
fun LiveVideoPlayer(
    liveSession: LiveSdkSession?,
    deviceName: String,
    streamLoading: Boolean,
    streamError: String?,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var playerError by remember(liveSession) { mutableStateOf<String?>(null) }
    var muted by remember { mutableStateOf(true) }
    var playing by remember(liveSession) { mutableStateOf(false) }

    Box(modifier = modifier.background(Color(0xFF101318), RoundedCornerShape(14.dp))) {
        if (liveSession != null) {
            Ys7SurfacePlayer(
                session = liveSession,
                muted = muted,
                onPlaying = {
                    playing = true
                    playerError = null
                },
                onError = {
                    playing = false
                    playerError = it
                },
                modifier = Modifier.fillMaxSize(),
            )
        }

        Surface(
            color = WarnRed,
            shape = RoundedCornerShape(6.dp),
            modifier = Modifier.align(Alignment.TopStart).padding(10.dp),
        ) {
            Text(
                "● 直播 · $deviceName",
                color = Color.White,
                fontSize = 13.sp,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
            )
        }

        IconButton(
            onClick = { muted = !muted },
            enabled = playing,
            modifier = Modifier.align(Alignment.TopEnd).padding(4.dp),
        ) {
            Icon(
                imageVector = if (muted) {
                    Icons.AutoMirrored.Filled.VolumeOff
                } else {
                    Icons.AutoMirrored.Filled.VolumeUp
                },
                contentDescription = if (muted) "打开直播声音" else "静音",
                tint = if (playing) Color.White else Color(0xFF7B8491),
            )
        }

        when {
            streamLoading || (liveSession != null && !playing && playerError == null) -> {
                CircularProgressIndicator(
                    color = Color.White,
                    modifier = Modifier.align(Alignment.Center),
                )
            }

            liveSession == null || streamError != null || playerError != null -> {
                androidx.compose.foundation.layout.Column(
                    modifier = Modifier.align(Alignment.Center).padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Icon(
                        Icons.Filled.Videocam,
                        contentDescription = null,
                        tint = Color(0xFF9AA3AF),
                    )
                    Text(
                        streamError ?: playerError ?: "直播会话尚未就绪",
                        color = Color(0xFFD1D5DB),
                        fontSize = 14.sp,
                    )
                    IconButton(onClick = onRetry) {
                        Icon(
                            Icons.Filled.Refresh,
                            contentDescription = "刷新直播",
                            tint = Color.White,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun Ys7SurfacePlayer(
    session: LiveSdkSession,
    muted: Boolean,
    onPlaying: () -> Unit,
    onError: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val application = context.applicationContext as Application
    val playerResult = remember(session) {
        runCatching {
            Ys7SdkRuntime.configure(application, session).let { sdk ->
                sdk to sdk.createPlayer(session.device_serial, session.channel_no)
            }
        }
    }
    val pair = playerResult.getOrNull()

    LaunchedEffect(playerResult) {
        playerResult.exceptionOrNull()?.let {
            onError(it.message ?: "萤石直播组件初始化失败")
        }
    }
    if (pair == null) return

    val (sdk, player) = pair
    val surfaceView = remember(session) {
        SurfaceView(context).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            keepScreenOn = true
        }
    }
    val handler = remember(session) {
        Handler(Looper.getMainLooper()) { message ->
            when (message.what) {
                EZRealPlayConstants.MSG_REALPLAY_PLAY_SUCCESS -> onPlaying()
                EZRealPlayConstants.MSG_REALPLAY_PLAY_FAIL -> {
                    val errorInfo = message.obj as? ErrorInfo
                    val errorCode = errorInfo?.errorCode
                    val text = when (errorCode) {
                        ErrorCode.ERROR_INNER_VERIFYCODE_NEED,
                        ErrorCode.ERROR_INNER_VERIFYCODE_ERROR ->
                            "设备已启用视频加密，请配置设备验证码"
                        null -> "萤石直播播放失败，请刷新重试"
                        else -> "萤石直播播放失败（错误码 $errorCode）"
                    }
                    onError(text)
                }
            }
            true
        }
    }

    LaunchedEffect(player, muted) {
        if (muted) player.closeSound() else player.openSound()
    }

    DisposableEffect(player, surfaceView, lifecycleOwner) {
        var started = false

        fun start(holder: SurfaceHolder) {
            if (started || !holder.surface.isValid) return
            player.setSurfaceHold(holder)
            player.setHandler(handler)
            player.setHardDecode(true)
            player.closeSound()
            started = player.startRealPlay()
            if (!started) onError("萤石直播启动失败，请刷新重试")
        }

        fun stop() {
            if (!started) return
            player.stopRealPlay()
            started = false
        }

        val surfaceCallback = object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                    start(holder)
                }
            }

            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int,
            ) = Unit

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                stop()
                player.setSurfaceHold(null)
            }
        }
        val lifecycleObserver = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> start(surfaceView.holder)
                Lifecycle.Event.ON_STOP -> stop()
                else -> Unit
            }
        }

        surfaceView.holder.addCallback(surfaceCallback)
        lifecycleOwner.lifecycle.addObserver(lifecycleObserver)
        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            start(surfaceView.holder)
        }

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(lifecycleObserver)
            surfaceView.holder.removeCallback(surfaceCallback)
            stop()
            player.setSurfaceHold(null)
            handler.removeCallbacksAndMessages(null)
            sdk.releasePlayer(player)
        }
    }

    AndroidView(
        factory = { surfaceView },
        modifier = modifier,
    )
}
