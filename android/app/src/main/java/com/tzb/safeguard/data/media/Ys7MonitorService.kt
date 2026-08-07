package com.tzb.safeguard.data.media

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.SurfaceTexture
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.tzb.safeguard.MainActivity
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.model.LiveSdkSession
import com.tzb.safeguard.data.model.RealtimeRiskEvent
import com.tzb.safeguard.data.realtime.AlertWebSocketClient
import com.videogo.errorlayer.ErrorInfo
import com.videogo.openapi.EZConstants.EZRealPlayConstants
import com.videogo.openapi.EZOpenSDK
import com.videogo.openapi.EZPlayer
import java.time.Instant
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.MediaPlayer.PlayM4.Player

data class MonitorServiceStatus(
    val enabled: Boolean = false,
    val mediaConnected: Boolean = false,
    val alertsConnected: Boolean = false,
    val detail: String = "持续守护未开启",
    val lastAudioAt: String? = null,
)

class Ys7MonitorService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var monitorJob: Job? = null
    private var alertJob: Job? = null
    private var activePlayer: EZPlayer? = null
    private var activeSdk: EZOpenSDK? = null
    private var activeRelay: CameraAudioRelay? = null
    private var activeSurface: SurfaceTexture? = null
    private var activeHandler: Handler? = null
    private var disconnectSignal: CompletableDeferred<String>? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        scope.launch {
            ServiceLocator.authStore.accessToken.collect { token ->
                if (token == null && status.value.enabled) stopMonitoring(clearPreference = true)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopMonitoring(clearPreference = true)
            return START_NOT_STICKY
        }
        startMonitoring()
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopPlayer()
        scope.cancel()
        _status.value = MonitorServiceStatus()
        super.onDestroy()
    }

    private fun startMonitoring() {
        preferences().edit().putBoolean(KEY_ENABLED, true).apply()
        updateStatus(status.value.copy(enabled = true, detail = "正在连接摄像头与告警通道"))
        ServiceCompat.startForeground(
            this,
            MONITOR_NOTIFICATION_ID,
            monitorNotification(status.value),
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK,
        )
        if (monitorJob?.isActive != true) monitorJob = scope.launch { monitorLoop() }
        if (alertJob?.isActive != true) alertJob = scope.launch { alertLoop() }
    }

    private fun stopMonitoring(clearPreference: Boolean) {
        if (clearPreference) preferences().edit().putBoolean(KEY_ENABLED, false).apply()
        monitorJob?.cancel()
        alertJob?.cancel()
        monitorJob = null
        alertJob = null
        stopPlayer()
        _status.value = MonitorServiceStatus()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private suspend fun monitorLoop() {
        var retryDelay = 1_000L
        while (currentCoroutineContext().isActive) {
            try {
                updateStatus(status.value.copy(mediaConnected = false, detail = "正在连接萤石摄像头"))
                val elderId = ServiceLocator.repository.getCurrentElderId().getOrThrow()
                val devices = ServiceLocator.repository.getDevices(elderId).getOrThrow().devices
                val device = devices.firstOrNull { it.online } ?: devices.firstOrNull()
                    ?: error("当前账号没有可守护的摄像头")
                val session = ServiceLocator.repository.getLiveSdkSession(device.device_id).getOrThrow()
                runPlayer(session)
                error("萤石直播连接已结束")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "media connection failed", error)
                stopPlayer()
                updateStatus(
                    status.value.copy(
                        mediaConnected = false,
                        detail = "摄像头断线，${retryDelay / 1_000} 秒后重连：${error.message}",
                    )
                )
                delay(retryDelay)
                retryDelay = (retryDelay * 2).coerceAtMost(30_000L)
            }
        }
    }

    private suspend fun runPlayer(session: LiveSdkSession) {
        val application = applicationContext as Application
        val sdk = Ys7SdkRuntime.configure(application, session)
        val player = sdk.createPlayer(session.device_serial, session.channel_no)
        val relay = CameraAudioRelay(ServiceLocator.repository, session.device_serial, scope)
        val surface = SurfaceTexture(0).apply { setDefaultBufferSize(16, 16) }
        val disconnected = CompletableDeferred<String>()
        val handler = Handler(Looper.getMainLooper()) { message ->
            when (message.what) {
                EZRealPlayConstants.MSG_REALPLAY_PLAY_SUCCESS -> {
                    val audioEngine = Player.getInstance()
                    audioEngine.setAudioDataCallBack(player.playPort) { _, pcm, size, sampleRate ->
                        relay.accept(pcm, size, sampleRate)
                        val now = System.currentTimeMillis()
                        if (now - lastAudioStatusAt >= 1_000) {
                            lastAudioStatusAt = now
                            updateStatus(
                                status.value.copy(
                                    mediaConnected = true,
                                    detail = "摄像头音频监听中",
                                    lastAudioAt = Instant.ofEpochMilli(now).toString(),
                                )
                            )
                        }
                    }
                    player.openSound()
                    audioEngine.adjustWaveAudio(player.playPort, Player.VOLUME_MUTE)
                    updateStatus(status.value.copy(mediaConnected = true, detail = "摄像头音频监听中"))
                }

                EZRealPlayConstants.MSG_REALPLAY_PLAY_FAIL -> {
                    val info = message.obj as? ErrorInfo
                    disconnected.complete("萤石播放失败（${info?.errorCode ?: "unknown"}）")
                }
            }
            true
        }
        activeSdk = sdk
        activePlayer = player
        activeRelay = relay
        activeSurface = surface
        activeHandler = handler
        disconnectSignal = disconnected
        player.setSurfaceEx(surface)
        player.setHandler(handler)
        player.setHardDecode(false)
        player.closeSound()
        check(player.startRealPlay()) { "萤石直播启动失败" }
        disconnected.await().let { error(it) }
    }

    private suspend fun alertLoop() {
        var retryDelay = 1_000L
        val socketClient = AlertWebSocketClient(ServiceLocator.httpClient)
        while (currentCoroutineContext().isActive) {
            try {
                val ticket = ServiceLocator.repository.createWebSocketTicket().getOrThrow().ticket
                socketClient.listen(
                    ticket = ticket,
                    onConnected = {
                        retryDelay = 1_000L
                        updateStatus(status.value.copy(alertsConnected = true))
                    },
                    onEvent = ::showRiskNotification,
                )
                error("告警连接已关闭")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "alert connection failed", error)
                updateStatus(status.value.copy(alertsConnected = false))
                delay(retryDelay)
                retryDelay = (retryDelay * 2).coerceAtMost(30_000L)
            }
        }
    }

    private fun stopPlayer() {
        disconnectSignal?.cancel()
        disconnectSignal = null
        activePlayer?.let { player ->
            runCatching { Player.getInstance().setAudioDataCallBack(player.playPort, null) }
            runCatching { player.stopRealPlay() }
            activeSdk?.releasePlayer(player)
        }
        activeHandler?.removeCallbacksAndMessages(null)
        activeRelay?.close()
        activeSurface?.release()
        activePlayer = null
        activeSdk = null
        activeHandler = null
        activeRelay = null
        activeSurface = null
    }

    private fun updateStatus(next: MonitorServiceStatus) {
        _status.value = next
        if (next.enabled) {
            getSystemService(NotificationManager::class.java)
                .notify(MONITOR_NOTIFICATION_ID, monitorNotification(next))
        }
    }

    private fun monitorNotification(current: MonitorServiceStatus) =
        NotificationCompat.Builder(this, MONITOR_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setContentTitle("守护安 · 持续守护运行中")
            .setContentText(current.detail)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(mainPendingIntent())
            .addAction(
                android.R.drawable.ic_media_pause,
                "停止守护",
                PendingIntent.getService(
                    this,
                    2,
                    Intent(this, Ys7MonitorService::class.java).setAction(ACTION_STOP),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                ),
            )
            .build()

    private fun showRiskNotification(event: RealtimeRiskEvent) {
        val intent = Intent(this, MainActivity::class.java)
            .putExtra(MainActivity.EXTRA_EVENT_ID, event.event_id)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val pendingIntent = PendingIntent.getActivity(
            this,
            event.event_id.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        getSystemService(NotificationManager::class.java).notify(
            event.event_id.hashCode(),
            NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
                .setSmallIcon(android.R.drawable.stat_notify_error)
                .setContentTitle(event.title)
                .setContentText(event.summary)
                .setStyle(NotificationCompat.BigTextStyle().bigText(event.summary))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(pendingIntent)
                .build(),
        )
    }

    private fun mainPendingIntent(): PendingIntent = PendingIntent.getActivity(
        this,
        1,
        Intent(this, MainActivity::class.java),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )

    private fun createNotificationChannels() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                MONITOR_CHANNEL_ID,
                "持续守护状态",
                NotificationManager.IMPORTANCE_LOW,
            )
        )
        manager.createNotificationChannel(
            NotificationChannel(
                ALERT_CHANNEL_ID,
                "风险告警",
                NotificationManager.IMPORTANCE_HIGH,
            )
        )
    }

    private fun preferences() = getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    companion object {
        const val ACTION_START = "com.tzb.safeguard.action.START_MONITOR"
        const val ACTION_STOP = "com.tzb.safeguard.action.STOP_MONITOR"
        private const val PREFERENCES = "ys7_monitor"
        private const val KEY_ENABLED = "enabled"
        private const val MONITOR_CHANNEL_ID = "continuous_monitor"
        private const val ALERT_CHANNEL_ID = "risk_alerts"
        private const val MONITOR_NOTIFICATION_ID = 1001
        private const val TAG = "Ys7MonitorService"
        private val _status = MutableStateFlow(MonitorServiceStatus())
        val status = _status.asStateFlow()
        private var lastAudioStatusAt = 0L

        fun start(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, Ys7MonitorService::class.java).setAction(ACTION_START),
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, Ys7MonitorService::class.java).setAction(ACTION_STOP)
            )
        }
    }
}
