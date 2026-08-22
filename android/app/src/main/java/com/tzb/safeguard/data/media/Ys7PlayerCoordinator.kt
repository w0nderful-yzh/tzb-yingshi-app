package com.tzb.safeguard.data.media

import android.app.Application
import android.graphics.SurfaceTexture
import android.os.Handler
import android.os.Looper
import android.view.SurfaceHolder
import com.tzb.safeguard.data.model.LiveSdkSession
import com.videogo.errorlayer.ErrorInfo
import com.videogo.openapi.EZConstants.EZRealPlayConstants
import com.videogo.openapi.EZOpenSDK
import com.videogo.openapi.EZPlayer
import org.MediaPlayer.PlayM4.Player

/**
 * Owns the single process-wide EZPlayer used by visible preview and background monitoring.
 *
 * The lifecycle lock protects Kotlin state only. No EZPlayer/PlayM4 call is allowed while that
 * lock is held because native callback registration and player teardown can wait for callbacks.
 */
object Ys7PlayerCoordinator {
    interface Listener {
        fun onPlaying()
        fun onError(errorCode: Int?)
    }

    class Lease internal constructor(private val token: Any) : AutoCloseable {
        fun attachSurface(holder: SurfaceHolder) =
            Ys7PlayerCoordinator.attachSurface(token, holder)

        fun detachSurface(holder: SurfaceHolder) =
            Ys7PlayerCoordinator.detachSurface(token, holder)

        fun setMuted(muted: Boolean) = Ys7PlayerCoordinator.setMuted(token, muted)

        fun setAudioConsumer(consumer: ((ByteArray, Int, Int) -> Unit)?) =
            Ys7PlayerCoordinator.setAudioConsumer(token, consumer)

        override fun close() = Ys7PlayerCoordinator.release(token)
    }

    private data class SessionKey(
        val appKey: String,
        val deviceSerial: String,
        val channelNo: Int,
    )

    private data class Owner(
        val listener: Listener,
        var muted: Boolean = true,
        var surface: SurfaceHolder? = null,
        var surfaceOrder: Long = 0,
        var audioConsumer: ((ByteArray, Int, Int) -> Unit)? = null,
    )

    private data class PlayerResources(
        val sdk: EZOpenSDK,
        val player: EZPlayer,
        val handler: Handler,
        val backgroundSurface: SurfaceTexture,
    )

    private data class SurfaceBinding(
        val player: EZPlayer,
        val visibleSurface: SurfaceHolder?,
        val backgroundSurface: SurfaceTexture,
    )

    private data class VolumeBinding(
        val player: EZPlayer,
        val audible: Boolean,
    )

    private val lifecycleLock = Any()
    private val owners = linkedMapOf<Any, Owner>()
    private var currentKey: SessionKey? = null
    private var resources: PlayerResources? = null
    private var surfaceSequence = 0L
    private var playing = false
    private var failed = false

    /** Read directly from the native PCM thread; never takes [lifecycleLock]. */
    @Volatile
    private var audioConsumersSnapshot: List<(ByteArray, Int, Int) -> Unit> = emptyList()

    fun acquire(
        application: Application,
        session: LiveSdkSession,
        listener: Listener,
    ): Lease {
        val token = Any()
        val shouldCreate: Boolean
        val alreadyPlaying: Boolean
        synchronized(lifecycleLock) {
            val requestedKey = session.key()
            val activeKey = currentKey
            check(activeKey == null || activeKey == requestedKey) {
                "已有其他萤石设备正在使用共享播放器"
            }
            owners[token] = Owner(listener = listener)
            currentKey = requestedKey
            refreshAudioConsumersLocked()
            shouldCreate = resources == null || failed
            alreadyPlaying = playing && !shouldCreate
        }

        try {
            if (shouldCreate) recreatePlayer(application, session)
        } catch (error: Throwable) {
            release(token)
            throw error
        }
        if (alreadyPlaying) listener.onPlaying()
        return Lease(token)
    }

    private fun attachSurface(token: Any, holder: SurfaceHolder) {
        val binding = synchronized(lifecycleLock) {
            val owner = owners[token] ?: return
            owner.surface = holder
            owner.surfaceOrder = ++surfaceSequence
            surfaceBindingLocked()
        }
        applySurfaceBinding(binding)
    }

    private fun detachSurface(token: Any, holder: SurfaceHolder) {
        val binding = synchronized(lifecycleLock) {
            val owner = owners[token] ?: return
            if (owner.surface !== holder) return
            owner.surface = null
            surfaceBindingLocked()
        }
        applySurfaceBinding(binding)
    }

    private fun setMuted(token: Any, muted: Boolean) {
        val binding = synchronized(lifecycleLock) {
            owners[token]?.muted = muted
            volumeBindingLocked()
        }
        applyVolumeBinding(binding)
    }

    private fun setAudioConsumer(
        token: Any,
        consumer: ((ByteArray, Int, Int) -> Unit)?,
    ) {
        synchronized(lifecycleLock) {
            owners[token]?.audioConsumer = consumer
            refreshAudioConsumersLocked()
        }
    }

    private fun release(token: Any) {
        var resourcesToRelease: PlayerResources? = null
        var surfaceBinding: SurfaceBinding? = null
        var volumeBinding: VolumeBinding? = null
        synchronized(lifecycleLock) {
            if (owners.remove(token) == null) return
            refreshAudioConsumersLocked()
            if (owners.isEmpty()) {
                resourcesToRelease = resources
                resources = null
                currentKey = null
                playing = false
                failed = false
            } else {
                surfaceBinding = surfaceBindingLocked()
                volumeBinding = volumeBindingLocked()
            }
        }

        resourcesToRelease?.let(::releaseNativePlayer)
        applySurfaceBinding(surfaceBinding)
        applyVolumeBinding(volumeBinding)
    }

    private fun recreatePlayer(application: Application, session: LiveSdkSession) {
        val previous = synchronized(lifecycleLock) {
            val detached = resources
            resources = null
            playing = false
            failed = false
            audioConsumersSnapshot = emptyList()
            detached
        }
        previous?.let(::releaseNativePlayer)

        val configuredSdk: EZOpenSDK
        val createdPlayer: EZPlayer
        val handler: Handler
        val backgroundSurface: SurfaceTexture
        try {
            configuredSdk = Ys7SdkRuntime.configure(application, session)
            createdPlayer = configuredSdk.createPlayer(session.device_serial, session.channel_no)
            handler = Handler(Looper.getMainLooper()) { message ->
                when (message.what) {
                    EZRealPlayConstants.MSG_REALPLAY_PLAY_SUCCESS -> onPlaySuccess(createdPlayer)
                    EZRealPlayConstants.MSG_REALPLAY_PLAY_FAIL -> {
                        val info = message.obj as? ErrorInfo
                        onPlayFailure(createdPlayer, info?.errorCode)
                    }
                }
                true
            }
            backgroundSurface = SurfaceTexture(0).apply { setDefaultBufferSize(16, 16) }
        } catch (error: Throwable) {
            synchronized(lifecycleLock) { failed = true }
            throw error
        }

        val candidate = PlayerResources(
            sdk = configuredSdk,
            player = createdPlayer,
            handler = handler,
            backgroundSurface = backgroundSurface,
        )
        val installed = synchronized(lifecycleLock) {
            if (owners.isEmpty() || currentKey != session.key() || resources != null) {
                false
            } else {
                resources = candidate
                refreshAudioConsumersLocked()
                true
            }
        }
        if (!installed) {
            releaseNativePlayer(candidate)
            return
        }

        try {
            createdPlayer.setHandler(handler)
            // Software decoding preserves the PCM callback behavior used by the monitor service.
            createdPlayer.setHardDecode(false)
            createdPlayer.closeSound()
            applySurfaceBinding(snapshotSurfaceBinding(createdPlayer))
            check(createdPlayer.startRealPlay()) { "萤石直播启动失败" }
        } catch (error: Throwable) {
            val detached = synchronized(lifecycleLock) {
                if (resources === candidate) {
                    resources = null
                    playing = false
                    failed = true
                    audioConsumersSnapshot = emptyList()
                    candidate
                } else {
                    null
                }
            }
            detached?.let(::releaseNativePlayer)
            throw error
        }
    }

    private fun onPlaySuccess(activePlayer: EZPlayer) {
        val listeners: List<Listener>
        val volumeBinding: VolumeBinding?
        synchronized(lifecycleLock) {
            if (resources?.player !== activePlayer) return
            failed = false
            playing = true
            listeners = owners.values.map { it.listener }
            volumeBinding = volumeBindingLocked()
        }

        val audioEngine = Player.getInstance()
        audioEngine.setAudioDataCallBack(activePlayer.playPort) { _, pcm, size, sampleRate ->
            audioConsumersSnapshot.forEach { consumer ->
                runCatching { consumer(pcm, size, sampleRate) }
            }
        }
        activePlayer.openSound()
        applyVolumeBinding(volumeBinding)
        listeners.forEach { it.onPlaying() }
    }

    private fun onPlayFailure(activePlayer: EZPlayer, errorCode: Int?) {
        val listeners = synchronized(lifecycleLock) {
            if (resources?.player !== activePlayer) return
            failed = true
            playing = false
            owners.values.map { it.listener }
        }
        listeners.forEach { it.onError(errorCode) }
    }

    private fun snapshotSurfaceBinding(activePlayer: EZPlayer): SurfaceBinding? =
        synchronized(lifecycleLock) {
            if (resources?.player !== activePlayer) null else surfaceBindingLocked()
        }

    private fun surfaceBindingLocked(): SurfaceBinding? {
        val current = resources ?: return null
        val visibleSurface = owners.values
            .filter { it.surface?.surface?.isValid == true }
            .maxByOrNull { it.surfaceOrder }
            ?.surface
        return SurfaceBinding(
            player = current.player,
            visibleSurface = visibleSurface,
            backgroundSurface = current.backgroundSurface,
        )
    }

    private fun volumeBindingLocked(): VolumeBinding? {
        val current = resources ?: return null
        if (!playing) return null
        return VolumeBinding(
            player = current.player,
            audible = owners.values.any { it.surface != null && !it.muted },
        )
    }

    private fun refreshAudioConsumersLocked() {
        audioConsumersSnapshot = owners.values.mapNotNull { it.audioConsumer }
    }

    private fun applySurfaceBinding(binding: SurfaceBinding?) {
        binding ?: return
        if (binding.visibleSurface != null) {
            binding.player.setSurfaceHold(binding.visibleSurface)
        } else {
            binding.player.setSurfaceHold(null)
            binding.player.setSurfaceEx(binding.backgroundSurface)
        }
    }

    private fun applyVolumeBinding(binding: VolumeBinding?) {
        binding ?: return
        Player.getInstance().adjustWaveAudio(
            binding.player.playPort,
            if (binding.audible) Player.VOLUME_DEFAULT else Player.VOLUME_MUTE,
        )
    }

    private fun releaseNativePlayer(current: PlayerResources) {
        runCatching { Player.getInstance().setAudioDataCallBack(current.player.playPort, null) }
        runCatching { current.player.setSurfaceHold(null) }
        runCatching { current.player.stopRealPlay() }
        runCatching { current.sdk.releasePlayer(current.player) }
        current.handler.removeCallbacksAndMessages(null)
        current.backgroundSurface.release()
    }

    private fun LiveSdkSession.key() = SessionKey(app_key, device_serial, channel_no)
}
