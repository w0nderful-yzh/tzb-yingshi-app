package com.tzb.safeguard.data.media

import com.tzb.safeguard.data.repository.SafeRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

private const val TARGET_SAMPLE_RATE = 16_000
private const val PCM_BYTES_PER_SECOND = TARGET_SAMPLE_RATE * 2

/**
 * Keeps the native EZOpenSDK callback non-blocking and batches PCM into one-second uploads.
 * Speech boundaries remain owned by the backend VAD; this is transport batching, not ASR chunking.
 */
class CameraAudioRelay(
    private val repository: SafeRepository,
    private val deviceId: String,
    scope: CoroutineScope,
) {
    private val uploads = Channel<ByteArray>(
        capacity = 4,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val pending = ByteArray(PCM_BYTES_PER_SECOND)
    private var pendingSize = 0
    private val uploadJob = scope.launch(Dispatchers.IO) {
        for (pcm in uploads) {
            repository.relayCameraAudioPcm(deviceId, pcm, TARGET_SAMPLE_RATE)
        }
    }

    fun accept(data: ByteArray, size: Int, sampleRate: Int) {
        val validSize = minOf(size, data.size)
        if (sampleRate != TARGET_SAMPLE_RATE || validSize <= 0) return
        var sourceOffset = 0
        synchronized(pending) {
            while (sourceOffset < validSize) {
                val copied = minOf(validSize - sourceOffset, pending.size - pendingSize)
                data.copyInto(
                    pending,
                    destinationOffset = pendingSize,
                    startIndex = sourceOffset,
                    endIndex = sourceOffset + copied,
                )
                pendingSize += copied
                sourceOffset += copied
                if (pendingSize == pending.size) {
                    uploads.trySend(pending.copyOf())
                    pendingSize = 0
                }
            }
        }
    }

    fun close() {
        uploads.close()
        uploadJob.cancel()
    }
}
