package com.tzb.safeguard.data.realtime

import android.util.Log
import com.tzb.safeguard.BuildConfig
import com.tzb.safeguard.data.model.RealtimeEnvelope
import com.tzb.safeguard.data.model.RealtimeRiskEvent
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.Json
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

class AlertWebSocketClient(
    private val client: OkHttpClient,
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun listen(
        ticket: String,
        onConnected: () -> Unit,
        onEvent: (RealtimeRiskEvent) -> Unit,
    ) {
        val base = BuildConfig.API_BASE_URL.toHttpUrl()
        val httpUrl = checkNotNull(base.resolve("api/v1/ws/events"))
        // OkHttp 的 WebSocket API 接收 http/https URL，并在内部完成 Upgrade；
        // HttpUrl 本身不允许 ws/wss scheme。
        val url = httpUrl.newBuilder()
            .addQueryParameter("ticket", ticket)
            .build()
        val closed = CompletableDeferred<Throwable?>()
        val request = Request.Builder().url(url).build()
        Log.i(TAG, "connecting to ${url.redact()}")
        val socket = client.newWebSocket(
            request,
            object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    Log.i(TAG, "connected")
                    onConnected()
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    runCatching { json.decodeFromString<RealtimeEnvelope>(text) }
                        .getOrNull()
                        ?.takeIf { it.type == "risk_event.upserted" }
                        ?.event
                        ?.let(onEvent)
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    Log.i(TAG, "closed code=$code reason=$reason")
                    closed.complete(null)
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: Response?,
                ) {
                    Log.e(TAG, "connection failed: ${response?.code}", t)
                    closed.complete(t)
                }
            },
        )
        try {
            closed.await()?.let { throw it }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } finally {
            socket.cancel()
        }
    }

    private companion object {
        const val TAG = "AlertWebSocket"
    }
}
