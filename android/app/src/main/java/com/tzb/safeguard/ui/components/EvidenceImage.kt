package com.tzb.safeguard.ui.components

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ImageNotSupported
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

private sealed interface EvidenceImageState {
    data object Empty : EvidenceImageState
    data object Loading : EvidenceImageState
    data class Ready(val bitmap: ImageBitmap) : EvidenceImageState
    data object Failed : EvidenceImageState
}

@Composable
fun EvidenceImage(
    imageUrl: String?,
    timestamp: String = "",
    modifier: Modifier = Modifier,
) {
    val state by produceState<EvidenceImageState>(
        initialValue = if (imageUrl.isNullOrBlank()) EvidenceImageState.Empty else EvidenceImageState.Loading,
        key1 = imageUrl,
    ) {
        if (imageUrl.isNullOrBlank()) return@produceState
        value = withContext(Dispatchers.IO) {
            runCatching {
                val connection = URL(imageUrl).openConnection() as HttpURLConnection
                connection.connectTimeout = 5_000
                connection.readTimeout = 8_000
                connection.instanceFollowRedirects = true
                try {
                    connection.inputStream.use { stream ->
                        BitmapFactory.decodeStream(stream)?.asImageBitmap()
                            ?: error("invalid image")
                    }
                } finally {
                    connection.disconnect()
                }
            }.fold(
                onSuccess = { EvidenceImageState.Ready(it) },
                onFailure = { EvidenceImageState.Failed },
            )
        }
    }

    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFFF0F2F5)),
        contentAlignment = Alignment.Center,
    ) {
        when (val current = state) {
            is EvidenceImageState.Ready -> Image(
                bitmap = current.bitmap,
                contentDescription = "事件对应画面",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
            EvidenceImageState.Loading -> CircularProgressIndicator(
                color = Color(0xFF2563EB),
                strokeWidth = 2.dp,
            )
            else -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    Icons.Filled.ImageNotSupported,
                    contentDescription = null,
                    tint = Color(0xFF98A2B3),
                )
                Text("暂无事件画面", color = Color(0xFF667085), fontSize = 12.sp)
            }
        }
        if (timestamp.isNotBlank()) {
            Text(
                timestamp,
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .background(Color.Black.copy(alpha = 0.58f), RoundedCornerShape(topEnd = 7.dp))
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            )
        }
    }
}
