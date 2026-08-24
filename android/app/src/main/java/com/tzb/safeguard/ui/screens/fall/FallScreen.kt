package com.tzb.safeguard.ui.screens.fall

import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.fall.model.CameraMonitoringStatus
import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.model.RoomFallRisk
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.BgPage
import com.tzb.safeguard.ui.theme.LineColor
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.SafeGreen
import com.tzb.safeguard.ui.theme.TextSecondary
import com.tzb.safeguard.ui.theme.WarnAmber

@Composable
fun FallScreen(
    navController: NavHostController,
    viewModel: FallViewModel = appViewModel {
        FallViewModel(ServiceLocator.repository, ServiceLocator.fallRiskRepository)
    },
) {
    val stateFlow by viewModel.state.collectAsState()
    val state = stateFlow
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner, viewModel) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> viewModel.startPolling()
                Lifecycle.Event.ON_STOP -> viewModel.stopPolling()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
            viewModel.startPolling()
        }
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            viewModel.stopPolling()
        }
    }
    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(
                Modifier.fillMaxWidth().padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = { navController.popBackStack() }) {
                    Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
                }
                Text("跌倒风险监测", style = MaterialTheme.typography.headlineLarge)
            }
            Column(
                Modifier.weight(1f).padding(16.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                when (state) {
                    UiState.Loading -> {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(10.dp))
                            Text("正在读取监测状态")
                        }
                    }
                    is UiState.Error -> {
                        AppCard {
                            Text("跌倒风险监测服务暂不可用", style = MaterialTheme.typography.bodyMedium)
                            Text(state.message, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                            TextButton(onClick = viewModel::load) { Text("重新加载") }
                        }
                    }
                    is UiState.Success -> FallRiskContent(state.data)
                }
            }
        }
    }
}

@Composable
private fun FallRiskContent(overview: FallRiskOverview) {
    overview.rooms.sortedBy(::roomDisplayOrder).forEach { room ->
        RoomRiskCard(
            room = room,
            cameraMonitoring = overview.camera_monitoring.takeIf { room.room_id == "living_room" },
        )
    }
    Text(
        "风险结果以各房间最新监测数据为准；尚未形成正式结果时显示“暂无正式判断”。",
        style = MaterialTheme.typography.bodySmall,
        color = TextSecondary,
    )
}

@Composable
private fun RoomRiskCard(
    room: RoomFallRisk,
    cameraMonitoring: CameraMonitoringStatus?,
) {
    val isCameraLedRoom = room.room_id == "living_room"
    val presentation = roomRiskPresentation(room, isCameraLedRoom)
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                if (isCameraLedRoom) Icons.Filled.Videocam else Icons.Filled.Radar,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(30.dp),
            )
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text(room.room_name, style = MaterialTheme.typography.titleMedium)
                Text(
                    if (isCameraLedRoom) {
                        decisionPathLabel(room.decision_path)
                    } else {
                        "毫米波雷达守护"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "当前风险状态：${presentation.label}",
            style = MaterialTheme.typography.titleMedium,
            color = presentation.color,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(6.dp))
        if (isCameraLedRoom) {
            Text(riskSummary(room), style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(6.dp))
            if (cameraMonitoring != null) {
                SensorStatusItem("视频监控", cameraMonitoring.camera_stream_status)
                Spacer(Modifier.height(4.dp))
                SensorStatusItem("跌倒算法", cameraMonitoring.camera_algorithm_status)
            } else {
                SensorStatusItem("摄像头", room.camera_status)
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "雷达证据：${radarEvidenceLabel(room)}",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
        } else {
            RadarOnlyStatus(room.radar_status)
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "更新时间：${formatFallUpdatedAt(room.updated_at)}",
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
    }
}

@Composable
private fun RadarOnlyStatus(status: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier.size(6.dp).background(sensorColor(status), RoundedCornerShape(3.dp)),
        )
        Spacer(Modifier.width(6.dp))
        Text(
            "设备：${radarDeviceSummary(status)}",
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
    }
}

@Composable
private fun SensorStatusItem(label: String, status: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier.size(6.dp).background(sensorColor(status), RoundedCornerShape(3.dp)),
        )
        Spacer(Modifier.width(6.dp))
        Text(
            "$label：${sensorLabel(status)}",
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
    }
}

private fun sensorColor(status: String): Color = when (status) {
    "available", "streaming", "running" -> SafeGreen
    "degraded", "starting", "connecting", "reconnecting", "waiting_data" -> WarnAmber
    "not_applicable" -> LineColor
    else -> TextSecondary
}

private fun sensorLabel(status: String): String = when (status) {
    "available" -> "监测中"
    "degraded" -> "信号一般"
    "not_applicable" -> "不适用"
    "unavailable" -> "暂不可用"
    "no_person" -> "未检测到人体"
    "streaming", "running" -> "运行中"
    "starting", "connecting" -> "启动中"
    "reconnecting" -> "重连中"
    "waiting_data" -> "等待有效画面"
    "stopped" -> "已停止"
    "error" -> "运行错误"
    else -> "未知"
}

private fun decisionPathLabel(path: String): String = when (path) {
    "camera_led_radar_evidence" -> "视觉主判断 · 雷达证据增强"
    "camera_only" -> "视觉监测"
    "radar_only" -> "雷达单模态监测"
    else -> "监测暂不可用"
}

private fun riskLevelLabel(level: String): String = when (level) {
    "critical", "high" -> "高风险"
    "medium" -> "需关注"
    "low" -> "低风险"
    "normal" -> "正常"
    else -> "无有效判断"
}

private fun riskLevelColor(level: String): Color = when (level) {
    "critical", "high" -> Color(0xFFD92D20)
    "medium" -> Color(0xFFDC6803)
    "normal", "low" -> SafeGreen
    else -> TextSecondary
}

private data class RiskPresentation(val label: String, val color: Color)

private fun riskPresentation(room: RoomFallRisk?): RiskPresentation {
    if (room == null || room.prediction_state == "no_person") {
        return RiskPresentation("无有效判断", TextSecondary)
    }
    return RiskPresentation(
        label = riskLevelLabel(room.risk_level),
        color = riskLevelColor(room.risk_level),
    )
}

private fun roomRiskPresentation(
    room: RoomFallRisk,
    isCameraLedRoom: Boolean,
): RiskPresentation {
    val hasRiskLevel = room.risk_level in setOf("normal", "low", "medium", "high", "critical")
    val hasFormalResult = hasRiskLevel &&
        room.prediction_state !in setOf("unknown", "unavailable", "no_person") &&
        (isCameraLedRoom || room.decision_path == "radar_only")
    return if (hasFormalResult) {
        riskPresentation(room)
    } else {
        RiskPresentation("暂无正式判断", TextSecondary)
    }
}

private fun riskSummary(room: RoomFallRisk?): String = when {
    room == null -> "暂未收到客厅跌倒风险结果"
    room.prediction_state == "no_person" -> "未检测到人体，暂无有效风险判断"
    room.risk_level == "unknown" || room.prediction_state == "unavailable" ->
        "当前没有足够数据形成有效判断"
    room.evidence_summary.isNotBlank() -> room.evidence_summary
    else -> "正在持续分析客厅画面"
}

private fun radarEvidenceLabel(room: RoomFallRisk): String = when (room.radar_status) {
    "available" -> if (room.decision_path == "camera_led_radar_evidence") {
        "已连接，联合证据可用"
    } else {
        "已连接，等待联合判断"
    }
    "degraded" -> "已连接，数据质量一般"
    else -> "暂不可用"
}

private fun radarDeviceSummary(status: String): String = when (status) {
    "available" -> "已连接 · 监测中 · 数据正常"
    "degraded" -> "已连接 · 监测中 · 数据质量一般"
    "not_applicable" -> "不适用"
    "unavailable" -> "暂不可用 · 暂无数据"
    else -> "状态未知"
}

private fun roomDisplayOrder(room: RoomFallRisk): Int = when (room.room_id) {
    "living_room" -> 0
    "bedroom" -> 1
    "bathroom" -> 2
    else -> 3
}

private fun formatFallUpdatedAt(raw: String?): String {
    if (raw.isNullOrBlank()) return "暂无"
    return runCatching {
        val instant = runCatching { java.time.Instant.parse(raw) }
            .getOrElse { java.time.OffsetDateTime.parse(raw).toInstant() }
        val exact = java.time.format.DateTimeFormatter.ofPattern("MM-dd HH:mm:ss")
            .withZone(java.time.ZoneId.systemDefault())
            .format(instant)
        val ageSeconds = java.time.Duration.between(instant, java.time.Instant.now()).seconds
        when (ageSeconds) {
            in -30..59 -> "刚刚 · $exact"
            in 60..3599 -> "${ageSeconds / 60}分钟前 · $exact"
            in 3600..86399 -> "${ageSeconds / 3600}小时前 · $exact"
            else -> exact
        }
    }.getOrDefault(raw)
}
