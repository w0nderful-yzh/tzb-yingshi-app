package com.tzb.safeguard.ui.screens.fall

import androidx.compose.foundation.background
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material.icons.filled.WatchLater
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
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

@Composable
fun FallScreen(
    navController: NavHostController,
    viewModel: FallViewModel = appViewModel {
        FallViewModel(ServiceLocator.repository, ServiceLocator.fallRiskRepository)
    },
) {
    val stateFlow by viewModel.state.collectAsState()
    val state = stateFlow
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
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
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
    CapabilityCard()
    Text("房间监测状态", style = MaterialTheme.typography.titleLarge)
    overview.rooms.forEach { room ->
        RoomRiskCard(room)
    }
    Text(
        "跌倒风险预测已接入雷达毫米波与摄像头多模态监测；各房间当前状态以实时监测为准，不构成医疗或急救判断。",
        style = MaterialTheme.typography.bodySmall,
        color = TextSecondary,
    )
}

@Composable
private fun CapabilityCard() {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Filled.Radar,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(34.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("多模态跌倒监测", style = MaterialTheme.typography.titleLarge)
                Text(
                    "已接入雷达毫米波 + 摄像头",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        Text("· 客厅：摄像头视觉主判断 + 雷达运动证据增强", style = MaterialTheme.typography.bodySmall)
        Text("· 卫生间：雷达单模态监测", style = MaterialTheme.typography.bodySmall)
        Text("· 卧室：雷达单模态监测", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun RoomRiskCard(room: RoomFallRisk) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.WatchLater, contentDescription = null, tint = Primary, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text(room.room_name, style = MaterialTheme.typography.titleMedium)
                Text(
                    decisionPathLabel(room.decision_path),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
            Text(
                riskLevelLabel(room.risk_level),
                color = riskLevelColor(room.risk_level),
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp,
            )
        }
        Spacer(Modifier.height(8.dp))
        Text(
            room.evidence_summary.ifBlank { "监测已接入，等待数据流" },
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SensorStatusItem("雷达", room.radar_status)
            SensorStatusItem("摄像头", room.camera_status)
        }
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
    "available" -> SafeGreen
    "not_applicable" -> LineColor
    else -> TextSecondary
}

private fun sensorLabel(status: String): String = when (status) {
    "available" -> "监测中"
    "not_applicable" -> "不适用"
    "unavailable" -> "等待数据流"
    else -> "未知"
}

private fun decisionPathLabel(path: String): String = when (path) {
    "camera_led_radar_evidence" -> "视觉主判断 · 雷达证据增强"
    "camera_only" -> "视觉监测"
    "radar_only" -> "雷达单模态监测"
    else -> "监测暂不可用"
}

private fun riskLevelLabel(level: String): String = when (level) {
    "critical" -> "紧急"
    "high" -> "高风险"
    "medium" -> "需关注"
    "low" -> "低风险"
    "normal" -> "正常"
    else -> "不可用"
}

private fun riskLevelColor(level: String): Color = when (level) {
    "critical", "high" -> Color(0xFFD92D20)
    "medium" -> Color(0xFFDC6803)
    "normal", "low" -> SafeGreen
    else -> TextSecondary
}
