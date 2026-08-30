package com.tzb.safeguard.ui.screens.fraud

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForwardIos
import androidx.compose.material.icons.filled.Shield
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.media.Ys7MonitorService
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.components.EmptyBox
import com.tzb.safeguard.ui.components.StateBox
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.components.formatTime
import com.tzb.safeguard.ui.components.levelColor
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.BgPage
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.SafeGreen
import com.tzb.safeguard.ui.theme.TextSecondary

@Composable
fun FraudScreen(
    navController: NavHostController,
    viewModel: FraudViewModel = appViewModel {
        FraudViewModel(ServiceLocator.repository)
    },
) {
    val state by viewModel.state.collectAsState()
    val guardStatus by Ys7MonitorService.status.collectAsState()
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_START) viewModel.load()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(
                Modifier.fillMaxWidth().padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = { navController.popBackStack() }) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回")
                }
                Text("诈骗风险防护", style = MaterialTheme.typography.headlineLarge)
            }
            Column(
                Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                MonitoringCard(guardStatus.fraudMonitoringStatus)
                StateBox(state, viewModel::load) { events ->
                    CurrentRiskCard(events)
                    if (events.isEmpty()) {
                        EmptyBox("暂无诈骗预警记录")
                    } else {
                        AppCard {
                            Text("最近预警记录", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.size(6.dp))
                            events.take(3).forEach { event ->
                                FraudEventRow(event) {
                                    navController.navigate(Routes.alertDetail(event.event_id))
                                }
                            }
                        }
                    }
                    TextButton(
                        onClick = { navController.navigate(Routes.ALERTS) },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("查看全部消息", color = Primary) }
                }
            }
        }
    }
}

@Composable
private fun MonitoringCard(status: String) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Shield, contentDescription = null, tint = Primary)
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("实时语音监测", style = MaterialTheme.typography.titleLarge)
                Text(
                    "基于对话语音实时识别",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        Spacer(Modifier.size(8.dp))
        DataRow("当前状态", monitoringStatusLabel(status))
        DataRow("数据来源", "摄像头环境音（16 kHz PCM）")
    }
}

@Composable
private fun CurrentRiskCard(events: List<RiskEvent>) {
    val latestOpen = events.firstOrNull { it.status == "open" || it.status == "acknowledged" }
    AppCard {
        Text("当前风险", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.size(8.dp))
        if (latestOpen == null) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(7.dp).background(SafeGreen, CircleShape))
                Spacer(Modifier.width(8.dp))
                Text("未发现诈骗风险，持续监测中", fontWeight = FontWeight.Medium)
            }
        } else {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(7.dp).background(levelColor(latestOpen.level), CircleShape))
                Spacer(Modifier.width(8.dp))
                Text(
                    latestOpen.fraud_state_label?.takeIf { it.isNotBlank() }
                        ?: latestOpen.title.ifBlank { "诈骗风险预警" },
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Spacer(Modifier.size(4.dp))
            Text(
                "${formatTime(latestOpen.occurred_at)} · 处置状态：${disposalLabel(latestOpen.status)}",
                color = TextSecondary,
                fontSize = 12.sp,
            )
            Text(latestOpen.summary, color = TextSecondary, fontSize = 13.sp, maxLines = 2)
        }
    }
}

@Composable
private fun FraudEventRow(event: RiskEvent, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(8.dp).background(levelColor(event.level), CircleShape))
        Spacer(Modifier.width(9.dp))
        Column(Modifier.weight(1f)) {
            Text(
                event.fraud_state_label?.takeIf { it.isNotBlank() }
                    ?: event.title.ifBlank { "诈骗风险预警" },
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(formatTime(event.occurred_at), color = TextSecondary, fontSize = 12.sp)
        }
        Icon(
            Icons.AutoMirrored.Filled.ArrowForwardIos,
            contentDescription = null,
            tint = TextSecondary,
            modifier = Modifier.size(12.dp),
        )
    }
}

@Composable
private fun DataRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(label, color = TextSecondary, fontSize = 13.sp)
        Spacer(Modifier.weight(1f))
        Text(value, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    }
}

private fun monitoringStatusLabel(status: String): String = when (status) {
    "running" -> "运行中"
    "starting" -> "启动中"
    "unavailable" -> "未启用"
    "stopped" -> "未开启守护"
    else -> "不可用"
}

private fun disposalLabel(status: String): String = when (status) {
    "open" -> "待处置"
    "acknowledged" -> "处置中"
    "resolved" -> "已完成"
    "false_alarm" -> "已标记误报"
    else -> status
}
