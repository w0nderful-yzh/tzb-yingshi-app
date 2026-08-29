package com.tzb.safeguard.ui.screens.home

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForwardIos
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Elderly
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.filled.WatchLater
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.model.RoomFallRisk
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import com.tzb.safeguard.ui.components.AppBottomBar
import com.tzb.safeguard.ui.components.AppTabs
import com.tzb.safeguard.ui.components.StateBox
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.components.formatTime
import com.tzb.safeguard.ui.components.levelColor
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.screens.monitor.LiveVideoPlayer
import com.tzb.safeguard.ui.theme.LineColor
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.SafeGreen
import com.tzb.safeguard.ui.theme.TextMain
import com.tzb.safeguard.ui.theme.TextSecondary

private val PredictionCardBackground = Color(0xFFF8FAFF)

@Composable
fun HomeScreen(
    navController: NavHostController,
    vm: HomeViewModel = appViewModel {
        HomeViewModel(
            ServiceLocator.repository,
            ServiceLocator.fallRiskRepository,
            ServiceLocator.psychologyRepository,
        )
    },
) {
    val state by vm.state.collectAsState()
    val notice by vm.notice.collectAsState()
    Scaffold(
        modifier = Modifier.statusBarsPadding(),
        bottomBar = {
            AppBottomBar(AppTabs, Routes.HOME) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = Color.White,
    ) { padding ->
        StateBox(state, vm::load, Modifier.padding(padding)) { data ->
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                item {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("家人看护", style = MaterialTheme.typography.headlineLarge)
                        Spacer(Modifier.weight(1f))
                        IconButton(onClick = { navController.navigate(Routes.PROFILE) }) {
                            Icon(Icons.Filled.Add, contentDescription = "添加家人", modifier = Modifier.size(30.dp))
                        }
                    }
                }
                item {
                    FamilyLiveCard(data = data, onRetry = vm::retryLive)
                }
                item {
                    QuickActions(
                        onLive = { navController.navigate(Routes.MONITOR) },
                        onHistory = vm::requestHistoryPlayback,
                        onIntervene = { navController.navigate(Routes.ALERTS) },
                        onSettings = { navController.navigate(Routes.PROFILE) },
                    )
                }
                item {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("今日预测", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.weight(1f))
                        TextButton(onClick = { navController.navigate(Routes.ALERTS) }) {
                            Text("全部", color = TextSecondary)
                            Spacer(Modifier.width(4.dp))
                            Icon(
                                Icons.AutoMirrored.Filled.ArrowForwardIos,
                                contentDescription = null,
                                tint = TextSecondary,
                                modifier = Modifier.size(12.dp),
                            )
                        }
                    }
                }
                item {
                    FraudRiskSummaryCard(
                        pending = data.pendingWarnings,
                        recent = data.recentWarnings,
                    ) {
                        navController.navigate(Routes.ALERTS) { launchSingleTop = true }
                    }
                }
                item {
                    FallRiskSummaryCard(data.fallRisk) {
                        navController.navigate(Routes.FALL) { launchSingleTop = true }
                    }
                }
                item {
                    PsychologySummaryCard(data.psychology) {
                        navController.navigate(Routes.CARE) { launchSingleTop = true }
                    }
                }
                item { Spacer(Modifier.height(4.dp)) }
            }
        }
    }

    notice?.let {
        AlertDialog(
            onDismissRequest = vm::clearNotice,
            title = { Text("功能提示") },
            text = { Text(it) },
            confirmButton = { TextButton(onClick = vm::clearNotice) { Text("知道了") } },
        )
    }
}

@Composable
private fun FamilyLiveCard(data: HomeData, onRetry: () -> Unit) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = Color.White,
        border = BorderStroke(1.dp, LineColor),
        shadowElevation = 2.dp,
    ) {
        Column(Modifier.padding(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(shape = CircleShape, color = Color(0xFFFFE8D8)) {
                    Icon(
                        Icons.Filled.Elderly,
                        contentDescription = null,
                        tint = Color(0xFF6B4C3B),
                        modifier = Modifier.padding(8.dp).size(25.dp),
                    )
                }
                Spacer(Modifier.width(9.dp))
                Column(Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(data.elder?.name ?: "被守护家人", fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Spacer(Modifier.width(8.dp))
                        Surface(shape = RoundedCornerShape(6.dp), color = Color(0xFFE7F8EE)) {
                            Text(
                                if (data.selectedDevice?.online == true) "● 在线" else "● 离线",
                                color = if (data.selectedDevice?.online == true) SafeGreen else TextSecondary,
                                fontSize = 11.sp,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                            )
                        }
                    }
                    Text(
                        data.selectedDevice?.let { displayRoom(it.room.ifBlank { it.name }) }
                            ?: "尚未绑定设备",
                        color = TextSecondary,
                        fontSize = 13.sp,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            LiveVideoPlayer(
                liveSession = data.liveSession,
                deviceName = data.selectedDevice?.name ?: "摄像头",
                streamLoading = data.streamLoading,
                streamError = data.streamError,
                onRetry = onRetry,
                modifier = Modifier.fillMaxWidth().aspectRatio(1.65f),
            )
        }
    }
}

@Composable
private fun QuickActions(
    onLive: () -> Unit,
    onHistory: () -> Unit,
    onIntervene: () -> Unit,
    onSettings: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = Color.White,
        border = BorderStroke(1.dp, LineColor),
    ) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 7.dp, vertical = 12.dp)) {
            QuickAction(Icons.Filled.Videocam, "查看直播", Color(0xFFE1F8EC), Color(0xFF16A05D), Modifier.weight(1f), onLive)
            QuickAction(Icons.Filled.History, "历史回看", Color(0xFFE8F0FF), Primary, Modifier.weight(1f), onHistory)
            QuickAction(Icons.Filled.NotificationsActive, "介入提醒", Color(0xFFF0E9FF), Color(0xFF7653D6), Modifier.weight(1f), onIntervene)
            QuickAction(Icons.Filled.Settings, "更多设置", Color(0xFFF0F2F5), Color(0xFF667085), Modifier.weight(1f), onSettings)
        }
    }
}

@Composable
private fun QuickAction(
    icon: ImageVector,
    label: String,
    iconBg: Color,
    iconColor: Color,
    modifier: Modifier,
    onClick: () -> Unit,
) {
    Column(
        modifier = modifier.clickable(onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Surface(shape = RoundedCornerShape(12.dp), color = iconBg) {
            Icon(icon, contentDescription = label, tint = iconColor, modifier = Modifier.padding(9.dp).size(22.dp))
        }
        Spacer(Modifier.height(6.dp))
        Text(label, fontSize = 12.sp, color = TextMain, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun FraudRiskSummaryCard(
    pending: List<RiskEvent>,
    recent: List<RiskEvent>,
    onClick: () -> Unit,
) {
    val statusText = when {
        pending.isNotEmpty() -> "${pending.size} 条待处置"
        recent.isEmpty() -> "持续守护中"
        else -> "均已处置"
    }
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = PredictionCardBackground,
        border = BorderStroke(1.dp, LineColor),
        modifier = Modifier.clickable(onClick = onClick),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Shield, contentDescription = null, tint = Primary)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text("诈骗风险防护", fontWeight = FontWeight.SemiBold)
                    Text("基于对话语音实时识别", color = TextSecondary, fontSize = 12.sp)
                }
                Text(statusText, color = TextSecondary, fontSize = 12.sp)
                Spacer(Modifier.width(4.dp))
                Icon(
                    Icons.AutoMirrored.Filled.ArrowForwardIos,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(12.dp),
                )
            }
            recent.firstOrNull()?.let { event ->
                Spacer(Modifier.height(9.dp))
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Surface(
                        shape = CircleShape,
                        color = levelColor(event.level),
                        modifier = Modifier.size(8.dp),
                    ) {}
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.weight(1f)) {
                        Text(
                            event.fraud_state_label ?: "诈骗风险预警",
                            fontWeight = FontWeight.Medium,
                            fontSize = 13.sp,
                            maxLines = 1,
                        )
                        Text(
                            "${formatTime(event.occurred_at)} · ${disposalLabel(event.status)}",
                            color = TextSecondary,
                            fontSize = 12.sp,
                        )
                    }
                }
                if (recent.size > 1) {
                    Spacer(Modifier.height(4.dp))
                    Text("已有 ${recent.size} 条记录，点击查看全部", color = TextSecondary, fontSize = 12.sp)
                }
            }
        }
    }
}

private fun disposalLabel(status: String): String = when (status) {
    "open" -> "待处置"
    "acknowledged" -> "处置中"
    "resolved" -> "已完成"
    "false_alarm" -> "已标记误报"
    else -> status
}

@Composable
private fun FallRiskSummaryCard(overview: FallRiskOverview?, onClick: () -> Unit) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = PredictionCardBackground,
        border = BorderStroke(1.dp, LineColor),
        modifier = Modifier.clickable(onClick = onClick),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.WatchLater, contentDescription = null, tint = Primary)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text("跌倒风险预测", fontWeight = FontWeight.SemiBold)
                    Text(
                        "雷达+摄像头多模态监测",
                        color = TextSecondary,
                        fontSize = 12.sp,
                    )
                }
                Text(
                    overview?.let { fallOverallLabel(it.overall_risk_level) }
                        ?: "服务暂不可用",
                    color = TextSecondary,
                    fontSize = 12.sp,
                )
                Spacer(Modifier.width(4.dp))
                Icon(
                    Icons.AutoMirrored.Filled.ArrowForwardIos,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(12.dp),
                )
            }
            overview?.rooms?.forEach { room ->
                Spacer(Modifier.height(9.dp))
                FallRiskRoomRow(room)
            }
        }
    }
}

@Composable
private fun FallRiskRoomRow(room: RoomFallRisk) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(room.room_name, fontWeight = FontWeight.Medium, fontSize = 13.sp)
        Spacer(Modifier.weight(1f))
        Text(
            fallRiskLevelLabel(room.risk_level),
            color = fallRiskLevelColor(room.risk_level),
            fontWeight = FontWeight.SemiBold,
            fontSize = 12.sp,
        )
    }
}

private fun fallOverallLabel(level: String): String = when (level) {
    "critical", "high" -> "存在需要关注的跌倒风险"
    "medium" -> "发现风险变化，正在持续监测"
    "normal", "low" -> "各房间正在稳定监测"
    else -> "服务暂不可用"
}

private fun fallRiskLevelLabel(level: String): String = when (level) {
    "critical" -> "紧急"
    "high" -> "高风险"
    "medium" -> "需关注"
    "low" -> "低风险"
    "normal" -> "正常"
    else -> "不可用"
}

private fun fallRiskLevelColor(level: String): Color = when (level) {
    "critical", "high" -> Color(0xFFD92D20)
    "medium" -> Color(0xFFDC6803)
    "normal", "low" -> SafeGreen
    else -> TextSecondary
}

@Composable
private fun PsychologySummaryCard(state: UiState<PsychologyOverview>, onClick: () -> Unit) {
    val status = when (state) {
        UiState.Loading -> "读取中"
        is UiState.Error -> "服务暂不可用"
        is UiState.Success -> when (state.data.assessment_state) {
            "collecting" -> "正在采集资料"
            "observation_available" -> "最近评估已完成"
            "insufficient_data" -> "数据不足"
            else -> "服务暂不可用"
        }
    }
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = PredictionCardBackground,
        border = BorderStroke(1.dp, LineColor),
        modifier = Modifier.clickable(onClick = onClick),
    ) {
        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Psychology, contentDescription = null, tint = Primary)
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("心理健康评估", fontWeight = FontWeight.SemiBold)
                Text("基于摄像头面部行为特征", color = TextSecondary, fontSize = 12.sp)
            }
            Text(status, color = TextSecondary, fontSize = 12.sp)
            Spacer(Modifier.width(4.dp))
            Icon(
                Icons.AutoMirrored.Filled.ArrowForwardIos,
                contentDescription = null,
                tint = TextSecondary,
                modifier = Modifier.size(12.dp),
            )
        }
    }
}

private fun displayRoom(room: String): String = when (room.lowercase()) {
    "living_room" -> "客厅"
    "bedroom" -> "卧室"
    "kitchen" -> "厨房"
    "study" -> "书房"
    else -> room.replace('_', ' ')
}
