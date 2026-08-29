package com.tzb.safeguard.ui.screens.alertdetail

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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.NotificationsActive
import androidx.compose.material.icons.filled.PersonSearch
import androidx.compose.material.icons.filled.PhoneInTalk
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.model.EventDetail
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.components.BigActionButton
import com.tzb.safeguard.ui.components.EvidenceImage
import com.tzb.safeguard.ui.components.StateBox
import com.tzb.safeguard.ui.components.eventStatusLabel
import com.tzb.safeguard.ui.components.formatTime
import com.tzb.safeguard.ui.components.levelColor
import com.tzb.safeguard.ui.components.verificationStatusLabel
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.LineColor
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.TextMain
import com.tzb.safeguard.ui.theme.TextSecondary

@Composable
fun AlertDetailScreen(
    navController: NavHostController,
    eventId: String,
    vm: AlertDetailViewModel = appViewModel { AlertDetailViewModel(ServiceLocator.repository, eventId) },
) {
    val state by vm.state.collectAsState()
    val action by vm.action.collectAsState()

    LaunchedEffect(action) {
        val result = action
        if (result is ActionState.Done && result.closePage) {
            kotlinx.coroutines.delay(500)
            navController.popBackStack()
        }
    }

    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = Color.White) { padding ->
        StateBox(state, vm::load, Modifier.padding(padding)) { detail ->
            var selectedFrame by remember(detail.event_id) {
                mutableIntStateOf(detail.evidence_frames.lastIndex.coerceAtLeast(0))
            }
            val frame = detail.evidence_frames.getOrNull(selectedFrame)
            val heroImageUrl = frame?.image_url ?: detail.evidence_image_url
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                item { DetailTopBar { navController.popBackStack() } }
                item { EventHero(detail) }
                if (heroImageUrl != null) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1.55f)
                                .clickable { navController.navigate(Routes.MONITOR) },
                        ) {
                            EvidenceImage(
                                imageUrl = heroImageUrl,
                                timestamp = formatTime(frame?.captured_at ?: detail.occurred_at).removePrefix("今天 "),
                                modifier = Modifier.fillMaxSize(),
                            )
                            Surface(
                                shape = CircleShape,
                                color = Color.Black.copy(alpha = 0.55f),
                                modifier = Modifier.align(Alignment.Center),
                            ) {
                                Icon(
                                    Icons.Filled.PlayArrow,
                                    contentDescription = "查看实时画面",
                                    tint = Color.White,
                                    modifier = Modifier.padding(12.dp).size(28.dp),
                                )
                            }
                        }
                    }
                }
                item { PredictionReasonCard(detail) }
                item { BasicInfoCard(detail) }
                if (detail.evidence_frames.isNotEmpty()) {
                    item {
                        AppCard {
                            Text("关联画面", fontWeight = FontWeight.Bold, fontSize = 17.sp)
                            Spacer(Modifier.height(9.dp))
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                itemsIndexed(detail.evidence_frames) { index, evidence ->
                                    Surface(
                                        modifier = Modifier
                                            .width(112.dp)
                                            .clickable { selectedFrame = index },
                                        color = Color.Transparent,
                                    ) {
                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                            Surface(
                                                shape = RoundedCornerShape(9.dp),
                                                border = BorderStroke(
                                                    if (selectedFrame == index) 2.dp else 1.dp,
                                                    if (selectedFrame == index) Primary else LineColor,
                                                ),
                                            ) {
                                                EvidenceImage(
                                                    imageUrl = evidence.image_url,
                                                    modifier = Modifier.fillMaxWidth().height(62.dp),
                                                )
                                            }
                                            Spacer(Modifier.height(4.dp))
                                            Text(
                                                formatTime(evidence.captured_at).removePrefix("今天 "),
                                                color = if (selectedFrame == index) Primary else TextSecondary,
                                                fontSize = 12.sp,
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                item {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (detail.type == "fall_suspected") {
                            if (detail.status == "open") {
                                BigActionButton("收到预警，立即查看", icon = Icons.Filled.PersonSearch) {
                                    vm.patchStatus("acknowledged")
                                }
                            }
                            if (detail.status == "open" || detail.status == "acknowledged") {
                                BigActionButton(
                                    "已确认老人安全",
                                    icon = Icons.Filled.CheckCircle,
                                    outlined = true,
                                ) { vm.patchStatus("resolved") }
                                TextButton(
                                    onClick = { vm.patchStatus("false_alarm") },
                                    modifier = Modifier.fillMaxWidth(),
                                ) { Text("误报，并未跌倒", color = TextSecondary) }
                            }
                        } else {
                            if (detail.status == "open") {
                                BigActionButton("立即介入", icon = Icons.Filled.PhoneInTalk) {
                                    vm.patchStatus("acknowledged")
                                }
                            }
                            if (detail.status == "open" || detail.status == "acknowledged") {
                                BigActionButton(
                                    "设备语音提醒（待接入）",
                                    icon = Icons.Filled.NotificationsActive,
                                    outlined = true,
                                ) { vm.sendInterventionReminder() }
                                BigActionButton(
                                    "风险已核实并解除",
                                    icon = Icons.Filled.CheckCircle,
                                    outlined = true,
                                ) { vm.patchStatus("resolved") }
                                TextButton(
                                    onClick = { vm.patchStatus("false_alarm") },
                                    modifier = Modifier.fillMaxWidth(),
                                ) { Text("标记为误报", color = TextSecondary) }
                            }
                        }
                    }
                }
            }
        }
    }

    when (val result = action) {
        is ActionState.Done -> AlertDialog(
            onDismissRequest = vm::resetAction,
            title = { Text("操作成功") },
            text = { Text(result.message) },
            confirmButton = { TextButton(onClick = vm::resetAction) { Text("知道了") } },
        )
        is ActionState.Failed -> AlertDialog(
            onDismissRequest = vm::resetAction,
            title = { Text("功能暂不可用") },
            text = { Text(result.message) },
            confirmButton = { TextButton(onClick = vm::resetAction) { Text("知道了") } },
        )
        else -> Unit
    }
}

@Composable
private fun DetailTopBar(onBack: () -> Unit) {
    Box(Modifier.fillMaxWidth().height(48.dp)) {
        IconButton(onClick = onBack, modifier = Modifier.align(Alignment.CenterStart)) {
            Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
        }
        Text(
            "预警详情",
            fontWeight = FontWeight.Bold,
            fontSize = 18.sp,
            modifier = Modifier.align(Alignment.Center),
        )
    }
}

@Composable
private fun EventHero(detail: EventDetail) {
    val title = if (detail.type == "fall_suspected") {
        if (detail.level == "emergency") "疑似跌倒预警" else "跌倒高风险预警"
    } else {
        when (detail.fraud?.scene) {
            "home_visit" -> "入户诈骗风险预警"
            "telecom" -> "电信诈骗风险预警"
            else -> "诈骗风险预警"
        }
    }
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Surface(shape = CircleShape, color = levelColor(detail.level)) {
                Icon(
                    Icons.Filled.Shield,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.padding(8.dp).size(20.dp),
                )
            }
            Spacer(Modifier.width(10.dp))
            Column {
                Text(title, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                Text(
                    "${formatTime(detail.occurred_at)} · ${eventStatusLabel(detail.status)}"
                        + detail.verification_status?.let {
                            " · ${verificationStatusLabel(detail.verification_status)}"
                        }.orEmpty(),
                    color = TextSecondary,
                    fontSize = 13.sp,
                )
            }
        }
    }
}

@Composable
private fun PredictionReasonCard(detail: EventDetail) {
    val isFall = detail.type == "fall_suspected"
    AppCard {
        Text(if (isFall) "事件说明" else "预测原因", fontWeight = FontWeight.Bold, fontSize = 17.sp)
        Spacer(Modifier.height(7.dp))
        if (isFall) {
            Text(
                when (detail.level) {
                    "emergency" -> "检测到疑似跌倒，请立即确认老人状态。"
                    else -> "摄像头多模态判断为跌倒高风险，请立即关注老人状态。"
                },
                color = TextMain,
                fontSize = 14.sp,
                lineHeight = 22.sp,
            )
        } else {
            Text(
                detail.fraud?.transition_reason?.ifBlank { null }
                    ?: detail.analysis.reasons.firstOrNull()?.value
                    ?: "多模态证据显示诈骗风险正在上升，建议家属尽快关注。",
                color = TextMain,
                fontSize = 14.sp,
                lineHeight = 22.sp,
            )
            detail.fraud?.let {
                Spacer(Modifier.height(8.dp))
                Text(
                    "当前阶段 S${it.state_index} · ${it.state_label}｜建议：${decisionLabel(it.decision)}",
                    color = levelColor(detail.level),
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}

@Composable
private fun BasicInfoCard(detail: EventDetail) {
    AppCard {
        Text("基本信息", fontWeight = FontWeight.Bold, fontSize = 17.sp)
        Spacer(Modifier.height(7.dp))
        InfoRow("时间", formatTime(detail.occurred_at))
        InfoRow(
            "位置",
            detail.location.ifBlank {
                if (detail.device_id.isBlank()) "家中摄像头" else "摄像头 ${detail.device_id}"
            },
        )
        InfoRow(
            "建议",
            if (detail.type == "fall_suspected") {
                if (detail.level == "emergency") "立即联系或前往查看老人，确认是否需要帮助"
                else "尽快确认老人当前状态，留意是否有受伤"
            } else {
                decisionLabel(detail.fraud?.decision.orEmpty())
            },
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.Top) {
        Text(label, color = TextSecondary, fontSize = 14.sp, modifier = Modifier.width(48.dp))
        Text(value, color = TextMain, fontSize = 14.sp, modifier = Modifier.weight(1f))
    }
}

private fun decisionLabel(decision: String): String = when (decision) {
    "verify" -> "尽快联系家人，协助核实对方身份"
    "warn" -> "立即提醒家人停止透露敏感信息"
    "block" -> "立即介入，阻止转账、验证码或远程授权"
    "intervene" -> "高风险趋势，家属应立即联系并现场核实"
    else -> "继续观察风险变化，必要时联系家人"
}
