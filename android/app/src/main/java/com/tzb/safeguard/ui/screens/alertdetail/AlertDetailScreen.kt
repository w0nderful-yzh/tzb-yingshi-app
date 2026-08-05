package com.tzb.safeguard.ui.screens.alertdetail

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.EventDetail
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 告警详情与分级处置。对应 prototype/pages/alert-detail.html。
 * 老人端：「我没事 / 我需要帮助」；家属端：回呼 / 标记误报 / 查看画面。
 * 页面不挂底部导航（全屏处置场景），顶部提供返回。
 */
@Composable
fun AlertDetailScreen(
    navController: NavHostController,
    eventId: String,
    vm: AlertDetailViewModel = appViewModel {
        AlertDetailViewModel(ServiceLocator.repository, eventId)
    }
) {
    val state by vm.state.collectAsState()
    val action by vm.action.collectAsState()
    val isFamily = Session.role == "family"

    // 处置完成后返回上一页
    LaunchedEffect(action) {
        val a = action
        if (a is ActionState.Done && a.closePage) {
            kotlinx.coroutines.delay(600)
            navController.popBackStack()
        }
    }

    Scaffold(containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            StateBox(state = state, onRetry = vm::load, modifier = Modifier.fillMaxSize()) { detail ->
                LazyColumn(
                    contentPadding = PaddingValues(14.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    item { DetailHeader(detail, onBack = { navController.popBackStack() }) }

                    // 老人端紧急确认面板
                    if (!isFamily && detail.level == "emergency" && detail.status == "open") {
                        item {
                            SosConfirmPanel(
                                running = action is ActionState.Running,
                                onImOk = { vm.confirm("im_ok") },
                                onNeedHelp = { vm.confirm("need_help") }
                            )
                        }
                    }

                    // 现场证据
                    item { EvidenceCard(detail) }

                    // AI 判断依据（可解释）
                    if (detail.analysis.reasons.isNotEmpty()) {
                        item {
                            AppCard {
                                Text("系统判断依据", style = MaterialTheme.typography.titleLarge)
                                Spacer(Modifier.height(4.dp))
                                detail.analysis.reasons.forEach { KeyValueRow(it.label, it.value) }
                                KeyValueRow("置信度", "${detail.analysis.confidence}（辅助判断）")
                                Spacer(Modifier.height(8.dp))
                                Text(detail.analysis.disclaimer, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }

                    // 通知进度步骤条
                    if (detail.notifications.isNotEmpty()) {
                        item { NotifySteps(detail) }
                    }

                    // 家属端操作区
                    if (isFamily) {
                        item {
                            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                BigActionButton(text = "一键呼叫老人", icon = Icons.Filled.Call,
                                    onClick = { vm.callElder() })
                                BigActionButton(text = "查看实时画面", outlined = true) {
                                    navController.navigate(Routes.MONITOR) { launchSingleTop = true }
                                }
                                if (detail.status == "open") {
                                    BigActionButton(text = "标记为误报", icon = Icons.Filled.Flag,
                                        containerColor = Color(0xFFEEF1F5), contentColor = TextMain,
                                        onClick = { vm.patchStatus("false_alarm") })
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 动作反馈
    when (val a = action) {
        is ActionState.Done -> AlertDialog(
            onDismissRequest = vm::resetAction,
            title = { Text("操作成功") },
            text = { Text(a.message) },
            confirmButton = { TextButton(onClick = vm::resetAction) { Text("知道了") } }
        )
        is ActionState.Failed -> AlertDialog(
            onDismissRequest = vm::resetAction,
            title = { Text("操作失败") },
            text = { Text(a.message) },
            confirmButton = { TextButton(onClick = vm::resetAction) { Text("知道了") } }
        )
        else -> Unit
    }
}

@Composable
private fun DetailHeader(detail: EventDetail, onBack: () -> Unit) {
    val bg = levelBgColor(detail.level)
    val fg = levelColor(detail.level)
    AppCard(containerColor = bg) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(Icons.Filled.ArrowBack, contentDescription = "返回", tint = fg)
            }
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    LevelBadge(detail.level)
                    Spacer(Modifier.width(8.dp))
                    Text(eventTypeMeta(detail.type).first, color = fg, fontWeight = FontWeight.Bold, fontSize = 20.sp)
                }
                Text(
                    "${formatTime(detail.occurred_at)} · ${detail.device_id}",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

/** 老人端紧急确认面板：倒计时提示 + 双大按钮 */
@Composable
private fun SosConfirmPanel(running: Boolean, onImOk: () -> Unit, onNeedHelp: () -> Unit) {
    AppCard(containerColor = WarnRedBg) {
        Text("我没事 / 我需要帮助", fontSize = 22.sp, fontWeight = FontWeight.ExtraBold,
            color = WarnRed, modifier = Modifier.fillMaxWidth(),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        Text("若 60 秒内无人确认，将自动呼叫紧急联系人",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.fillMaxWidth(),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        Spacer(Modifier.height(14.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.weight(1f)) {
                BigActionButton(
                    text = if (running) "提交中…" else "我没事",
                    icon = Icons.Filled.Check,
                    containerColor = SafeGreen,
                    onClick = { if (!running) onImOk() }
                )
            }
            Box(Modifier.weight(1f)) {
                BigActionButton(
                    text = if (running) "提交中…" else "呼叫帮助",
                    icon = Icons.Filled.Call,
                    containerColor = WarnRed,
                    onClick = { if (!running) onNeedHelp() }
                )
            }
        }
    }
}

/** 现场证据：脱敏截图占位 */
@Composable
private fun EvidenceCard(detail: EventDetail) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(16f / 9f)
            .background(Color(0xFF101318), RoundedCornerShape(14.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Filled.Image, contentDescription = null, tint = Color(0xFF9AA3AF), modifier = Modifier.size(44.dp))
            Text(
                if (detail.evidence_image_url != null) "事件截图（加载中）" else "事件截图证据（脱敏后展示）",
                color = Color(0xFF9AA3AF), fontSize = 15.sp
            )
        }
    }
    Text("依据隐私规范，仅保留事件瞬间截图，不保存连续录像。",
        style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 6.dp))
}

/** 通知进度步骤条 */
@Composable
private fun NotifySteps(detail: EventDetail) {
    AppCard {
        Text("通知进度", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(10.dp))
        val steps = buildList {
            add(Step("告警生成", formatTime(detail.occurred_at), done = true))
            detail.notifications.forEach {
                add(Step("已通知 ${it.target}", "渠道：${it.channel}", done = true))
            }
            add(Step(
                if (detail.escalation.status == "pending") "等待确认" else "外呼升级",
                if (detail.escalation.status == "pending") "60 秒无响应将自动拨打紧急联系人" else "已发起外呼",
                done = detail.escalation.status != "pending"
            ))
        }
        steps.forEachIndexed { i, step ->
            Row {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(
                        Modifier.size(26.dp).background(
                            if (step.done) SafeGreen else Color(0xFFC4C9D2), CircleShape
                        ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("${i + 1}", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                    if (i < steps.size - 1) {
                        Box(Modifier.width(3.dp).height(34.dp).background(LineColor))
                    }
                }
                Spacer(Modifier.width(12.dp))
                Column {
                    Text(step.title, style = MaterialTheme.typography.titleMedium)
                    Text(step.desc, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

private data class Step(val title: String, val desc: String, val done: Boolean)
