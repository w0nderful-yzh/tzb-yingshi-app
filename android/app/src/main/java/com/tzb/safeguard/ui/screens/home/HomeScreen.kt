package com.tzb.safeguard.ui.screens.home

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 首页 · 安全状态总览（老人端）。
 * 对应 prototype/pages/home.html：
 * 状态大卡 / SOS 大按钮 / 今日守护 / 设备状态 / 最近提醒 / 语音助手入口。
 */
@Composable
fun HomeScreen(
    navController: NavHostController,
    vm: HomeViewModel = appViewModel { HomeViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    val sosState by vm.sosState.collectAsState()
    var showSosConfirm by remember { mutableStateOf(false) }

    Scaffold(
        bottomBar = {
            AppBottomBar(ElderTabs, Routes.HOME) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            StateBox(state = state, onRetry = vm::load) { data ->
                LazyColumn(
                    contentPadding = PaddingValues(14.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    item {
                        PageHeaderBlock(
                            title = "您好，${data.user.name.ifBlank { "家人" }}",
                            subtitle = "家中安全守护中"
                        )
                    }

                    // 安全状态大卡：随 overall 变色
                    item { SafetyStatusCard(data.status.overall, data.status.overall_label,
                        "${data.status.devices_online} 台设备在线 · 待处理 ${data.status.active_event_count} 条") }

                    // SOS：先弹确认，防止误触
                    item {
                        BigActionButton(
                            text = if (sosState is SosState.Sending) "正在通知家人…" else "紧急求助",
                            icon = Icons.Filled.Sos,
                            containerColor = WarnRed,
                            onClick = { if (sosState !is SosState.Sending) showSosConfirm = true }
                        )
                        Text(
                            "点击后将同时呼叫子女并发送当前状态",
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }

                    // 今日守护
                    item {
                        AppCard {
                            Text("今日守护", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(10.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                StatBox("${data.status.today.event_count}", "风险事件",
                                    if (data.status.today.event_count > 0) WarnOrange else SafeGreen, Modifier.weight(1f))
                                StatBox("${data.status.today.active_hours}h", "活动时长", Primary, Modifier.weight(1f))
                            }
                            Spacer(Modifier.height(12.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                StatBox("${data.status.today.call_screened}", "来电已识别", Primary, Modifier.weight(1f))
                                StatBox("${data.status.devices_online}/${data.status.devices_total}", "设备在线", SafeGreen, Modifier.weight(1f))
                            }
                        }
                    }

                    // 设备状态
                    item {
                        AppCard {
                            Text("设备状态", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(6.dp))
                            data.devices.forEach { DeviceRow(it) }
                        }
                    }

                    // 最近提醒
                    item {
                        AppCard {
                            Text("最近提醒", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(10.dp))
                            if (data.recentEvents.isEmpty()) {
                                EmptyBox("暂无未处理的消息")
                            } else {
                                data.recentEvents.forEach { event ->
                                    AlertCard(event) {
                                        navController.navigate(Routes.alertDetail(event.event_id))
                                    }
                                    Spacer(Modifier.height(10.dp))
                                }
                            }
                            BigActionButton(text = "查看全部消息", outlined = true) {
                                navController.navigate(Routes.ALERTS) { launchSingleTop = true }
                            }
                        }
                    }

                    // 语音助手入口（适老语音辅助）
                    item {
                        AppCard {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Filled.Mic, contentDescription = null, tint = Primary, modifier = Modifier.size(36.dp))
                                Spacer(Modifier.width(12.dp))
                                Column {
                                    Text("语音助手已开启", style = MaterialTheme.typography.titleMedium)
                                    Text("说“小安小安”可语音播报状态、读消息", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // SOS 确认弹窗
    if (showSosConfirm) {
        AlertDialog(
            onDismissRequest = { showSosConfirm = false },
            title = { Text("发起紧急求助？") },
            text = { Text("将立即通知您的 2 位紧急联系人。如果是误触，请点取消。") },
            confirmButton = {
                TextButton(onClick = { showSosConfirm = false; vm.sendSos() }) {
                    Text("确认求助", color = WarnRed, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = { TextButton(onClick = { showSosConfirm = false }) { Text("取消") } }
        )
    }

    // SOS 结果反馈
    when (val s = sosState) {
        is SosState.Sent -> AlertDialog(
            onDismissRequest = vm::resetSos,
            title = { Text("求助已发出") },
            text = { Text("已通知 ${s.notifiedContacts} 位紧急联系人，请保持电话畅通。") },
            confirmButton = { TextButton(onClick = vm::resetSos) { Text("知道了") } }
        )
        is SosState.Failed -> AlertDialog(
            onDismissRequest = vm::resetSos,
            title = { Text("求助发送失败") },
            text = { Text(s.message + "\n请直接拨打家人电话。") },
            confirmButton = { TextButton(onClick = { vm.resetSos(); vm.sendSos() }) { Text("重试") } },
            dismissButton = { TextButton(onClick = vm::resetSos) { Text("关闭") } }
        )
        else -> Unit
    }
}

@Composable
private fun PageHeaderBlock(title: String, subtitle: String) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 4.dp)) {
        Text(title, style = MaterialTheme.typography.headlineLarge)
        Text(subtitle, style = MaterialTheme.typography.bodySmall)
    }
}

/** 安全状态大卡：safe/attention/danger 三档配色 */
@Composable
private fun SafetyStatusCard(overall: String, label: String, sub: String) {
    val (bg, fg, icon, badge) = when (overall) {
        "danger" -> Quad(WarnRedBg, WarnRed, Icons.Filled.Error, "当前风险：紧急")
        "attention" -> Quad(WarnOrangeBg, WarnOrange, Icons.Filled.Warning, "当前风险：需关注")
        else -> Quad(SafeGreenBg, SafeGreen, Icons.Filled.CheckCircle, "当前风险：安全")
    }
    AppCard(containerColor = bg) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = fg, modifier = Modifier.size(56.dp))
            Spacer(Modifier.width(14.dp))
            Column {
                Text(label, fontSize = 24.sp, fontWeight = FontWeight.ExtraBold, color = fg)
                Text(sub, style = MaterialTheme.typography.bodySmall)
            }
        }
        Spacer(Modifier.height(10.dp))
        Surface(shape = androidx.compose.foundation.shape.RoundedCornerShape(50), color = androidx.compose.ui.graphics.Color.White) {
            Text(badge, color = fg, fontWeight = FontWeight.Bold, fontSize = 15.sp,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 5.dp))
        }
    }
}

private data class Quad<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)

@Composable
private fun StatBox(number: String, label: String, color: androidx.compose.ui.graphics.Color, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(14.dp),
        color = BgPage
    ) {
        Column(Modifier.padding(vertical = 14.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(number, fontSize = 28.sp, fontWeight = FontWeight.ExtraBold, color = color)
            Text(label, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun DeviceRow(device: Device) {
    val (label, color) = when {
        !device.online -> "离线" to TextSecondary
        device.signal == "weak" -> "信号弱" to WarnAmber
        else -> "在线" to SafeGreen
    }
    Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Filled.Videocam, contentDescription = null, tint = color, modifier = Modifier.size(28.dp))
        Spacer(Modifier.width(10.dp))
        Text(device.name, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text(label, color = color, fontWeight = FontWeight.Bold, fontSize = 16.sp)
    }
}
