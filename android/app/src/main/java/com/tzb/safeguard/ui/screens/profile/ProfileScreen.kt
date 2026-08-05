package com.tzb.safeguard.ui.screens.profile

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
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
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.Contact
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 个人 / 设备管理页。对应 prototype/pages/profile.html。
 * 老人端只读查看；家属端展示管理入口（编辑类操作本期均为占位）。
 */
@Composable
fun ProfileScreen(
    navController: NavHostController,
    vm: ProfileViewModel = appViewModel { ProfileViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    val isFamily = Session.role == "family"
    val tabs = if (isFamily) FamilyTabs else ElderTabs
    var showRoleDialog by remember { mutableStateOf(false) }

    Scaffold(
        bottomBar = {
            AppBottomBar(tabs, Routes.PROFILE) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PageHeader("我的", "个人信息 · 设备 · 紧急联系人")
            StateBox(state = state, onRetry = vm::load, modifier = Modifier.fillMaxSize()) { data ->
                LazyColumn(
                    contentPadding = PaddingValues(14.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    // 个人卡片
                    item {
                        AppCard {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Surface(shape = CircleShape, color = androidx.compose.ui.graphics.Color(0xFFDBE7FB)) {
                                    Icon(Icons.Filled.Person, contentDescription = null, tint = Primary,
                                        modifier = Modifier.padding(14.dp).size(36.dp))
                                }
                                Spacer(Modifier.width(14.dp))
                                Column {
                                    Text(data.user.name.ifBlank { "未设置姓名" }, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold)
                                    Text("已由 ${data.user.bound_family_count} 位家属守护", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }

                    // 紧急联系人
                    item {
                        AppCard {
                            Text("紧急联系人（按顺序呼叫）", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(6.dp))
                            if (data.contacts.isEmpty()) {
                                EmptyBox("暂无紧急联系人")
                            } else {
                                data.contacts.forEach { ContactRow(it) }
                            }
                            if (isFamily) {
                                Spacer(Modifier.height(8.dp))
                                BigActionButton(text = "编辑联系人", outlined = true) { /* TODO: PUT /contacts */ }
                            }
                        }
                    }

                    // 设备管理
                    item {
                        AppCard {
                            Text("我的设备", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(6.dp))
                            if (data.devices.isEmpty()) {
                                EmptyBox("暂无设备")
                            } else {
                                data.devices.forEach { ProfileDeviceRow(it) }
                            }
                            if (isFamily) {
                                Spacer(Modifier.height(8.dp))
                                BigActionButton(text = "＋ 添加萤石设备", containerColor = androidx.compose.ui.graphics.Color(0xFFEEF1F5),
                                    contentColor = TextMain) { /* TODO: POST /devices 扫码绑定 */ }
                            }
                        }
                    }

                    // 守护设置（只读展示 + 家属端占位开关）
                    item { SettingsCard(isFamily) }

                    // 身份切换（联调期入口）
                    item {
                        BigActionButton(
                            text = if (isFamily) "切换到老人端" else "切换到家属端",
                            icon = Icons.Filled.SwapHoriz,
                            outlined = true
                        ) { showRoleDialog = true }
                    }
                }
            }
        }
    }

    if (showRoleDialog) {
        AlertDialog(
            onDismissRequest = { showRoleDialog = false },
            title = { Text("切换使用身份？") },
            text = { Text("将返回身份选择页。") },
            confirmButton = {
                TextButton(onClick = {
                    showRoleDialog = false
                    navController.navigate(Routes.ROLE) {
                        popUpTo(0) { inclusive = true }
                    }
                }) { Text("确认") }
            },
            dismissButton = { TextButton(onClick = { showRoleDialog = false }) { Text("取消") } }
        )
    }
}

@Composable
private fun ContactRow(contact: Contact) {
    Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Surface(shape = CircleShape, color = levelBgColor(
            when (contact.order) { 1 -> "emergency"; 2 -> "warning"; else -> "reminder" }
        )) {
            Text("${contact.order}", fontWeight = FontWeight.Bold,
                color = levelColor(when (contact.order) { 1 -> "emergency"; 2 -> "warning"; else -> "reminder" }),
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp))
        }
        Spacer(Modifier.width(10.dp))
        Text(contact.name, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text(contact.phone, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun ProfileDeviceRow(device: Device) {
    val (label, color) = when {
        !device.online -> "离线" to TextSecondary
        device.signal == "weak" -> "信号弱" to WarnAmber
        else -> "在线" to SafeGreen
    }
    Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Filled.Videocam, contentDescription = null, tint = color, modifier = Modifier.size(28.dp))
        Spacer(Modifier.width(10.dp))
        Column {
            Text(device.name, style = MaterialTheme.typography.bodyMedium)
            Text(device.device_id, style = MaterialTheme.typography.bodySmall)
        }
        Spacer(Modifier.weight(1f))
        Text(label, color = color, fontWeight = FontWeight.Bold, fontSize = 16.sp)
    }
}

/** 守护设置：开关为本地展示态，正式版读写 GET/PUT /settings */
@Composable
private fun SettingsCard(isFamily: Boolean) {
    var fraud by remember { mutableStateOf(true) }
    var fall by remember { mutableStateOf(true) }
    var night by remember { mutableStateOf(true) }
    var voice by remember { mutableStateOf(true) }

    AppCard {
        Text("守护设置", style = MaterialTheme.typography.titleLarge)
        SettingSwitchRow("防诈骗电话监听", fraud, isFamily) { fraud = it }
        SettingSwitchRow("跌倒检测", fall, isFamily) { fall = it }
        SettingSwitchRow("夜间离床提醒", night, isFamily) { night = it }
        SettingSwitchRow("语音播报提示", voice, isFamily) { voice = it }
        Spacer(Modifier.height(8.dp))
        Text("隐私承诺：不保存连续音视频，仅保留风险事件瞬间证据，可随时关闭。",
            style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun SettingSwitchRow(label: String, checked: Boolean, enabled: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange, enabled = enabled)
    }
}
