package com.tzb.safeguard.ui.screens.profile

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.core.content.ContextCompat
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.media.MonitorServiceStatus
import com.tzb.safeguard.data.media.Ys7MonitorService
import com.tzb.safeguard.data.model.Contact
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 个人 / 设备管理页。对应 prototype/pages/profile.html。
 * 只展示后端已有的用户、设备和联系人数据，不提供未实现的编辑开关。
 */
@Composable
fun ProfileScreen(
    navController: NavHostController,
    vm: ProfileViewModel = appViewModel { ProfileViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    val monitorStatus by Ys7MonitorService.status.collectAsState()
    val context = LocalContext.current
    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) Ys7MonitorService.start(context)
    }
    val startMonitor = {
        if (
            Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            Ys7MonitorService.start(context)
        } else {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }
    Scaffold(
        modifier = Modifier.statusBarsPadding(),
        bottomBar = {
            AppBottomBar(AppTabs, Routes.PROFILE) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PageHeader("我的", "账号、设备与紧急联系人")
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
                                    Text("家属守护账号", style = MaterialTheme.typography.bodySmall)
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
                        }
                    }

                    item {
                        MonitorServiceCard(
                            status = monitorStatus,
                            onStart = startMonitor,
                            onStop = { Ys7MonitorService.stop(context) },
                        )
                    }

                    item { SettingsCard() }

                    item {
                        OutlinedButton(
                            onClick = {
                                Ys7MonitorService.stop(context)
                                vm.logout()
                            },
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Filled.Logout, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("退出登录")
                        }
                    }

                }
            }
        }
    }
}

@Composable
private fun MonitorServiceCard(
    status: MonitorServiceStatus,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Filled.HealthAndSafety,
                contentDescription = null,
                tint = if (status.enabled) SafeGreen else TextSecondary,
            )
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("持续守护", style = MaterialTheme.typography.titleLarge)
                Text(status.detail, style = MaterialTheme.typography.bodySmall)
            }
            Button(onClick = if (status.enabled) onStop else onStart) {
                Text(if (status.enabled) "停止" else "开启")
            }
        }
        if (status.enabled) {
            Spacer(Modifier.height(10.dp))
            CapabilityRow(
                "摄像头音频",
                if (status.mediaConnected) "监听中" else "连接中",
                if (status.mediaConnected) SafeGreen else WarnAmber,
            )
            CapabilityRow(
                "实时告警",
                if (status.alertsConnected) "已连接" else "重连中",
                if (status.alertsConnected) SafeGreen else WarnAmber,
            )
        }
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

@Composable
private fun SettingsCard() {
    AppCard {
        Text("能力接入状态", style = MaterialTheme.typography.titleLarge)
        CapabilityRow("诈骗风险预测", "运行中", SafeGreen)
        CapabilityRow("跌倒风险预测", "待模块接入", TextSecondary)
        CapabilityRow("心理健康趋势", "待模块接入", TextSecondary)
        Spacer(Modifier.height(8.dp))
        Text("萤石密钥仅保留在后端；事件证据按后端留存策略管理。",
            style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun CapabilityRow(label: String, status: String, color: androidx.compose.ui.graphics.Color) {
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text(status, style = MaterialTheme.typography.bodySmall, color = color, fontWeight = FontWeight.Bold)
    }
}
