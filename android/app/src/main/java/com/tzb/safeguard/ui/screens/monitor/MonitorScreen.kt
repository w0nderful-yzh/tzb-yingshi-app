package com.tzb.safeguard.ui.screens.monitor

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.Session
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 实时监控页。对应 prototype/pages/monitor.html。
 * 视频区通过后端签发的短时会话接入萤石原生 SDK，下方为设备切换与 AI 识别状态。
 */
@Composable
fun MonitorScreen(
    navController: NavHostController,
    vm: MonitorViewModel = appViewModel { MonitorViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    val isFamily = Session.role == "family"
    val tabs = if (isFamily) FamilyTabs else ElderTabs
    val currentRoute = Routes.MONITOR

    Scaffold(
        bottomBar = {
            AppBottomBar(tabs, currentRoute) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PageHeader("实时监控", "画面仅本人与授权家属可见 · 默认不留存录像")
            StateBox(state = state, onRetry = vm::load, modifier = Modifier.fillMaxSize()) { data ->
                LazyColumn(
                    contentPadding = PaddingValues(14.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    item {
                        LiveVideoPlayer(
                            liveSession = data.liveSession,
                            deviceName = data.selected?.name ?: "摄像头",
                            streamLoading = data.streamLoading,
                            streamError = data.streamError,
                            onRetry = vm::retryLive,
                            modifier = Modifier.fillMaxWidth().aspectRatio(1.6f),
                        )
                    }

                    // 摄像头切换
                    item {
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(data.devices) { device ->
                                val selected = device.device_id == data.selected?.device_id
                                FilterChip(
                                    selected = selected,
                                    onClick = { vm.select(device) },
                                    label = { Text(device.name.replace("摄像头", ""), fontSize = 16.sp) },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = Primary,
                                        selectedLabelColor = Color.White
                                    )
                                )
                            }
                        }
                    }

                    // AI 识别状态
                    item {
                        AppCard {
                            Text("智能识别状态", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(4.dp))
                            data.recognition.forEachIndexed { _, (k, v) ->
                                KeyValueRow(k, v, if (v == "低" || v == "无异常") SafeGreen else TextMain)
                            }
                            Spacer(Modifier.height(8.dp))
                            Text("说明：识别结果仅为辅助判断，最终以人工确认为准。", style = MaterialTheme.typography.bodySmall)
                        }
                    }

                    // 快捷操作
                    item {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Box(Modifier.weight(1f)) {
                                BigActionButton(text = "语音通话", icon = Icons.Filled.Mic,
                                    containerColor = Color(0xFFEEF1F5), contentColor = TextMain) { /* TODO: 双向语音 */ }
                            }
                            Box(Modifier.weight(1f)) {
                                BigActionButton(text = "设备设置", icon = Icons.Filled.Settings,
                                    containerColor = Color(0xFFEEF1F5), contentColor = TextMain) {
                                    navController.navigate(Routes.PROFILE) { launchSingleTop = true }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
