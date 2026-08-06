package com.tzb.safeguard.ui.screens.monitor

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Lock
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
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.components.StateBox
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

@Composable
fun MonitorScreen(
    navController: NavHostController,
    vm: MonitorViewModel = appViewModel { MonitorViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(Modifier.fillMaxWidth().padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { navController.popBackStack() }) {
                    Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
                }
                Column {
                    Text("现场复核", style = MaterialTheme.typography.headlineLarge)
                    Text("仅在风险事件中按需查看", style = MaterialTheme.typography.bodySmall)
                }
            }
            StateBox(state, vm::load, Modifier.fillMaxSize()) { data ->
                LazyColumn(
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    item {
                        LiveVideoPlayer(
                            liveSession = data.liveSession,
                            deviceName = data.selected?.name ?: "摄像头",
                            streamLoading = data.streamLoading,
                            streamError = data.streamError,
                            onRetry = vm::retryLive,
                            modifier = Modifier.fillMaxWidth().aspectRatio(1.6f)
                        )
                    }
                    item {
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(data.devices) { device ->
                                FilterChip(
                                    selected = device.device_id == data.selected?.device_id,
                                    onClick = { vm.select(device) },
                                    label = { Text(device.name, fontSize = 16.sp) },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = Primary,
                                        selectedLabelColor = Color.White
                                    )
                                )
                            }
                        }
                    }
                    item {
                        AppCard {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Filled.Lock, contentDescription = null, tint = Primary)
                                Spacer(Modifier.width(10.dp))
                                Column {
                                    Text("授权访问", style = MaterialTheme.typography.titleMedium)
                                    Text("直播凭证由后端短时签发，客户端不保存萤石 AppSecret。", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
