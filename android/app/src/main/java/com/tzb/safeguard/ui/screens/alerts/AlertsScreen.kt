package com.tzb.safeguard.ui.screens.alerts

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
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
 * 告警消息中心。对应 prototype/pages/alerts.html。
 * 按级别筛选 + 列表 + 点击进入详情处置。
 */
@Composable
fun AlertsScreen(
    navController: NavHostController,
    vm: AlertsViewModel = appViewModel { AlertsViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()
    val isFamily = Session.role == "family"
    val tabs = if (isFamily) FamilyTabs else ElderTabs

    Scaffold(
        bottomBar = {
            AppBottomBar(tabs, Routes.ALERTS) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            val unread = (state as? UiState.Success)?.data?.unreadCount ?: 0
            PageHeader("消息中心", if (unread > 0) "$unread 条未读 · 按风险等级排序" else "全部已读")

            // 分级筛选
            val currentFilter = (state as? UiState.Success)?.data?.filter ?: AlertFilter.ALL
            LazyRow(
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(AlertFilter.entries.toList()) { f ->
                    FilterChip(
                        selected = currentFilter == f,
                        onClick = { vm.selectFilter(f) },
                        label = { Text(f.label, fontSize = 16.sp) },
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = when (f) {
                                AlertFilter.EMERGENCY -> WarnRed
                                AlertFilter.WARNING -> WarnOrange
                                AlertFilter.REMINDER -> WarnAmber
                                else -> Primary
                            },
                            selectedLabelColor = Color.White
                        )
                    )
                }
            }

            StateBox(state = state, onRetry = vm::load, modifier = Modifier.fillMaxSize()) { data ->
                if (data.events.isEmpty()) {
                    EmptyBox(if (data.filter == AlertFilter.ALL) "暂无消息" else "该分类下暂无消息")
                } else {
                    LazyColumn(
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 4.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(data.events, key = { it.event_id }) { event ->
                            AlertCard(event) {
                                navController.navigate(Routes.alertDetail(event.event_id))
                            }
                        }
                    }
                }
            }
        }
    }
}
