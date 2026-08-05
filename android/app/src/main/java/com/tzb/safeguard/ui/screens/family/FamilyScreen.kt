package com.tzb.safeguard.ui.screens.family

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.*

/**
 * 家属端守护看板。对应 prototype/pages/family.html。
 * 含：老人当前状态、待办告警、安全事件历史、活动规律、远程管理入口。
 * 情绪趋势图表按接口文档暂缓。
 */
@Composable
fun FamilyScreen(
    navController: NavHostController,
    vm: FamilyViewModel = appViewModel { FamilyViewModel(ServiceLocator.repository) }
) {
    val state by vm.state.collectAsState()

    Scaffold(
        bottomBar = {
            AppBottomBar(FamilyTabs, Routes.FAMILY) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            // 深蓝色头部（家属端视觉区分）
            Surface(color = Primary) {
                Column(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 14.dp)) {
                    val elderName = (state as? UiState.Success)?.data?.elder?.name ?: ""
                    Text("守护看板${if (elderName.isNotBlank()) " · $elderName" else ""}",
                        style = MaterialTheme.typography.headlineLarge, color = androidx.compose.ui.graphics.Color.White)
                    Text("远程查看 · 数据近 30 天", style = MaterialTheme.typography.bodySmall,
                        color = androidx.compose.ui.graphics.Color(0xFFCFE0F7))
                }
            }

            StateBox(state = state, onRetry = vm::load, modifier = Modifier.fillMaxSize()) { data ->
                LazyColumn(
                    contentPadding = PaddingValues(14.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    // 当前状态横幅
                    item { ElderStatusBanner(data.elder?.overall ?: "safe", data.elder?.last_active_at ?: "") {
                        navController.navigate(Routes.MONITOR) { launchSingleTop = true }
                    } }

                    // 待办告警
                    if (data.pendingEvents.isNotEmpty()) {
                        item {
                            Text("待处理告警（${data.pendingEvents.size}）", style = MaterialTheme.typography.titleLarge)
                        }
                        items(data.pendingEvents, key = { it.event_id }) { event ->
                            AlertCard(event) { navController.navigate(Routes.alertDetail(event.event_id)) }
                        }
                        item {
                            val top = data.pendingEvents.first()
                            BigActionButton(text = "立即处理「${top.title}」", icon = Icons.Filled.PriorityHigh,
                                containerColor = levelColor(top.level)) {
                                navController.navigate(Routes.alertDetail(top.event_id))
                            }
                        }
                    }

                    // 安全事件历史
                    item {
                        AppCard {
                            Text("安全事件历史（近 30 天）", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(8.dp))
                            ChartLegend(listOf("紧急" to WarnRed, "警告" to WarnOrange, "提醒" to WarnAmber))
                            Spacer(Modifier.height(8.dp))
                            if (data.eventsStats.buckets.isEmpty()) {
                                EmptyBox("暂无统计数据")
                            } else {
                                EventsBarChart(data.eventsStats.buckets)
                                Spacer(Modifier.height(6.dp))
                                val total = data.eventsStats.buckets.sumOf { it.reminder + it.warning + it.emergency }
                                Text("共 $total 起事件", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }

                    // 活动规律
                    item {
                        AppCard {
                            Text("每日活动规律（近 7 天平均）", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(8.dp))
                            if (data.activity.hours.isEmpty()) {
                                EmptyBox("暂无活动数据")
                            } else {
                                ActivityHeatStrip(data.activity.hours)
                                Spacer(Modifier.height(6.dp))
                                Text("颜色越深表示活动量越大；规律突变将自动生成「无活动」提醒。",
                                    style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }

                    // 情绪趋势：按接口文档暂缓
                    item {
                        AppCard {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Filled.Favorite, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(28.dp))
                                Spacer(Modifier.width(10.dp))
                                Column {
                                    Text("情绪趋势", style = MaterialTheme.typography.titleMedium)
                                    Text("心理关怀模块本期暂缓，接口落地后开放", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }

                    // 远程管理
                    item {
                        AppCard {
                            Text("远程管理", style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(10.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                Box(Modifier.weight(1f)) {
                                    BigActionButton(text = "实时画面", icon = Icons.Filled.Videocam,
                                        containerColor = androidx.compose.ui.graphics.Color(0xFFEEF1F5), contentColor = TextMain) {
                                        navController.navigate(Routes.MONITOR) { launchSingleTop = true }
                                    }
                                }
                                Box(Modifier.weight(1f)) {
                                    BigActionButton(text = "设备与联系人", icon = Icons.Filled.Settings,
                                        containerColor = androidx.compose.ui.graphics.Color(0xFFEEF1F5), contentColor = TextMain) {
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
}

@Composable
private fun ElderStatusBanner(overall: String, lastActive: String, onViewLive: () -> Unit) {
    val (bg, fg, icon, label) = when (overall) {
        "danger" -> BannerMeta(WarnRedBg, WarnRed, Icons.Filled.Error, "有紧急告警待处理")
        "attention" -> BannerMeta(WarnOrangeBg, WarnOrange, Icons.Filled.Warning, "有告警需关注")
        else -> BannerMeta(SafeGreenBg, SafeGreen, Icons.Filled.CheckCircle, "老人当前安全")
    }
    AppCard(containerColor = bg) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = fg, modifier = Modifier.size(44.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(label, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, color = fg)
                Text("最后活动：${formatTime(lastActive)}", style = MaterialTheme.typography.bodySmall)
            }
            Button(
                onClick = onViewLive,
                colors = ButtonDefaults.buttonColors(containerColor = Primary),
                shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
            ) { Text("看实时", fontSize = 16.sp) }
        }
    }
}

private data class BannerMeta<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)
