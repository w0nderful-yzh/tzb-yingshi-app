package com.tzb.safeguard.ui.screens.alerts

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.ui.components.AppBottomBar
import com.tzb.safeguard.ui.components.AppTabs
import com.tzb.safeguard.ui.components.EmptyBox
import com.tzb.safeguard.ui.components.EvidenceImage
import com.tzb.safeguard.ui.components.StateBox
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.components.formatTime
import com.tzb.safeguard.ui.components.levelColor
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.LineColor
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.TextSecondary

@Composable
fun AlertsScreen(
    navController: NavHostController,
    vm: AlertsViewModel = appViewModel { AlertsViewModel(ServiceLocator.repository) },
) {
    val state by vm.state.collectAsState()
    Scaffold(
        modifier = Modifier.statusBarsPadding(),
        bottomBar = {
            AppBottomBar(AppTabs, Routes.ALERTS) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = Color.White,
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Text(
                "消息",
                style = MaterialTheme.typography.headlineLarge,
                modifier = Modifier.padding(start = 18.dp, top = 15.dp, bottom = 8.dp),
            )
            val selected = (state as? UiState.Success)?.data?.filter ?: AlertFilter.ALL
            LazyRow(
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 5.dp),
                horizontalArrangement = Arrangement.spacedBy(9.dp),
            ) {
                items(AlertFilter.entries.toList()) { filter ->
                    FilterChip(
                        selected = selected == filter,
                        onClick = { vm.selectFilter(filter) },
                        label = { Text(filter.label, fontSize = 14.sp) },
                        shape = RoundedCornerShape(8.dp),
                        colors = FilterChipDefaults.filterChipColors(
                            containerColor = Color(0xFFF4F5F7),
                            selectedContainerColor = Color(0xFFEAF1FF),
                            selectedLabelColor = Primary,
                        ),
                        border = FilterChipDefaults.filterChipBorder(
                            enabled = true,
                            selected = selected == filter,
                            borderColor = Color.Transparent,
                            selectedBorderColor = Primary,
                        ),
                    )
                }
            }
            StateBox(state, vm::load, Modifier.fillMaxSize()) { data ->
                if (data.events.isEmpty()) {
                    EmptyBox(if (data.filter == AlertFilter.ALL) "暂无诈骗风险预测" else "该分类下暂无消息")
                } else {
                    LazyColumn(
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        items(data.events, key = { it.event_id }) { event ->
                            PredictionMessageCard(event) {
                                navController.navigate(Routes.alertDetail(event.event_id))
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PredictionMessageCard(event: RiskEvent, onClick: () -> Unit) {
    val frame = event.evidence_frames.lastOrNull()
    val title = event.fraud_state_label ?: "诈骗风险预警"
    Surface(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(15.dp),
        color = Color.White,
        border = BorderStroke(1.dp, LineColor),
        shadowElevation = 2.dp,
    ) {
        Column(Modifier.padding(11.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(shape = CircleShape, color = levelColor(event.level)) {
                    Icon(
                        Icons.Filled.Shield,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.padding(7.dp).size(18.dp),
                    )
                }
                Spacer(Modifier.width(9.dp))
                Text(
                    title,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp,
                    modifier = Modifier.weight(1f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(formatTime(event.occurred_at), color = TextSecondary, fontSize = 12.sp)
            }
            Spacer(Modifier.size(9.dp))
            EvidenceImage(
                imageUrl = frame?.image_url ?: event.evidence_image_url,
                timestamp = formatTime(frame?.captured_at ?: event.occurred_at).removePrefix("今天 "),
                modifier = Modifier.fillMaxWidth().aspectRatio(2.15f),
            )
            Spacer(Modifier.size(8.dp))
            Text(
                "预测原因：${event.summary.ifBlank { "多模态证据显示诈骗风险正在上升" }}",
                color = TextSecondary,
                fontSize = 13.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                "位置：${event.location.ifBlank { "摄像头 ${event.device_id}" }}",
                color = TextSecondary,
                fontSize = 13.sp,
                maxLines = 1,
            )
        }
    }
}
