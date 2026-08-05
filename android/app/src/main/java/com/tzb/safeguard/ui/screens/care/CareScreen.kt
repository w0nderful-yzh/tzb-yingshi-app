package com.tzb.safeguard.ui.screens.care

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ui.components.*
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.theme.*

/**
 * 心理关怀页。对应 prototype/pages/care.html。
 * 注意：按接口文档约定，关怀模块后端接口本期暂缓，
 * 本页情绪打卡与心情趋势为本地演示数据，待 POST /moods 落地后接入仓库层。
 */
@Composable
fun CareScreen(navController: NavHostController) {
    var selectedMood by remember { mutableStateOf("开心") }
    val moods = listOf(
        MoodMeta("开心", Icons.Filled.SentimentVerySatisfied, SafeGreen),
        MoodMeta("平静", Icons.Filled.SentimentSatisfied, Primary),
        MoodMeta("难过", Icons.Filled.SentimentDissatisfied, WarnAmber),
        MoodMeta("烦躁", Icons.Filled.SentimentVeryDissatisfied, WarnRed)
    )

    Scaffold(
        bottomBar = {
            AppBottomBar(ElderTabs, Routes.CARE) { route ->
                navController.navigate(route) { launchSingleTop = true }
            }
        },
        containerColor = BgPage
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PageHeader("今天心情怎么样？", "点一下告诉我，我会陪着你")
            LazyColumn(
                contentPadding = PaddingValues(14.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                // 情绪打卡（本地演示）
                item {
                    AppCard {
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            moods.forEach { mood ->
                                MoodButton(mood, selected = selectedMood == mood.label,
                                    onClick = { selectedMood = mood.label },
                                    modifier = Modifier.weight(1f))
                            }
                        }
                        Spacer(Modifier.height(10.dp))
                        Text("已记录：今天 $selectedMood · 连续打卡 12 天",
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.fillMaxWidth(),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                    }
                }

                // AI 陪伴
                item {
                    AppCard {
                        Text("和“小安”聊聊", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.height(10.dp))
                        Surface(
                            shape = RoundedCornerShape(14.dp),
                            color = Color(0xFFEEF4FD)
                        ) {
                            Text(
                                "早上好！昨晚您睡得不错。今天天气晴朗，适合到阳台晒晒太阳。想听听老歌，还是聊聊家常？",
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.padding(14.dp)
                            )
                        }
                        Spacer(Modifier.height(12.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Box(Modifier.weight(1f)) {
                                BigActionButton(text = "按住说话", icon = Icons.Filled.Mic) { /* TODO: 语音对话 */ }
                            }
                            Box(Modifier.weight(1f)) {
                                BigActionButton(text = "听戏听歌", icon = Icons.Filled.MusicNote, outlined = true) { }
                            }
                        }
                    }
                }

                // 联系家人
                item {
                    AppCard {
                        Text("想家人了？", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.height(4.dp))
                        FamilyRow("儿子 张伟")
                        FamilyRow("女儿 张莉")
                        Spacer(Modifier.height(8.dp))
                        BigActionButton(text = "给女儿留言", containerColor = Color(0xFFEEF1F5), contentColor = TextMain) { }
                    }
                }

                // 7 天心情（本地演示数据）
                item {
                    AppCard {
                        Text("最近 7 天心情", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.height(10.dp))
                        MoodLineChart(listOf(4.2f, 4.0f, 4.5f, 3.0f, 3.6f, 4.2f, 4.0f))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            listOf("三", "四", "五", "六", "日", "一", "二").forEach {
                                Text(it, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                        Spacer(Modifier.height(6.dp))
                        Text("周六心情略低——家人来访后离开，属正常波动。", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

private data class MoodMeta(val label: String, val icon: ImageVector, val color: Color)

@Composable
private fun MoodButton(mood: MoodMeta, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier.heightIn(min = 92.dp),
        shape = RoundedCornerShape(14.dp),
        color = if (selected) Color(0xFFEAF2FD) else Color.White,
        border = BorderStroke(2.dp, if (selected) Primary else LineColor),
        onClick = onClick
    ) {
        Column(
            Modifier.padding(vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(mood.icon, contentDescription = mood.label, tint = mood.color, modifier = Modifier.size(34.dp))
            Spacer(Modifier.height(6.dp))
            Text(mood.label, style = MaterialTheme.typography.titleMedium,
                color = if (selected) PrimaryDark else TextMain)
        }
    }
}

@Composable
private fun FamilyRow(name: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Filled.AccountCircle, contentDescription = null, tint = Primary, modifier = Modifier.size(34.dp))
        Spacer(Modifier.width(10.dp))
        Text(name, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text("视频通话 ›", color = Primary, style = MaterialTheme.typography.titleMedium)
    }
}
