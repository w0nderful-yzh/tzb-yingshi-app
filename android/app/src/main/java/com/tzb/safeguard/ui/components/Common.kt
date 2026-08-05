package com.tzb.safeguard.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.ui.theme.*

// ---------------- UI 状态 ----------------

/** 页面三态：加载 / 成功 / 失败。空数据由 Success 内各页面自行渲染空态 */
sealed interface UiState<out T> {
    data object Loading : UiState<Nothing>
    data class Success<T>(val data: T) : UiState<T>
    data class Error(val message: String) : UiState<Nothing>
}

/** 统一的状态容器：Loading 转圈、Error 展示重试，Success 交给 content */
@Composable
fun <T> StateBox(
    state: UiState<T>,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable (T) -> Unit
) {
    when (state) {
        is UiState.Loading -> Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(color = Primary)
        }
        is UiState.Error -> Column(
            modifier.fillMaxSize().padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(Icons.Filled.CloudOff, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(48.dp))
            Spacer(Modifier.height(12.dp))
            Text(state.message, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
            Spacer(Modifier.height(16.dp))
            BigActionButton(text = "重新加载", icon = Icons.Filled.Refresh, onClick = onRetry)
        }
        is UiState.Success -> content(state.data)
    }
}

/** 空数据占位 */
@Composable
fun EmptyBox(text: String, modifier: Modifier = Modifier) {
    Column(
        modifier.fillMaxWidth().padding(vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(Icons.Filled.Inbox, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(44.dp))
        Spacer(Modifier.height(8.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
    }
}

// ---------------- 基础组件 ----------------

/** 卡片：白底 + 细描边 + 16dp 圆角，与原型一致 */
@Composable
fun AppCard(
    modifier: Modifier = Modifier,
    containerColor: Color = Color.White,
    content: @Composable ColumnScope.() -> Unit
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        color = containerColor,
        border = BorderStroke(1.dp, LineColor)
    ) {
        Column(Modifier.padding(16.dp), content = content)
    }
}

/** 大按钮：最小高度 60dp（适老触控热区），22sp 加粗；onClick 置末尾以支持尾随 lambda */
@Composable
fun BigActionButton(
    text: String,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    containerColor: Color = Primary,
    contentColor: Color = Color.White,
    outlined: Boolean = false,
    onClick: () -> Unit
) {
    if (outlined) {
        OutlinedButton(
            onClick = onClick,
            modifier = modifier.fillMaxWidth().heightIn(min = 60.dp),
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(2.dp, Primary),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = PrimaryDark)
        ) { ButtonInner(text, icon) }
    } else {
        Button(
            onClick = onClick,
            modifier = modifier.fillMaxWidth().heightIn(min = 60.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = containerColor, contentColor = contentColor)
        ) { ButtonInner(text, icon) }
    }
}

@Composable
private fun ButtonInner(text: String, icon: ImageVector?) {
    if (icon != null) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(24.dp))
        Spacer(Modifier.width(8.dp))
    }
    Text(text, fontSize = 20.sp, fontWeight = FontWeight.Bold)
}

/** 告警级别徽标：颜色 + 图标 + 文字三通道（不单独依赖颜色） */
@Composable
fun LevelBadge(level: String) {
    val (label, fg, bg, icon) = when (level) {
        "emergency" -> LevelMeta("紧急", WarnRed, WarnRedBg, Icons.Filled.Error)
        "warning" -> LevelMeta("警告", WarnOrange, WarnOrangeBg, Icons.Filled.Warning)
        "reminder" -> LevelMeta("提醒", WarnAmber, WarnAmberBg, Icons.Filled.Schedule)
        else -> LevelMeta("安全", SafeGreen, SafeGreenBg, Icons.Filled.CheckCircle)
    }
    Surface(shape = RoundedCornerShape(50), color = bg) {
        Row(
            Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = null, tint = fg, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
            Text(label, color = fg, fontSize = 15.sp, fontWeight = FontWeight.Bold)
        }
    }
}

private data class LevelMeta(val label: String, val fg: Color, val bg: Color, val icon: ImageVector)

/** 事件类型 -> 中文标签 + 图标 + 颜色 */
fun eventTypeMeta(type: String): Triple<String, ImageVector, Color> = when (type) {
    "fall_suspected" -> Triple("跌倒", Icons.Filled.PersonalInjury, WarnRed)
    "fraud_suspected" -> Triple("防诈", Icons.Filled.GppMaybe, WarnOrange)
    "stranger" -> Triple("陌生人", Icons.Filled.Face, WarnOrange)
    "inactivity" -> Triple("无活动", Icons.Filled.HourglassEmpty, WarnAmber)
    "sos" -> Triple("紧急求助", Icons.Filled.Sos, WarnRed)
    "device_offline" -> Triple("设备离线", Icons.Filled.VideocamOff, WarnAmber)
    "night_leave_bed" -> Triple("夜间离床", Icons.Filled.Bedtime, WarnAmber)
    "sedentary" -> Triple("久坐", Icons.Filled.AirlineSeatReclineNormal, WarnAmber)
    else -> Triple("事件", Icons.Filled.Notifications, TextSecondary)
}

fun levelColor(level: String): Color = when (level) {
    "emergency" -> WarnRed
    "warning" -> WarnOrange
    "reminder" -> WarnAmber
    else -> SafeGreen
}

fun levelBgColor(level: String): Color = when (level) {
    "emergency" -> WarnRedBg
    "warning" -> WarnOrangeBg
    "reminder" -> WarnAmberBg
    else -> SafeGreenBg
}

fun eventStatusLabel(status: String): String = when (status) {
    "open" -> "待确认"
    "acknowledged" -> "已知晓"
    "resolved" -> "已处理"
    "false_alarm" -> "误报"
    else -> status
}

/** 告警条目卡片：左侧 8dp 级别色条，与原型一致 */
@Composable
fun AlertCard(event: RiskEvent, onClick: () -> Unit) {
    val (typeLabel, typeIcon, typeColor) = eventTypeMeta(event.type)
    Surface(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(14.dp),
        color = Color.White,
        border = BorderStroke(1.dp, LineColor)
    ) {
        Row(Modifier.height(IntrinsicSize.Min)) {
            // 级别色条
            Box(Modifier.width(8.dp).fillMaxHeight().padding(0.dp)) {
                Surface(color = levelColor(event.level), modifier = Modifier.fillMaxSize()) {}
            }
            Row(Modifier.padding(14.dp), verticalAlignment = Alignment.Top) {
                Surface(shape = RoundedCornerShape(10.dp), color = levelBgColor(event.level)) {
                    Icon(typeIcon, contentDescription = typeLabel, tint = typeColor,
                        modifier = Modifier.padding(9.dp).size(24.dp))
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(event.title, style = MaterialTheme.typography.titleMedium)
                    if (event.summary.isNotBlank()) {
                        Text(event.summary, style = MaterialTheme.typography.bodySmall, maxLines = 2)
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(formatTime(event.occurred_at), style = MaterialTheme.typography.bodySmall)
                }
                Text(
                    eventStatusLabel(event.status),
                    fontSize = 14.sp,
                    fontWeight = if (event.status == "open") FontWeight.Bold else FontWeight.Normal,
                    color = if (event.status == "open") levelColor(event.level) else SafeGreen
                )
            }
        }
    }
}

/** ISO 时间 -> 展示用短格式（"今天 15:02" / "08-03 16:48"） */
fun formatTime(iso: String): String {
    if (iso.isBlank()) return ""
    return try {
        val parser = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", java.util.Locale.US)
        val date = parser.parse(iso) ?: return iso
        val today = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date())
        val day = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(date)
        val hm = java.text.SimpleDateFormat("HH:mm", java.util.Locale.US).format(date)
        if (day == today) "今天 $hm" else java.text.SimpleDateFormat("MM-dd HH:mm", java.util.Locale.US).format(date)
    } catch (e: Exception) { iso }
}

/** 键值行（详情/状态展示） */
@Composable
fun KeyValueRow(key: String, value: String, valueColor: Color = TextMain) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(key, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
        Spacer(Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, color = valueColor)
    }
    HorizontalDivider(color = LineColor.copy(alpha = 0.5f))
}

// ---------------- 底部导航 ----------------

data class TabItem(val route: String, val label: String, val icon: ImageVector, val badge: Int = 0)

val ElderTabs = listOf(
    TabItem("home", "首页", Icons.Filled.Home),
    TabItem("monitor", "监控", Icons.Filled.Videocam),
    TabItem("alerts", "消息", Icons.Filled.Notifications, badge = 2),
    TabItem("care", "关怀", Icons.Filled.Favorite),
    TabItem("profile", "我的", Icons.Filled.Person)
)

val FamilyTabs = listOf(
    TabItem("family", "看板", Icons.Filled.Dashboard),
    TabItem("alerts", "告警", Icons.Filled.Notifications, badge = 1),
    TabItem("monitor", "实时", Icons.Filled.Videocam),
    TabItem("profile", "设置", Icons.Filled.Settings)
)

@Composable
fun AppBottomBar(tabs: List<TabItem>, currentRoute: String?, onNavigate: (String) -> Unit) {
    NavigationBar(containerColor = Color.White, tonalElevation = 0.dp) {
        tabs.forEach { tab ->
            NavigationBarItem(
                selected = currentRoute == tab.route,
                onClick = { onNavigate(tab.route) },
                icon = {
                    BadgedBox(badge = {
                        if (tab.badge > 0) Badge(containerColor = WarnRed) { Text("${tab.badge}") }
                    }) {
                        Icon(tab.icon, contentDescription = tab.label, modifier = Modifier.size(28.dp))
                    }
                },
                label = { Text(tab.label, fontSize = 14.sp) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = Primary,
                    selectedTextColor = Primary,
                    unselectedIconColor = TextSecondary,
                    unselectedTextColor = TextSecondary,
                    indicatorColor = Color(0xFFEAF2FD)
                )
            )
        }
    }
}

/** 页面头：大标题 + 副标题，替代 TopAppBar（避免实验性 API） */
@Composable
fun PageHeader(
    title: String,
    subtitle: String = "",
    containerColor: Color = Color.White,
    titleColor: Color = TextMain,
    subColor: Color = TextSecondary
) {
    Surface(color = containerColor) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 14.dp)) {
            Text(title, style = MaterialTheme.typography.headlineLarge, color = titleColor)
            if (subtitle.isNotBlank()) {
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = subColor)
            }
        }
    }
    HorizontalDivider(color = LineColor)
}
