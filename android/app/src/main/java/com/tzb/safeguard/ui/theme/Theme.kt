package com.tzb.safeguard.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * 家属端极简视觉：纯白背景、深色正文、蓝色操作强调。
 * - 告警三级色：提醒琥珀 / 警告橙 / 紧急红
 * - 适老化字号：正文 20sp 起
 */
val Primary = Color(0xFF2563EB)
val PrimaryDark = Color(0xFF1D4ED8)
val TextMain = Color(0xFF111827)
val TextSecondary = Color(0xFF667085)
val BgPage = Color.White
val LineColor = Color(0xFFE7EAF0)

val SafeGreen = Color(0xFF15803D)
val SafeGreenBg = Color(0xFFE8F5EC)
val WarnAmber = Color(0xFFB45309)      // 提醒
val WarnAmberBg = Color(0xFFFEF3E2)
val WarnOrange = Color(0xFFC2410C)     // 警告
val WarnOrangeBg = Color(0xFFFFEEE4)
val WarnRed = Color(0xFFB91C1C)        // 紧急
val WarnRedBg = Color(0xFFFDE8E8)

private val LightColors = lightColorScheme(
    primary = Primary,
    onPrimary = Color.White,
    background = BgPage,
    onBackground = TextMain,
    surface = Color.White,
    onSurface = TextMain,
    surfaceVariant = BgPage,
    onSurfaceVariant = TextSecondary,
    outline = LineColor,
    error = WarnRed
)

val AppTypography = Typography(
    headlineLarge = TextStyle(fontSize = 25.sp, fontWeight = FontWeight.Bold, color = TextMain),
    headlineMedium = TextStyle(fontSize = 22.sp, fontWeight = FontWeight.Bold, color = TextMain),
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.Bold, color = TextMain),
    titleMedium = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = TextMain),
    bodyLarge = TextStyle(fontSize = 18.sp, color = TextMain, lineHeight = 27.sp),
    bodyMedium = TextStyle(fontSize = 16.sp, color = TextMain, lineHeight = 24.sp),
    bodySmall = TextStyle(fontSize = 14.sp, color = TextSecondary, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
)

@Composable
fun SafeGuardTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        typography = AppTypography,
        content = content
    )
}
