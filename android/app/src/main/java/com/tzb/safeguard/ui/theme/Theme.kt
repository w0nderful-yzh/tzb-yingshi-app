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
 * 设计规范与 prototype/assets/style.css 保持一致：
 * - 高对比度浅色底（正文 #1A1A1A / 底 #FFFFFF，对比度 > 15:1）
 * - 告警三级色：提醒琥珀 / 警告橙 / 紧急红
 * - 适老化字号：正文 20sp 起
 */
val Primary = Color(0xFF1B5FC1)
val PrimaryDark = Color(0xFF144A99)
val TextMain = Color(0xFF1A1A1A)
val TextSecondary = Color(0xFF4B5563)
val BgPage = Color(0xFFF5F6F8)
val LineColor = Color(0xFFD8DCE3)

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

/** 适老化排版：整体比普通 App 大 2-4sp */
val ElderTypography = Typography(
    headlineLarge = TextStyle(fontSize = 26.sp, fontWeight = FontWeight.Bold, color = TextMain),
    headlineMedium = TextStyle(fontSize = 23.sp, fontWeight = FontWeight.Bold, color = TextMain),
    titleLarge = TextStyle(fontSize = 21.sp, fontWeight = FontWeight.Bold, color = TextMain),
    titleMedium = TextStyle(fontSize = 19.sp, fontWeight = FontWeight.Bold, color = TextMain),
    bodyLarge = TextStyle(fontSize = 20.sp, color = TextMain, lineHeight = 30.sp),
    bodyMedium = TextStyle(fontSize = 17.sp, color = TextMain, lineHeight = 26.sp),
    bodySmall = TextStyle(fontSize = 15.sp, color = TextSecondary, lineHeight = 22.sp),
    labelLarge = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Bold)
)

@Composable
fun SafeGuardTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        typography = ElderTypography,
        content = content
    )
}
