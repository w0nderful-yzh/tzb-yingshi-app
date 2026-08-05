package com.tzb.safeguard.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.tzb.safeguard.data.model.StatsBucket
import com.tzb.safeguard.ui.theme.*

/**
 * 数据看板图表：不引第三方库，用 Compose Canvas 手绘，
 * 视觉与 prototype/pages/family.html 的 SVG 图表保持一致。
 */

/** 图例 */
@Composable
fun ChartLegend(items: List<Pair<String, Color>>) {
    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        items.forEach { (label, color) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Canvas(Modifier.size(12.dp)) { drawRect(color) }
                Spacer(Modifier.width(5.dp))
                Text(label, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/** 安全事件历史柱状图：按周期分组，提醒/警告/紧急三根并列 */
@Composable
fun EventsBarChart(buckets: List<StatsBucket>, modifier: Modifier = Modifier) {
    if (buckets.isEmpty()) return
    val maxValue = (buckets.maxOf { maxOf(it.reminder, it.warning, it.emergency) }).coerceAtLeast(1)

    Column(modifier.fillMaxWidth()) {
        Canvas(Modifier.fillMaxWidth().height(140.dp)) {
            val groupWidth = size.width / buckets.size
            val barWidth = groupWidth / 5f
            val chartHeight = size.height

            buckets.forEachIndexed { i, bucket ->
                val baseX = groupWidth * i + barWidth
                listOf(
                    bucket.reminder to WarnAmber,
                    bucket.warning to WarnOrange,
                    bucket.emergency to WarnRed
                ).forEachIndexed { j, (value, color) ->
                    val h = if (value == 0) 0f else (value.toFloat() / maxValue) * (chartHeight * 0.85f)
                    drawRect(
                        color = if (value == 0) color.copy(alpha = 0.15f) else color,
                        topLeft = Offset(baseX + barWidth * j, chartHeight - h.coerceAtLeast(4f)),
                        size = androidx.compose.ui.geometry.Size(barWidth * 0.8f, h.coerceAtLeast(4f))
                    )
                }
            }
            // 基线
            drawLine(LineColor, Offset(0f, chartHeight - 1f), Offset(size.width, chartHeight - 1f), strokeWidth = 2f)
        }
        // X 轴标签
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceAround) {
            buckets.forEach { Text(it.period, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

/** 24 小时活动热力条：颜色深浅表示活动量 */
@Composable
fun ActivityHeatStrip(hours: List<Double>, modifier: Modifier = Modifier) {
    if (hours.isEmpty()) return
    Column(modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().height(56.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
            hours.forEach { v ->
                val alpha = (0.08f + v.toFloat() * 0.92f).coerceIn(0f, 1f)
                Box(
                    Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .padding(vertical = 2.dp)
                ) {
                    Canvas(Modifier.fillMaxSize()) {
                        drawRoundRect(
                            color = Primary.copy(alpha = alpha),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(6f, 6f)
                        )
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            listOf("0时", "6时", "12时", "18时", "23时").forEach {
                Text(it, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/** 简单折线图（关怀页 7 天心情演示用；接口暂缓，数据本地） */
@Composable
fun MoodLineChart(points: List<Float>, modifier: Modifier = Modifier) {
    if (points.size < 2) return
    Canvas(modifier.fillMaxWidth().height(110.dp)) {
        val stepX = size.width / (points.size - 1)
        // 心情 1-5 映射到高度（5 在上）
        fun yOf(v: Float) = size.height - ((v - 1f) / 4f) * (size.height * 0.8f) - size.height * 0.1f

        // 基线
        drawLine(LineColor, Offset(0f, size.height - 1f), Offset(size.width, size.height - 1f), strokeWidth = 2f)

        for (i in 0 until points.size - 1) {
            drawLine(
                color = Primary,
                start = Offset(stepX * i, yOf(points[i])),
                end = Offset(stepX * (i + 1), yOf(points[i + 1])),
                strokeWidth = 5f
            )
        }
        points.forEachIndexed { i, v ->
            drawCircle(color = Primary, radius = 7f, center = Offset(stepX * i, yOf(v)))
        }
    }
}
