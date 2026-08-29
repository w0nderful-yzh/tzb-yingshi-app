package com.tzb.safeguard.ui.screens.care

import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ServiceLocator
import com.tzb.safeguard.data.psychology.model.CognitiveCompletedReference
import com.tzb.safeguard.data.psychology.model.CognitiveOverview
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import com.tzb.safeguard.data.psychology.model.PsychologyCompletedReference
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.navigation.appViewModel
import com.tzb.safeguard.ui.theme.BgPage
import com.tzb.safeguard.ui.theme.Primary
import com.tzb.safeguard.ui.theme.TextSecondary

@Composable
fun CareScreen(
    navController: NavHostController,
    viewModel: CareViewModel = appViewModel {
        CareViewModel(ServiceLocator.repository, ServiceLocator.psychologyRepository)
    },
) {
    val psychologyState by viewModel.psychologyState.collectAsState()
    val cognitiveState by viewModel.cognitiveState.collectAsState()
    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(
                Modifier.fillMaxWidth().padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = { navController.popBackStack() }) {
                    Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
                }
                Text("心理与认知评估", style = MaterialTheme.typography.headlineLarge)
            }
            Column(
                Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                PsychologyAssessmentCard(psychologyState, viewModel::loadPsychologyOverview)
                CognitiveAssessmentCard(cognitiveState, viewModel::loadCognitiveOverview)
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun CognitiveAssessmentCard(
    state: UiState<CognitiveOverview>,
    onRetry: () -> Unit,
) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Filled.Psychology,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(34.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("认知状态辅助评估", style = MaterialTheme.typography.titleLarge)
                Text(
                    "基于日常语音声学特征",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        when (state) {
            UiState.Loading -> {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(10.dp))
                    Text("正在读取最近一次认知评估状态")
                }
            }
            is UiState.Error -> {
                Text("认知状态辅助评估服务暂不可用")
                Text(state.message, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                TextButton(onClick = onRetry) { Text("重新加载") }
            }
            is UiState.Success -> CognitiveAssessmentContent(state.data)
        }
    }
}

@Composable
private fun CognitiveAssessmentContent(overview: CognitiveOverview) {
    val stateLabel = cognitiveStateLabel(overview.assessment_state)
    DataRow("当前状态", stateLabel)
    DataRow("分析来源", "语音声学特征")

    if (overview.assessment_state == "completed") {
        overview.estimated_mmse_score?.let { score ->
            Spacer(Modifier.height(12.dp))
            Text("AI辅助MMSE估计", style = MaterialTheme.typography.titleSmall)
            Text(
                "${"%.1f".format(score)} / 30",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
        }
        Spacer(Modifier.height(8.dp))
        DataRow("数据质量", dataQualityLabel(overview.data_quality))
        overview.updated_at?.takeIf { it.isNotBlank() }?.let {
            DataRow("最近评估时间", formatAssessmentTime(it))
        }
        Spacer(Modifier.height(6.dp))
        Text(overview.evidence_summary, style = MaterialTheme.typography.bodyMedium)
    } else {
        Spacer(Modifier.height(8.dp))
        Text(overview.evidence_summary, style = MaterialTheme.typography.bodyMedium)
        DataRow("数据质量", dataQualityLabel(overview.data_quality))
        if (overview.assessment_state == "processing") {
            overview.latest_completed?.let { completed ->
                Spacer(Modifier.height(14.dp))
                Text("上一轮辅助评估", style = MaterialTheme.typography.titleSmall)
                Text(
                    "新一轮语音采集中，以下为上一轮已完成结果",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
                Spacer(Modifier.height(8.dp))
                PreviousCognitiveCompletedContent(completed)
            }
        }
    }

    Spacer(Modifier.height(10.dp))
    Text(
        overview.disclaimer.ifBlank {
            "AI辅助认知状态评估仅供日常关怀参考，不构成认知障碍或医疗诊断"
        },
        style = MaterialTheme.typography.bodySmall,
        color = TextSecondary,
    )
}

@Composable
private fun PreviousCognitiveCompletedContent(completed: CognitiveCompletedReference) {
    completed.estimated_mmse_score?.let { score ->
        Text(
            "AI辅助MMSE估计 ${"%.1f".format(score)} / 30",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
    }
    DataRow("分析来源", "语音声学特征")
    DataRow("数据质量", dataQualityLabel(completed.data_quality))
    completed.updated_at?.takeIf { it.isNotBlank() }?.let {
        DataRow("最近评估时间", formatAssessmentTime(it))
    }
}

private fun cognitiveStateLabel(state: String): String = when (state) {
    "processing" -> "正在采集语音资料"
    "completed" -> "辅助评估已完成"
    "failed" -> "本次分析未完成"
    "insufficient_data" -> "有效语音资料不足"
    else -> "服务暂不可用"
}

@Composable
private fun PsychologyAssessmentCard(
    state: UiState<PsychologyOverview>,
    onRetry: () -> Unit,
) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Filled.FavoriteBorder,
                contentDescription = null,
                tint = Primary,
                modifier = Modifier.size(34.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("心理健康评估", style = MaterialTheme.typography.titleLarge)
                Text(
                    "基于近期摄像头面部行为特征",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        when (state) {
            UiState.Loading -> {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(10.dp))
                    Text("正在读取最近一次评估状态")
                }
            }
            is UiState.Error -> {
                Text("心理评估服务暂不可用", style = MaterialTheme.typography.bodyMedium)
                Text(state.message, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                TextButton(onClick = onRetry) { Text("重新加载") }
            }
            is UiState.Success -> PsychologyAssessmentContent(state.data)
        }
    }
}

@Composable
private fun PsychologyAssessmentContent(overview: PsychologyOverview) {
    val stateLabel = when (overview.assessment_state) {
        "collecting" -> "正在采集资料"
        "observation_available" -> "评估已完成"
        "insufficient_data" -> "数据不足"
        else -> "服务暂不可用"
    }
    Text(stateLabel, style = MaterialTheme.typography.titleMedium)
    if (overview.assessment_state == "observation_available") {
        Text("已完成近期心理行为特征综合分析", style = MaterialTheme.typography.bodyMedium)
        overview.estimated_phq8_score?.let { score ->
            Spacer(Modifier.height(12.dp))
            Text("参考评估分数", style = MaterialTheme.typography.titleSmall)
            Text(
                "${"%.1f".format(score)} / 24",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                "最近一次心理评估模型输出",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
            DataRow("辅助风险程度", psychologyRiskLevelLabel(overview.risk_level))
        }
        Spacer(Modifier.height(10.dp))
        if (overview.source_modality == "camera_behavior") {
            Text("分析维度", style = MaterialTheme.typography.titleSmall)
            AnalysisScopeItems()
            Spacer(Modifier.height(8.dp))
        }
        DataRow("数据质量", dataQualityLabel(overview.data_quality))
        overview.updated_at?.takeIf { it.isNotBlank() }?.let {
            DataRow("最近评估时间", formatAssessmentTime(it))
        }
        Spacer(Modifier.height(6.dp))
        Text("健康建议", style = MaterialTheme.typography.titleSmall)
        Text(
            overview.guidance.ifBlank { "结果仅供日常关怀参考，建议结合日常沟通和专业评估" },
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            overview.disclaimer.ifBlank { "该结果仅供日常关怀参考，不构成心理或医疗诊断" },
            style = MaterialTheme.typography.bodySmall,
            color = TextSecondary,
        )
    } else {
        Text(overview.evidence_summary, style = MaterialTheme.typography.bodyMedium)
        if (overview.assessment_state == "collecting") {
            overview.latest_completed?.let { completed ->
                Spacer(Modifier.height(14.dp))
                Text("上一轮辅助评估", style = MaterialTheme.typography.titleSmall)
                Text(
                    "新一轮资料采集中，以下为上一轮已完成结果",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                )
                Spacer(Modifier.height(8.dp))
                PreviousCompletedContent(completed)
            }
        }
    }
}

@Composable
private fun PreviousCompletedContent(completed: PsychologyCompletedReference) {
    completed.estimated_phq8_score?.let { score ->
        Text(
            "算法辅助评分 ${"%.1f".format(score)} / 24",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
        DataRow("辅助风险程度", psychologyRiskLevelLabel(completed.risk_level))
    }
    Text(completed.evidence_summary, style = MaterialTheme.typography.bodyMedium)
    DataRow("数据质量", dataQualityLabel(completed.data_quality))
    completed.updated_at?.takeIf { it.isNotBlank() }?.let {
        DataRow("上一轮完成时间", formatAssessmentTime(it))
    }
    Text(
        completed.disclaimer.ifBlank { "算法评分仅用于辅助评估，不构成心理或医疗诊断" },
        style = MaterialTheme.typography.bodySmall,
        color = TextSecondary,
    )
}

@Composable
private fun AnalysisScopeItems() {
    val items = listOf("面部行为特征", "视线变化", "头部姿态", "面部动作单元")
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items.forEach { item ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(4.dp).background(Primary, RoundedCornerShape(2.dp)),
                )
                Spacer(Modifier.width(8.dp))
                Text(item, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
            }
        }
    }
}

@Composable
private fun DataRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        Spacer(Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
    }
}

private fun dataQualityLabel(quality: String): String = when (quality) {
    "usable" -> "良好"
    "limited" -> "有限"
    "insufficient" -> "不足"
    else -> "未知"
}

private fun psychologyRiskLevelLabel(level: String): String = when (level) {
    "no_risk" -> "暂时无风险"
    "mild" -> "轻度风险"
    "moderate" -> "中度风险"
    "severe" -> "重度风险"
    else -> "暂无法判断"
}

/** 解析 ISO-8601（含毫秒与 Z），转本地时区显示；解析失败时原样返回。 */
private fun formatAssessmentTime(iso: String): String = try {
    val local = java.time.Instant.parse(iso).atZone(java.time.ZoneId.systemDefault())
    java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm").format(local)
} catch (e: Exception) {
    iso
}
