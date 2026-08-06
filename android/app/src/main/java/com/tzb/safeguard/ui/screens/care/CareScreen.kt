package com.tzb.safeguard.ui.screens.care

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.PersonalInjury
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.tzb.safeguard.ui.components.AppCard
import com.tzb.safeguard.ui.theme.*

/** 其他团队的模块接入占位，不生成任何本地模拟结论。 */
@Composable
fun CareScreen(navController: NavHostController) {
    Scaffold(modifier = Modifier.statusBarsPadding(), containerColor = BgPage) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            Row(Modifier.fillMaxWidth().padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { navController.popBackStack() }) {
                    Icon(Icons.Filled.ArrowBack, contentDescription = "返回")
                }
                Text("后续守护能力", style = MaterialTheme.typography.headlineLarge)
            }
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                PlaceholderModule(
                    Icons.Filled.PersonalInjury,
                    "跌倒风险预测",
                    "待跌倒模块通过统一风险事件接口接入。未来应关注失衡、步态与环境风险趋势，而不只报告已经跌倒。"
                )
                PlaceholderModule(
                    Icons.Filled.FavoriteBorder,
                    "心理健康趋势",
                    "待心理关怀模块定义非诊断性趋势与人工复核流程。当前版本不生成情绪评分、诊断或陪聊内容。"
                )
                Text(
                    "两个模块接入后复用用户、设备、事件状态和家属处置能力，不进入防诈状态机。",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }
    }
}

@Composable
private fun PlaceholderModule(icon: androidx.compose.ui.graphics.vector.ImageVector, title: String, detail: String) {
    AppCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(34.dp))
            Spacer(Modifier.width(12.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleLarge)
                Text("待模块接入", style = MaterialTheme.typography.bodySmall, color = WarnAmber)
            }
        }
        Spacer(Modifier.height(10.dp))
        Text(detail, style = MaterialTheme.typography.bodyMedium)
    }
}
