package com.tzb.safeguard.ui.screens.role

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Elderly
import androidx.compose.material.icons.filled.FamilyRestroom
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import com.tzb.safeguard.Session
import com.tzb.safeguard.ui.components.BigActionButton
import com.tzb.safeguard.ui.navigation.Routes
import com.tzb.safeguard.ui.theme.TextSecondary

/**
 * 启动页：选择使用身份（老人端 / 家属端）。
 * 正式版应由登录账号角色自动进入，此处为联调期手动入口。
 */
@Composable
fun RoleSelectScreen(navController: NavHostController) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("守护安", style = MaterialTheme.typography.headlineLarge)
        Spacer(Modifier.height(6.dp))
        Text(
            "居家安全守护 · 防诈骗 · 防跌倒",
            style = MaterialTheme.typography.bodyMedium,
            color = TextSecondary,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(48.dp))

        BigActionButton(text = "我是老人（本人使用）", icon = Icons.Filled.Elderly) {
            Session.role = "elder"
            navController.navigate(Routes.HOME) {
                popUpTo(Routes.ROLE) { inclusive = true }
            }
        }
        Spacer(Modifier.height(16.dp))
        BigActionButton(
            text = "我是家属（远程守护）",
            icon = Icons.Filled.FamilyRestroom,
            outlined = true
        ) {
            Session.role = "family"
            navController.navigate(Routes.FAMILY) {
                popUpTo(Routes.ROLE) { inclusive = true }
            }
        }
    }
}
