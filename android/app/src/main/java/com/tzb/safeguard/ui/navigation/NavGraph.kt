package com.tzb.safeguard.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.tzb.safeguard.ui.screens.alertdetail.AlertDetailScreen
import com.tzb.safeguard.ui.screens.alerts.AlertsScreen
import com.tzb.safeguard.ui.screens.care.CareScreen
import com.tzb.safeguard.ui.screens.home.HomeScreen
import com.tzb.safeguard.ui.screens.monitor.MonitorScreen
import com.tzb.safeguard.ui.screens.profile.ProfileScreen

/** 路由集中定义，避免散落硬编码 */
object Routes {
    const val HOME = "home"
    const val MONITOR = "monitor"
    const val ALERTS = "alerts"
    const val CARE = "care"
    const val PROFILE = "profile"
    const val ALERT_DETAIL = "alert_detail/{eventId}"

    fun alertDetail(eventId: String) = "alert_detail/$eventId"
}

/** 简化 ViewModel 创建：不使用 Hilt，直接从 ServiceLocator 注入仓库 */
@Composable
inline fun <reified VM : ViewModel> appViewModel(crossinline factory: () -> VM): VM =
    viewModel(factory = viewModelFactory {
        initializer { factory() }
    })

@Composable
fun AppNavHost(
    navController: NavHostController = rememberNavController(),
    openEventId: String? = null,
    onOpenEventConsumed: () -> Unit = {},
) {
    LaunchedEffect(openEventId) {
        if (!openEventId.isNullOrBlank()) {
            navController.navigate(Routes.alertDetail(openEventId)) { launchSingleTop = true }
            onOpenEventConsumed()
        }
    }
    NavHost(navController = navController, startDestination = Routes.HOME) {
        composable(Routes.HOME) { HomeScreen(navController) }
        composable(Routes.MONITOR) { MonitorScreen(navController) }
        composable(Routes.ALERTS) { AlertsScreen(navController) }
        composable(Routes.CARE) { CareScreen(navController) }
        composable(Routes.PROFILE) { ProfileScreen(navController) }

        // 预警详情
        composable(
            route = Routes.ALERT_DETAIL,
            arguments = listOf(navArgument("eventId") { type = NavType.StringType })
        ) { backStackEntry ->
            val eventId = backStackEntry.arguments?.getString("eventId").orEmpty()
            AlertDetailScreen(navController, eventId)
        }
    }
}
