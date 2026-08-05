package com.tzb.safeguard.ui.screens.alerts

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** 消息中心筛选条件 */
enum class AlertFilter(val label: String, val level: String?, val status: String?) {
    ALL("全部", null, null),
    EMERGENCY("紧急", "emergency", null),
    WARNING("警告", "warning", null),
    REMINDER("提醒", "reminder", null),
    DONE("已处理", null, "resolved")
}

data class AlertsData(
    val filter: AlertFilter,
    val events: List<RiskEvent>,
    val unreadCount: Int
)

class AlertsViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<AlertsData>>(UiState.Loading)
    val state = _state.asStateFlow()

    private var filter = AlertFilter.ALL

    init { load() }

    fun selectFilter(f: AlertFilter) {
        filter = f
        load()
    }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            // 家属端必须显式传 elder_id（服务端校验绑定关系）
            val elderId = if (Session.role == "family") Session.currentElderId else null
            repo.getEvents(elderId = elderId, level = filter.level, status = filter.status)
                .onSuccess { data ->
                    val unread = data.events.count { it.status == "open" }
                    _state.value = UiState.Success(AlertsData(filter, data.events, unread))
                }
                .onFailure { _state.value = UiState.Error(it.message ?: "加载失败") }
        }
    }
}
