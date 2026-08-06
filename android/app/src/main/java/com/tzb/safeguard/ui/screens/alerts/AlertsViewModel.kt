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

/** 风险趋势按介入状态筛选，不在一级信息中拆分诈骗场景。 */
enum class AlertFilter(val label: String) {
    ALL("全部"),
    PREDICTION("预测"),
    INTERVENTION("介入"),
    SYSTEM("系统")
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
            repo.getEvents(elderId = Session.currentElderId)
                .onSuccess { data ->
                    val fraudEvents = data.events.filter { it.type == "fraud_suspected" }
                    val visible = when (filter) {
                        AlertFilter.ALL -> fraudEvents
                        AlertFilter.PREDICTION -> fraudEvents.filter { it.status == "open" }
                        AlertFilter.INTERVENTION -> fraudEvents.filter { it.status == "acknowledged" }
                        AlertFilter.SYSTEM -> fraudEvents.filter {
                            it.status == "resolved" || it.status == "false_alarm"
                        }
                    }
                    val unread = fraudEvents.count { it.status == "open" }
                    _state.value = UiState.Success(AlertsData(filter, visible, unread))
                }
                .onFailure { _state.value = UiState.Error(it.message ?: "加载失败") }
        }
    }
}
