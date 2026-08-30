package com.tzb.safeguard.ui.screens.alerts

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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

/** 消息 Tab 未读角标的单一来源；由已加载事件列表的页面写入（与 unreadCount 同口径：open 且未删除）。 */
object AlertsUnread {
    private val _count = MutableStateFlow(0)
    val count: StateFlow<Int> = _count.asStateFlow()
    fun set(value: Int) {
        _count.value = value
    }
}

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
            val elderId = repo.getCurrentElderId().getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            repo.getEvents(elderId = elderId)
                .onSuccess { data ->
                    _allEvents = data.events.filter { it.type == "fraud_suspected" || it.type == "fall_suspected" }
                    AlertsUnread.set(unreadCount())
                    _state.value = UiState.Success(AlertsData(filter, applyFilter(filter), unreadCount()))
                }
                .onFailure { _state.value = UiState.Error(it.message ?: "加载失败") }
        }
    }

    /** 删除单条消息（后端软删除）；失败时提示但不移除本地条目。 */
    fun deleteEvent(eventId: String) {
        viewModelScope.launch {
            repo.deleteEvent(eventId)
                .onSuccess {
                    _allEvents = _allEvents.filterNot { it.event_id == eventId }
                    AlertsUnread.set(unreadCount())
                    _state.value = UiState.Success(AlertsData(filter, applyFilter(filter), unreadCount()))
                }
                .onFailure {
                    _state.value = UiState.Error(it.message ?: "删除失败")
                }
        }
    }

    private var _allEvents: List<RiskEvent> = emptyList()

    private fun unreadCount(): Int = _allEvents.count { it.status == "open" }

    private fun applyFilter(f: AlertFilter): List<RiskEvent> = when (f) {
        AlertFilter.ALL -> _allEvents.filter { it.verification_status != "retracted" }
        AlertFilter.PREDICTION -> _allEvents.filter {
            it.status == "open" && it.verification_status != "retracted"
        }
        AlertFilter.INTERVENTION -> _allEvents.filter {
            it.status == "acknowledged" && it.verification_status != "retracted"
        }
        AlertFilter.SYSTEM -> _allEvents.filter {
            (it.status == "resolved" || it.status == "false_alarm") &&
                it.verification_status != "retracted"
        }
    }
}
