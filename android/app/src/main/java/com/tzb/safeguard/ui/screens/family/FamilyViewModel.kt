package com.tzb.safeguard.ui.screens.family

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.ActivityData
import com.tzb.safeguard.data.model.ElderInfo
import com.tzb.safeguard.data.model.EventsStatsData
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** 家属端看板聚合数据 */
data class FamilyData(
    val elder: ElderInfo?,
    val pendingEvents: List<RiskEvent>,
    val eventsStats: EventsStatsData,
    val activity: ActivityData
)

class FamilyViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<FamilyData>>(UiState.Loading)
    val state = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            // 四个请求相互独立，并发加载；统计失败不拖垮整页
            val eldersDeferred = async { repo.getElders() }
            val pendingDeferred = async {
                repo.getEvents(elderId = Session.currentElderId, status = "open")
            }
            val statsDeferred = async { repo.getEventsStats(elderId = Session.currentElderId, days = 30) }
            val activityDeferred = async { repo.getActivityStats(elderId = Session.currentElderId, days = 7) }

            val elders = eldersDeferred.await().getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            val pending = pendingDeferred.await().getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            // 图表数据降级：失败时给空集合，页面展示空态而不是整页报错
            val stats = statsDeferred.await().getOrNull() ?: EventsStatsData()
            val activity = activityDeferred.await().getOrNull() ?: ActivityData()

            _state.value = UiState.Success(
                FamilyData(elders.elders.firstOrNull(), pending.events, stats, activity)
            )
        }
    }
}
