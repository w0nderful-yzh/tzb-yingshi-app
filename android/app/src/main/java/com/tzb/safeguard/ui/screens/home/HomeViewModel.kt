package com.tzb.safeguard.ui.screens.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.model.SafetyStatus
import com.tzb.safeguard.data.model.UserInfo
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** 首页聚合数据：用户 + 安全状态 + 设备 + 最近未结事件 */
data class HomeData(
    val user: UserInfo,
    val status: SafetyStatus,
    val devices: List<Device>,
    val recentEvents: List<RiskEvent>
)

sealed interface SosState {
    data object Idle : SosState
    data object Sending : SosState
    data class Sent(val notifiedContacts: Int) : SosState
    data class Failed(val message: String) : SosState
}

class HomeViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<HomeData>>(UiState.Loading)
    val state = _state.asStateFlow()

    private val _sosState = MutableStateFlow<SosState>(SosState.Idle)
    val sosState = _sosState.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            // 串行加载，任一失败即整体失败并提示重试（数据强相关，避免半残页面）
            val user = repo.getMe().getOrElse { return@launch fail(it) }
            val status = repo.getSafetyStatus().getOrElse { return@launch fail(it) }
            val devices = repo.getDevices().getOrElse { return@launch fail(it) }
            val events = repo.getEvents(status = "open").getOrElse { return@launch fail(it) }
            _state.value = UiState.Success(
                HomeData(user, status, devices.devices, events.events.take(3))
            )
        }
    }

    private fun fail(e: Throwable) {
        _state.value = UiState.Error(e.message ?: "加载失败")
    }

    /** 一键紧急求助：POST /api/v1/sos */
    fun sendSos() {
        if (_sosState.value is SosState.Sending) return
        viewModelScope.launch {
            _sosState.value = SosState.Sending
            repo.sendSos()
                .onSuccess { _sosState.value = SosState.Sent(it.notified_contacts) }
                .onFailure { _sosState.value = SosState.Failed(it.message ?: "求助发送失败") }
        }
    }

    fun resetSos() { _sosState.value = SosState.Idle }
}
