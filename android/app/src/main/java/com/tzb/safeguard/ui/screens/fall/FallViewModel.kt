package com.tzb.safeguard.ui.screens.fall

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.repository.FallRiskRepository
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class FallViewModel(
    private val safeRepository: SafeRepository,
    private val fallRiskRepository: FallRiskRepository,
) : ViewModel() {
    private val _state = MutableStateFlow<UiState<FallRiskOverview>>(UiState.Loading)
    val state = _state.asStateFlow()
    private var pollingJob: Job? = null

    fun startPolling() {
        if (pollingJob?.isActive == true) return
        pollingJob = viewModelScope.launch {
            while (isActive) {
                refresh(showLoading = _state.value !is UiState.Success)
                delay(REFRESH_INTERVAL_MS)
            }
        }
    }

    fun stopPolling() {
        pollingJob?.cancel()
        pollingJob = null
    }

    fun load() {
        viewModelScope.launch {
            refresh(showLoading = true)
        }
    }

    private suspend fun refresh(showLoading: Boolean) {
        if (showLoading) _state.value = UiState.Loading
        val elderId = safeRepository.getCurrentElderId().getOrElse {
            if (_state.value !is UiState.Success) {
                _state.value = UiState.Error(it.message ?: "当前账号还没有绑定老人")
            }
            return
        }
        fallRiskRepository.getOverview(elderId)
            .onSuccess { _state.value = UiState.Success(it) }
            .onFailure {
                if (_state.value !is UiState.Success) {
                    _state.value = UiState.Error(it.message ?: "跌倒风险监测服务暂不可用")
                }
            }
    }

    private companion object {
        const val REFRESH_INTERVAL_MS = 2_500L
    }
}
