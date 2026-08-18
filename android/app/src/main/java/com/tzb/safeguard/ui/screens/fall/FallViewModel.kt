package com.tzb.safeguard.ui.screens.fall

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.repository.FallRiskRepository
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FallViewModel(
    private val safeRepository: SafeRepository,
    private val fallRiskRepository: FallRiskRepository,
) : ViewModel() {
    private val _state = MutableStateFlow<UiState<FallRiskOverview>>(UiState.Loading)
    val state = _state.asStateFlow()

    init {
        load()
    }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val elderId = safeRepository.getCurrentElderId().getOrElse {
                _state.value = UiState.Error(it.message ?: "当前账号还没有绑定老人")
                return@launch
            }
            fallRiskRepository.getOverview(elderId)
                .onSuccess { _state.value = UiState.Success(it) }
                .onFailure {
                    _state.value = UiState.Error(it.message ?: "跌倒风险监测服务暂不可用")
                }
        }
    }
}
