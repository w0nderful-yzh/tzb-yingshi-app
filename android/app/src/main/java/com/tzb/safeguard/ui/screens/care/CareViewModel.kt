package com.tzb.safeguard.ui.screens.care

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import com.tzb.safeguard.data.psychology.repository.PsychologyRepository
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class CareViewModel(
    private val safeRepository: SafeRepository,
    private val psychologyRepository: PsychologyRepository,
) : ViewModel() {
    private val _psychologyState = MutableStateFlow<UiState<PsychologyOverview>>(UiState.Loading)
    val psychologyState = _psychologyState.asStateFlow()

    init {
        loadPsychologyOverview()
    }

    fun loadPsychologyOverview() {
        viewModelScope.launch {
            _psychologyState.value = UiState.Loading
            val elderId = safeRepository.getCurrentElderId().getOrElse {
                _psychologyState.value = UiState.Error(it.message ?: "当前账号还没有绑定老人")
                return@launch
            }
            psychologyRepository.getOverview(elderId)
                .onSuccess { _psychologyState.value = UiState.Success(it) }
                .onFailure {
                    _psychologyState.value = UiState.Error(it.message ?: "心理健康评估服务暂不可用")
                }
        }
    }
}

