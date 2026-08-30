package com.tzb.safeguard.ui.screens.fraud

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FraudViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<List<RiskEvent>>>(UiState.Loading)
    val state = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val elderId = repo.getCurrentElderId().getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            repo.getEvents(elderId = elderId)
                .onSuccess { data ->
                    val fraud = data.events.filter {
                        it.type == "fraud_suspected" && it.verification_status != "retracted"
                    }
                    _state.value = UiState.Success(fraud)
                }
                .onFailure { _state.value = UiState.Error(it.message ?: "加载失败") }
        }
    }
}
