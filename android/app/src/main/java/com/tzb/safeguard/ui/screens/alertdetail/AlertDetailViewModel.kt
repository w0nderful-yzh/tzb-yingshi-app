package com.tzb.safeguard.ui.screens.alertdetail

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.model.EventDetail
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** 处置动作的执行状态 */
sealed interface ActionState {
    data object Idle : ActionState
    data object Running : ActionState
    data class Done(val message: String, val closePage: Boolean = false) : ActionState
    data class Failed(val message: String) : ActionState
}

class AlertDetailViewModel(
    private val repo: SafeRepository,
    private val eventId: String
) : ViewModel() {

    private val _state = MutableStateFlow<UiState<EventDetail>>(UiState.Loading)
    val state = _state.asStateFlow()

    private val _action = MutableStateFlow<ActionState>(ActionState.Idle)
    val action = _action.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            repo.getEventDetail(eventId)
                .onSuccess { _state.value = UiState.Success(it) }
                .onFailure { _state.value = UiState.Error(it.message ?: "加载失败") }
        }
    }

    /** 老人端：我没事 / 我需要帮助 */
    fun confirm(action: String) = runAction(closePage = action == "im_ok") {
        val label = if (action == "im_ok") "已上报：我没事" else "已请求帮助，家人将立即收到电话"
        repo.confirmEvent(eventId, action).map { label }
    }

    /** 家属端处置：acknowledged / resolved / false_alarm */
    fun patchStatus(status: String) = runAction(closePage = status != "acknowledged") {
        val label = when (status) {
            "acknowledged" -> "已知晓"
            "resolved" -> "已标记为处理完成"
            else -> "已标记为误报，将用于降低误报"
        }
        repo.patchEventStatus(eventId, status).map { label }
    }

    /** 家属端一键回呼 */
    fun callElder() = runAction(closePage = false) {
        repo.callElder(eventId).map { "正在呼叫，请稍候…" }
    }

    private fun runAction(closePage: Boolean, block: suspend () -> Result<String>) {
        if (_action.value is ActionState.Running) return
        viewModelScope.launch {
            _action.value = ActionState.Running
            block()
                .onSuccess { _action.value = ActionState.Done(it, closePage) }
                .onFailure { _action.value = ActionState.Failed(it.message ?: "操作失败") }
        }
    }

    fun resetAction() { _action.value = ActionState.Idle }
}
