package com.tzb.safeguard.ui.screens.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.data.model.ElderInfo
import com.tzb.safeguard.data.model.LiveSdkSession
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class HomeData(
    val elder: ElderInfo?,
    val devices: List<Device>,
    val selectedDevice: Device?,
    val liveSession: LiveSdkSession? = null,
    val streamLoading: Boolean = false,
    val streamError: String? = null,
    val pendingWarnings: List<RiskEvent>,
    val recentWarnings: List<RiskEvent>,
)

class HomeViewModel(private val repo: SafeRepository) : ViewModel() {
    private val _state = MutableStateFlow<UiState<HomeData>>(UiState.Loading)
    val state = _state.asStateFlow()

    private val _notice = MutableStateFlow<String?>(null)
    val notice = _notice.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val elders = repo.getElders().getOrElse { return@launch fail(it) }
            val elder = elders.elders.firstOrNull()
                ?: return@launch fail(IllegalStateException("当前账号还没有绑定老人"))
            Session.currentElderId = elder.elder_id
            val devices = repo.getDevices(elder.elder_id).getOrElse { return@launch fail(it) }
            val events = repo.getEvents(elderId = elder.elder_id).getOrElse {
                return@launch fail(it)
            }
            val fraudWarnings = events.events.filter { it.type == "fraud_suspected" }
            val selected = devices.devices.firstOrNull { it.online } ?: devices.devices.firstOrNull()
            _state.value = UiState.Success(
                HomeData(
                    elder = elder,
                    devices = devices.devices,
                    selectedDevice = selected,
                    streamLoading = selected != null,
                    pendingWarnings = fraudWarnings.filter {
                        it.status == "open" || it.status == "acknowledged"
                    },
                    recentWarnings = fraudWarnings.take(3),
                )
            )
            selected?.let { loadLiveSession(it.device_id) }
        }
    }

    fun retryLive() {
        val current = _state.value as? UiState.Success ?: return
        val deviceId = current.data.selectedDevice?.device_id ?: return
        _state.value = UiState.Success(
            current.data.copy(liveSession = null, streamLoading = true, streamError = null)
        )
        viewModelScope.launch { loadLiveSession(deviceId) }
    }

    fun requestHistoryPlayback() {
        val current = _state.value as? UiState.Success ?: return
        val device = current.data.selectedDevice
        if (device == null) {
            _notice.value = "当前没有可用摄像头"
            return
        }
        viewModelScope.launch {
            val elderId = repo.getCurrentElderId().getOrElse {
                _notice.value = it.message ?: "当前账号还没有绑定老人"
                return@launch
            }
            repo.getHistoryPlayback(device.device_id, elderId)
                .onSuccess { _notice.value = "历史回放地址已获取" }
                .onFailure { _notice.value = "历史回放正在接入萤石云端能力，当前暂不可用" }
        }
    }

    fun clearNotice() { _notice.value = null }

    private suspend fun loadLiveSession(deviceId: String) {
        repo.getLiveSdkSession(deviceId)
            .onSuccess { session ->
                val latest = _state.value as? UiState.Success ?: return@onSuccess
                if (latest.data.selectedDevice?.device_id == deviceId) {
                    _state.value = UiState.Success(
                        latest.data.copy(
                            liveSession = session,
                            streamLoading = false,
                            streamError = null,
                        )
                    )
                }
            }
            .onFailure { error ->
                val latest = _state.value as? UiState.Success ?: return@onFailure
                if (latest.data.selectedDevice?.device_id == deviceId) {
                    _state.value = UiState.Success(
                        latest.data.copy(
                            liveSession = null,
                            streamLoading = false,
                            streamError = error.message ?: "获取直播会话失败",
                        )
                    )
                }
            }
    }

    private fun fail(error: Throwable) {
        _state.value = UiState.Error(error.message ?: "加载失败")
    }
}
