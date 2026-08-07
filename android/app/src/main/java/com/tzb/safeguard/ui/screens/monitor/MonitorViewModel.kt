package com.tzb.safeguard.ui.screens.monitor

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.data.model.LiveSdkSession
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** 现场复核页数据：仅包含真实设备和萤石直播会话。 */
data class MonitorData(
    val devices: List<Device>,
    val selected: Device?,
    val liveSession: LiveSdkSession?,
    val streamLoading: Boolean = false,
    val streamError: String? = null
)

class MonitorViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<MonitorData>>(UiState.Loading)
    val state = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val elderId = repo.getCurrentElderId().getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            val devices = repo.getDevices(elderId).getOrElse {
                _state.value = UiState.Error(it.message ?: "加载失败")
                return@launch
            }
            val online = devices.devices.firstOrNull { it.online } ?: devices.devices.firstOrNull()
            _state.value = UiState.Success(buildData(devices.devices, online))
            online?.let { loadLiveSession(it.device_id) }
        }
    }

    /** 切换摄像头 */
    fun select(device: Device) {
        val current = _state.value
        if (current is UiState.Success) {
            _state.value = UiState.Success(
                current.data.copy(
                    selected = device,
                    liveSession = null,
                    streamLoading = true,
                    streamError = null,
                )
            )
            viewModelScope.launch { loadLiveSession(device.device_id) }
        }
    }

    fun retryLive() {
        val current = _state.value
        if (current is UiState.Success) {
            val deviceId = current.data.selected?.device_id ?: return
            _state.value = UiState.Success(
                current.data.copy(liveSession = null, streamLoading = true, streamError = null)
            )
            viewModelScope.launch { loadLiveSession(deviceId) }
        }
    }

    private suspend fun loadLiveSession(deviceId: String) {
        val current = _state.value
        if (current is UiState.Success) {
            repo.getLiveSdkSession(deviceId)
                .onSuccess { session ->
                    val latest = _state.value
                    if (latest is UiState.Success && latest.data.selected?.device_id == deviceId) {
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
                    val latest = _state.value
                    if (latest is UiState.Success && latest.data.selected?.device_id == deviceId) {
                        _state.value = UiState.Success(
                            latest.data.copy(
                                liveSession = null,
                                streamLoading = false,
                                streamError = error.message ?: "获取直播地址失败",
                            )
                        )
                    }
                }
            // 会话获取失败不阻断页面：画面区保留重试入口
        }
    }

    private fun buildData(devices: List<Device>, selected: Device?): MonitorData {
        return MonitorData(
            devices = devices,
            selected = selected,
            liveSession = null,
            streamLoading = selected != null
        )
    }
}
