package com.tzb.safeguard.ui.screens.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.Session
import com.tzb.safeguard.data.fall.model.FallRiskOverview
import com.tzb.safeguard.data.fall.repository.FallRiskRepository
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.data.model.ElderInfo
import com.tzb.safeguard.data.model.EventListData
import com.tzb.safeguard.data.model.LiveSdkSession
import com.tzb.safeguard.data.model.RiskEvent
import com.tzb.safeguard.data.psychology.model.PsychologyOverview
import com.tzb.safeguard.data.psychology.repository.PsychologyRepository
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import com.tzb.safeguard.ui.screens.alerts.AlertsUnread
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

private const val FALL_REFRESH_INTERVAL_MS = 1_000L

data class HomeData(
    val elder: ElderInfo?,
    val devices: List<Device>,
    val selectedDevice: Device?,
    val liveSession: LiveSdkSession? = null,
    val streamLoading: Boolean = false,
    val streamError: String? = null,
    val pendingWarnings: List<RiskEvent>,
    val recentWarnings: List<RiskEvent>,
    val fallRisk: FallRiskOverview? = null,
    val psychology: UiState<PsychologyOverview> = UiState.Loading,
)

class HomeViewModel(
    private val repo: SafeRepository,
    private val fallRiskRepo: FallRiskRepository,
    private val psychologyRepo: PsychologyRepository,
) : ViewModel() {
    private val _state = MutableStateFlow<UiState<HomeData>>(UiState.Loading)
    val state = _state.asStateFlow()

    private val _notice = MutableStateFlow<String?>(null)
    val notice = _notice.asStateFlow()

    init { load() }

    /** 回到首页时刷新：事件计数与心理卡立即刷新，跌倒卡进入与跌倒页同频的前台轮询。 */
    fun startForegroundRefresh() {
        refreshEvents()
        refreshPsychology()
        if (summaryRefreshJob?.isActive == true) return
        summaryRefreshJob = viewModelScope.launch {
            while (isActive) {
                refreshFallRisk()
                delay(FALL_REFRESH_INTERVAL_MS)
            }
        }
    }

    fun stopForegroundRefresh() {
        summaryRefreshJob?.cancel()
        summaryRefreshJob = null
    }

    private var summaryRefreshJob: Job? = null

    /** 事件列表只影响摘要卡，失败时保持现状，不拖垮首页。 */
    fun refreshEvents() {
        viewModelScope.launch {
            val elderId = repo.getCurrentElderId().getOrElse { return@launch }
            val events = repo.getEvents(elderId = elderId).getOrElse { return@launch }
            val latest = _state.value as? UiState.Success ?: return@launch
            AlertsUnread.set(unreadCount(events))
            _state.value = UiState.Success(
                latest.data.copy(
                    pendingWarnings = fraudWarnings(events).filter {
                        it.status == "open" || it.status == "acknowledged"
                    },
                    recentWarnings = fraudWarnings(events).take(3),
                )
            )
        }
    }

    /** 与消息页 unreadCount 同口径：诈骗+跌倒、open、未删除。 */
    private fun unreadCount(events: EventListData): Int = events.events.count {
        (it.type == "fraud_suspected" || it.type == "fall_suspected") &&
            it.status == "open" && it.verification_status != "retracted"
    }

    private suspend fun refreshFallRisk() {
        val elderId = repo.getCurrentElderId().getOrElse { return }
        fallRiskRepo.getOverview(elderId).onSuccess { overview ->
            val latest = _state.value as? UiState.Success ?: return@onSuccess
            if (latest.data.elder?.elder_id == elderId) {
                _state.value = UiState.Success(latest.data.copy(fallRisk = overview))
            }
        }
    }

    private fun refreshPsychology() {
        viewModelScope.launch {
            val elderId = repo.getCurrentElderId().getOrElse { return@launch }
            loadPsychology(elderId)
        }
    }

    private fun fraudWarnings(events: EventListData): List<RiskEvent> =
        events.events.filter {
            it.type == "fraud_suspected" && it.verification_status != "retracted"
        }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val elders = repo.getElders().getOrElse { return@launch fail(it) }
            val elder = elders.elders.firstOrNull()
                ?: return@launch fail(IllegalStateException("当前账号还没有绑定老人"))
            Session.currentElderId = elder.elder_id
            val devices = repo.getDevices(elder.elder_id).getOrElse { return@launch fail(it) }
            // 事件为附属数据：失败只降级摘要卡，不再把整个首页打成 Error。
            val events = repo.getEvents(elderId = elder.elder_id).getOrNull()
            val fraudEvents = events?.let { fraudWarnings(it) } ?: emptyList()
            val selected = devices.devices.firstOrNull { it.online } ?: devices.devices.firstOrNull()
            _state.value = UiState.Success(
                HomeData(
                    elder = elder,
                    devices = devices.devices,
                    selectedDevice = selected,
                    streamLoading = selected != null,
                    pendingWarnings = fraudEvents.filter {
                        it.status == "open" || it.status == "acknowledged"
                    },
                    recentWarnings = fraudEvents.take(3),
                )
            )
            launch { loadFallRisk(elder.elder_id) }
            launch { loadPsychology(elder.elder_id) }
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

    private suspend fun loadFallRisk(elderId: String) {
        fallRiskRepo.getOverview(elderId).onSuccess { overview ->
            val latest = _state.value as? UiState.Success ?: return@onSuccess
            if (latest.data.elder?.elder_id == elderId) {
                _state.value = UiState.Success(latest.data.copy(fallRisk = overview))
            }
        }
    }

    private suspend fun loadPsychology(elderId: String) {
        psychologyRepo.getOverview(elderId)
            .onSuccess { overview ->
                val latest = _state.value as? UiState.Success ?: return@onSuccess
                if (latest.data.elder?.elder_id == elderId) {
                    _state.value = UiState.Success(
                        latest.data.copy(psychology = UiState.Success(overview))
                    )
                }
            }
            .onFailure { error ->
                val latest = _state.value as? UiState.Success ?: return@onFailure
                if (latest.data.elder?.elder_id == elderId) {
                    _state.value = UiState.Success(
                        latest.data.copy(
                            psychology = UiState.Error(error.message ?: "心理健康评估服务暂不可用")
                        )
                    )
                }
            }
    }

    private fun fail(error: Throwable) {
        _state.value = UiState.Error(error.message ?: "加载失败")
    }
}
