package com.tzb.safeguard.ui.screens.profile

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.model.Contact
import com.tzb.safeguard.data.model.Device
import com.tzb.safeguard.data.model.UserInfo
import com.tzb.safeguard.data.repository.SafeRepository
import com.tzb.safeguard.ui.components.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ProfileData(
    val user: UserInfo,
    val contacts: List<Contact>,
    val devices: List<Device>
)

class ProfileViewModel(private val repo: SafeRepository) : ViewModel() {

    private val _state = MutableStateFlow<UiState<ProfileData>>(UiState.Loading)
    val state = _state.asStateFlow()

    init { load() }

    fun load() {
        viewModelScope.launch {
            _state.value = UiState.Loading
            val user = repo.getMe().getOrElse { return@launch fail(it) }
            // 联系人与设备失败不阻断整页：降级为空列表，由页面展示空态
            val contacts = repo.getContacts().getOrNull()?.contacts ?: emptyList()
            val devices = repo.getDevices().getOrNull()?.devices ?: emptyList()
            _state.value = UiState.Success(ProfileData(user, contacts, devices))
        }
    }

    private fun fail(e: Throwable) {
        _state.value = UiState.Error(e.message ?: "加载失败")
    }
}
