package com.tzb.safeguard.ui.screens.login

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tzb.safeguard.data.repository.SafeRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class LoginUiState(
    val loading: Boolean = false,
    val error: String? = null,
)

class LoginViewModel(private val repository: SafeRepository) : ViewModel() {
    private val _state = MutableStateFlow(LoginUiState())
    val state = _state.asStateFlow()

    fun login(loginName: String, password: String) {
        if (loginName.isBlank() || password.length < 8) {
            _state.value = LoginUiState(error = "请输入账号和至少 8 位密码")
            return
        }
        viewModelScope.launch {
            _state.value = LoginUiState(loading = true)
            repository.login(loginName.trim(), password)
                .onFailure {
                    _state.value = LoginUiState(error = it.message ?: "登录失败")
                }
        }
    }
}
