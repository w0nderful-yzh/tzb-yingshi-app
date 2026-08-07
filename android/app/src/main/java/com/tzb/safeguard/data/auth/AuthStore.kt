package com.tzb.safeguard.data.auth

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class AuthStore(context: Context) {
    private val preferences = context.getSharedPreferences("auth", Context.MODE_PRIVATE)
    private val _accessToken = MutableStateFlow(preferences.getString(TOKEN_KEY, null))
    val accessToken = _accessToken.asStateFlow()

    fun currentToken(): String? = _accessToken.value

    fun save(token: String) {
        preferences.edit().putString(TOKEN_KEY, token).apply()
        _accessToken.value = token
    }

    fun clear() {
        preferences.edit().remove(TOKEN_KEY).apply()
        _accessToken.value = null
    }

    private companion object {
        const val TOKEN_KEY = "access_token"
    }
}
