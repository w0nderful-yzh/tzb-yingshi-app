package com.tzb.safeguard

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import com.tzb.safeguard.ui.navigation.AppNavHost
import com.tzb.safeguard.ui.screens.login.LoginScreen
import com.tzb.safeguard.ui.theme.SafeGuardTheme

class MainActivity : ComponentActivity() {
    private val pendingEventId = mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingEventId.value = intent.getStringExtra(EXTRA_EVENT_ID)
        enableEdgeToEdge()
        setContent {
            SafeGuardTheme {
                val accessToken by ServiceLocator.authStore.accessToken.collectAsState()
                key(accessToken) {
                    if (accessToken == null) {
                        LoginScreen()
                    } else {
                        AppNavHost(
                            openEventId = pendingEventId.value,
                            onOpenEventConsumed = { pendingEventId.value = null },
                        )
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        pendingEventId.value = intent.getStringExtra(EXTRA_EVENT_ID)
    }

    companion object {
        const val EXTRA_EVENT_ID = "event_id"
    }
}
