package com.kaduguard.app.data.notifications

import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

@AndroidEntryPoint
class KaduGuardMessagingService : FirebaseMessagingService() {
    @Inject lateinit var deviceRegistrationManager: DeviceRegistrationManager

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        val title = remoteMessage.notification?.title
            ?: remoteMessage.data["title"]
            ?: "KaduGuard Alert"
        val body = remoteMessage.notification?.body
            ?: remoteMessage.data["body"]
            ?: "A new risk alert is available."

        NotificationHelper.showAlert(
            context = applicationContext,
            title = title,
            body = body,
        )
    }

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        serviceScope.launch {
            try {
                deviceRegistrationManager.registerDevice(applicationContext, token)
            } catch (_: Throwable) {
                // Best-effort re-registration.
            }
        }
    }
}