package com.kaduguard.app.data.notifications

import android.content.Context
import android.provider.Settings
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.model.DeviceRegisterRequest
import com.kaduguard.app.domain.repository.KaduGuardRepository
import javax.inject.Inject

class DeviceRegistrationManager @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
) {
    suspend fun registerDevice(context: Context, fcmToken: String) {
        tokenStore.saveFcmToken(fcmToken)

        val accessToken = tokenStore.getAccessTokenOnce() ?: return
        val deviceId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
            ?: "unknown-device"
        val deviceName = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}"

        repository.registerDevice(
            accessToken,
            DeviceRegisterRequest(
                device_id = deviceId,
                fcm_token = fcmToken,
                device_name = deviceName,
            ),
        )
    }

    suspend fun registerStoredToken(context: Context) {
        val fcmToken = tokenStore.getFcmTokenOnce() ?: return
        registerDevice(context, fcmToken)
    }
}