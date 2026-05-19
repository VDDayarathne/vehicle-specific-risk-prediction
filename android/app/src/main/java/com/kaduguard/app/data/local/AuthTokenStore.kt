package com.kaduguard.app.data.local

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.authDataStore by preferencesDataStore(name = "auth_store")

class AuthTokenStore(private val context: Context) {
    companion object {
        private val ACCESS_TOKEN = stringPreferencesKey("access_token")
        private val REFRESH_TOKEN = stringPreferencesKey("refresh_token")
        private val USER_EMAIL = stringPreferencesKey("user_email")
        private val FCM_TOKEN = stringPreferencesKey("fcm_token")
    }

    val accessToken: Flow<String?> = context.authDataStore.data.map { it[ACCESS_TOKEN] }
    val refreshToken: Flow<String?> = context.authDataStore.data.map { it[REFRESH_TOKEN] }
    val userEmail: Flow<String?> = context.authDataStore.data.map { it[USER_EMAIL] }
    val fcmToken: Flow<String?> = context.authDataStore.data.map { it[FCM_TOKEN] }

    suspend fun saveTokens(accessToken: String, refreshToken: String) {
        context.authDataStore.edit { preferences ->
            preferences[ACCESS_TOKEN] = accessToken
            preferences[REFRESH_TOKEN] = refreshToken
        }
    }

    suspend fun saveUserEmail(email: String) {
        context.authDataStore.edit { preferences ->
            preferences[USER_EMAIL] = email
        }
    }

    suspend fun saveFcmToken(token: String) {
        context.authDataStore.edit { preferences ->
            preferences[FCM_TOKEN] = token
        }
    }

    suspend fun clear() {
        context.authDataStore.edit { preferences ->
            preferences.remove(ACCESS_TOKEN)
            preferences.remove(REFRESH_TOKEN)
            preferences.remove(USER_EMAIL)
            preferences.remove(FCM_TOKEN)
        }
    }

    suspend fun getAccessTokenOnce(): String? = accessToken.first()
    suspend fun getRefreshTokenOnce(): String? = refreshToken.first()
    suspend fun getFcmTokenOnce(): String? = fcmToken.first()
}
