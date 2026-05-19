package com.kaduguard.app.data.network

import com.kaduguard.app.BuildConfig
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.model.RefreshTokenRequest
import com.kaduguard.app.data.model.TokenResponse
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import okhttp3.Authenticator
import okhttp3.Request
import okhttp3.Response
import okhttp3.Route
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

class TokenAuthenticator(
    private val tokenStore: AuthTokenStore,
) : Authenticator {
    override fun authenticate(route: Route?, response: Response): Request? {
        if (responseCount(response) >= 2) return null

        val refreshToken = runBlocking { tokenStore.getRefreshTokenOnce() } ?: return null

        val newTokens = runBlocking {
            val authService = Retrofit.Builder()
                .baseUrl(BuildConfig.BASE_URL)
                .client(OkHttpClient())
                .addConverterFactory(MoshiConverterFactory.create(Moshi.Builder().add(KotlinJsonAdapterFactory()).build()))
                .build()
                .create(RefreshAuthService::class.java)

            runCatching {
                authService.refresh(RefreshTokenRequest(refresh_token = refreshToken))
            }.getOrNull()
        } ?: return null

        runBlocking {
            tokenStore.saveTokens(newTokens.access_token, newTokens.refresh_token)
        }

        return response.request.newBuilder()
            .header("Authorization", "Bearer ${newTokens.access_token}")
            .build()
    }

    private fun responseCount(response: Response): Int {
        var count = 1
        var priorResponse = response.priorResponse
        while (priorResponse != null) {
            count++
            priorResponse = priorResponse.priorResponse
        }
        return count
    }

    private interface RefreshAuthService {
        @POST("api/auth/refresh")
        suspend fun refresh(@Body body: RefreshTokenRequest): TokenResponse
    }
}
