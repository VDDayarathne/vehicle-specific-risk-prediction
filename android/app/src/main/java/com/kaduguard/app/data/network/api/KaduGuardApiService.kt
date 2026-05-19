package com.kaduguard.app.data.network.api

import com.kaduguard.app.data.model.DeviceRegisterRequest
import com.kaduguard.app.data.model.DeviceRegisterResponse
import com.kaduguard.app.data.model.LoginRequest
import com.kaduguard.app.data.model.PredictionRequest
import com.kaduguard.app.data.model.PredictionResponse
import com.kaduguard.app.data.model.RegisterRequest
import com.kaduguard.app.data.model.RefreshTokenRequest
import com.kaduguard.app.data.model.RiskZonesResponse
import com.kaduguard.app.data.model.TokenResponse
import com.kaduguard.app.data.model.TripEndRequest
import com.kaduguard.app.data.model.TripEndResponse
import com.kaduguard.app.data.model.TripHistoryResponse
import com.kaduguard.app.data.model.TripStartRequest
import com.kaduguard.app.data.model.TripStartResponse
import com.kaduguard.app.data.model.UserProfile
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST

interface KaduGuardApiService {
    @POST("api/auth/register")
    suspend fun register(@Body body: RegisterRequest): TokenResponse

    @POST("api/auth/login")
    suspend fun login(@Body body: LoginRequest): TokenResponse

    @POST("api/auth/refresh")
    suspend fun refresh(@Body body: RefreshTokenRequest): TokenResponse

    @POST("api/auth/device-register")
    suspend fun registerDevice(
        @Header("Authorization") authorization: String,
        @Body body: DeviceRegisterRequest,
    ): DeviceRegisterResponse

    @GET("api/auth/me")
    suspend fun me(@Header("Authorization") authorization: String): UserProfile

    @POST("api/mobile/predict")
    suspend fun predict(
        @Header("Authorization") authorization: String,
        @Body body: PredictionRequest,
    ): PredictionResponse

    @POST("api/mobile/trip/start")
    suspend fun startTrip(
        @Header("Authorization") authorization: String,
        @Body body: TripStartRequest,
    ): TripStartResponse

    @POST("api/mobile/trip/end")
    suspend fun endTrip(
        @Header("Authorization") authorization: String,
        @Body body: TripEndRequest,
    ): TripEndResponse

    @GET("api/mobile/trip-summary")
    suspend fun tripSummary(
        @Header("Authorization") authorization: String,
    ): TripHistoryResponse

    @GET("api/mobile/risk-zones")
    suspend fun riskZones(
        @Header("Authorization") authorization: String,
    ): RiskZonesResponse

    @POST("api/telemetry")
    suspend fun sendTelemetry(
        @Header("Authorization") authorization: String,
        @Body body: Map<String, @JvmSuppressWildcards Any>,
    ): Any
}
