package com.kaduguard.app.data.repository

import android.content.Context
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
import com.kaduguard.app.data.network.ApiClient
import com.kaduguard.app.domain.repository.KaduGuardRepository

class KaduGuardRepositoryImpl(
    private val context: Context,
) : KaduGuardRepository {
    private val service by lazy { ApiClient.createService(context) }

    override suspend fun register(request: RegisterRequest): TokenResponse = service.register(request)

    override suspend fun login(request: LoginRequest): TokenResponse = service.login(request)

    override suspend fun refresh(request: RefreshTokenRequest): TokenResponse = service.refresh(request)

    override suspend fun me(token: String): UserProfile = service.me("Bearer $token")

    override suspend fun registerDevice(token: String, request: DeviceRegisterRequest): DeviceRegisterResponse =
        service.registerDevice("Bearer $token", request)

    override suspend fun predict(token: String, request: PredictionRequest): PredictionResponse =
        service.predict("Bearer $token", request)

    override suspend fun startTrip(token: String, request: TripStartRequest): TripStartResponse =
        service.startTrip("Bearer $token", request)

    override suspend fun endTrip(token: String, request: TripEndRequest): TripEndResponse =
        service.endTrip("Bearer $token", request)

    override suspend fun tripSummary(token: String): TripHistoryResponse =
        service.tripSummary("Bearer $token")

    override suspend fun riskZones(token: String): RiskZonesResponse =
        service.riskZones("Bearer $token")

    override suspend fun sendTelemetry(token: String, payload: Map<String, Any>): Any =
        service.sendTelemetry("Bearer $token", payload)
}
