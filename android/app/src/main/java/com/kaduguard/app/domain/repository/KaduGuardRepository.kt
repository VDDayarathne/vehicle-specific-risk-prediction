package com.kaduguard.app.domain.repository

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

interface KaduGuardRepository {
    suspend fun register(request: RegisterRequest): TokenResponse
    suspend fun login(request: LoginRequest): TokenResponse
    suspend fun refresh(request: RefreshTokenRequest): TokenResponse
    suspend fun me(token: String): UserProfile
    suspend fun registerDevice(token: String, request: DeviceRegisterRequest): DeviceRegisterResponse
    suspend fun predict(token: String, request: PredictionRequest): PredictionResponse
    suspend fun startTrip(token: String, request: TripStartRequest): TripStartResponse
    suspend fun endTrip(token: String, request: TripEndRequest): TripEndResponse
    suspend fun tripSummary(token: String): TripHistoryResponse
    suspend fun riskZones(token: String): RiskZonesResponse
    suspend fun sendTelemetry(token: String, payload: Map<String, Any>): Any
}
