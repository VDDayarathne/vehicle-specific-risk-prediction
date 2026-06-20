package com.kaduguard.app.data.model

data class LoginRequest(
    val email: String,
    val password: String,
)

data class RegisterRequest(
    val phone: String? = null,
    val email: String,
    val password: String,
    val vehicle_type: String,
)

data class TokenResponse(
    val access_token: String,
    val refresh_token: String,
    val token_type: String,
    val expires_in: Int,
)

data class PredictionRequest(
    val vehicle_type: String,
    val temperature_c: Double? = null,
    val humidity_pct: Double? = null,
    val rainfall_mm: Double? = null,
    val wind_speed_kmh: Double? = null,
    val visibility_km: Double? = null,
    val gradient_pct: Double? = null,
    val curvature: Double? = null,
    val road_surface: String = "asphalt",
    val lane_count: Int = 2,
    val hour_of_day: Int? = null,
    val day_of_week: Int? = null,
    val is_holiday: Boolean = false,
    val traffic_density: Double = 0.5,
    val device_id: String? = null,
    val latitude: Double? = null,
    val longitude: Double? = null,
    val speed_kmh: Double? = null,
    val heading_deg: Double? = null,
    val accel_m_s2: Double? = null,
    val rpm: Int? = null,
)

data class PredictionResponse(
    val risk_level: String,
    val risk_score: Double,
    val probabilities: Map<String, Double>,
    val vehicle_type: String,
    val recommended_speed_kmh: Int?,
    val alert_message: String,
)
