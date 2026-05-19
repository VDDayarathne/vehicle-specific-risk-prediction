package com.kaduguard.app.data.model

data class RefreshTokenRequest(
    val refresh_token: String,
)

data class DeviceRegisterRequest(
    val device_id: String,
    val fcm_token: String,
    val device_name: String? = null,
)

data class DeviceRegisterResponse(
    val status: String,
    val device_id: String,
    val message: String? = null,
)

data class UserProfile(
    val driver_id: String,
    val email: String,
    val phone: String? = null,
    val vehicle_type: String,
    val created_at: String,
    val is_active: Boolean,
)

data class TripStartRequest(
    val device_id: String,
    val vehicle_type: String,
    val latitude: Double,
    val longitude: Double,
)

data class TripStartResponse(
    val trip_id: String,
    val status: String,
    val message: String? = null,
)

data class TripEndRequest(
    val trip_id: String,
    val latitude: Double,
    val longitude: Double,
    val distance_km: Double? = null,
    val duration_minutes: Int? = null,
)

data class TripEndResponse(
    val trip_id: String,
    val status: String,
    val avg_risk_score: Double? = null,
    val max_risk_score: Double? = null,
    val high_risk_count: Int? = null,
)

data class TripSummaryItem(
    val trip_id: String,
    val start_time: String,
    val end_time: String? = null,
    val start_lat: Double,
    val start_lon: Double,
    val end_lat: Double? = null,
    val end_lon: Double? = null,
    val distance_km: Double,
    val duration_minutes: Int? = null,
    val avg_risk_score: Double,
    val max_risk_score: Double,
    val high_risk_count: Int,
    val medium_risk_count: Int,
    val low_risk_count: Int,
    val vehicle_type: String,
)

data class TripHistoryResponse(
    val trips: List<TripSummaryItem>,
    val total_trips: Int,
    val avg_risk: Double,
    val total_distance_km: Double,
)

data class RiskZone(
    val segment_id: String,
    val name: String,
    val start_lat: Double,
    val start_lon: Double,
    val end_lat: Double,
    val end_lon: Double,
    val gradient_pct: Double,
    val curvature: Double,
    val base_risk_level: String,
)

data class RiskZonesResponse(
    val zones: List<RiskZone>,
    val count: Int,
)
