package com.kaduguard.app.presentation.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.location.LocationService
import com.kaduguard.app.data.location.LocationSnapshot
import com.kaduguard.app.data.location.LocationTracker
import com.kaduguard.app.data.model.PredictionRequest
import com.kaduguard.app.data.model.TripEndRequest
import com.kaduguard.app.data.model.TripStartRequest
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import java.time.LocalDateTime
import javax.inject.Inject
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

val DRIVING_VEHICLE_TYPES = listOf("car", "van", "bus", "lorry", "motorcycle")

data class RiskZoneItem(
    val segmentId: String,
    val name: String,
    val startLat: Double,
    val startLon: Double,
    val endLat: Double,
    val endLon: Double,
    val gradientPct: Double,
    val curvature: Double,
    val baseRiskLevel: String,
)

data class DrivingPoint(
    val latitude: Double,
    val longitude: Double,
)

data class MapUiState(
    val zones: List<RiskZoneItem> = emptyList(),
    val vehicleType: String = DRIVING_VEHICLE_TYPES.first(),
    val currentLocation: LocationSnapshot? = null,
    val routePoints: List<DrivingPoint> = emptyList(),
    val tripId: String? = null,
    val isDriving: Boolean = false,
    val isStartingTrip: Boolean = false,
    val isEndingTrip: Boolean = false,
    val isLoading: Boolean = false,
    val riskLevel: String? = null,
    val riskScore: Double? = null,
    val recommendedSpeedKmh: Int? = null,
    val alertMessage: String? = null,
    val nearestZoneName: String? = null,
    val distanceKm: Double = 0.0,
    val startedAtMillis: Long? = null,
    val lastUpdatedLabel: String? = null,
    val errorMessage: String? = null,
)

@HiltViewModel
class MapViewModel @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
) : ViewModel() {
    private val _uiState = MutableStateFlow(MapUiState())
    val uiState: StateFlow<MapUiState> = _uiState.asStateFlow()

    private var predictionJob: Job? = null
    private var tripStartJob: Job? = null

    init {
        viewModelScope.launch {
            LocationTracker.latestLocation.collect { location ->
                val previous = _uiState.value.currentLocation
                val shouldAppend = location != null && _uiState.value.isDriving
                val distanceDelta = if (shouldAppend && previous != null) {
                    distanceBetweenKm(previous.latitude, previous.longitude, location!!.latitude, location.longitude)
                } else {
                    0.0
                }

                _uiState.value = _uiState.value.copy(
                    currentLocation = location,
                    routePoints = if (shouldAppend) {
                        _uiState.value.routePoints + DrivingPoint(location!!.latitude, location.longitude)
                    } else {
                        _uiState.value.routePoints
                    },
                    distanceKm = _uiState.value.distanceKm + distanceDelta,
                    errorMessage = if (location != null && _uiState.value.errorMessage == WAITING_FOR_LOCATION) null else _uiState.value.errorMessage,
                )

                if (location != null && _uiState.value.isStartingTrip && _uiState.value.tripId == null) {
                    startTripAtLocation(location)
                }
            }
        }
    }

    fun loadZones() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            try {
                val token = tokenStore.getAccessTokenOnce() ?: run {
                    _uiState.value = _uiState.value.copy(isLoading = false, errorMessage = "Not authenticated")
                    return@launch
                }
                val resp = repository.riskZones(token)
                val items = resp.zones.map {
                    RiskZoneItem(
                        segmentId = it.segment_id,
                        name = it.name,
                        startLat = it.start_lat,
                        startLon = it.start_lon,
                        endLat = it.end_lat,
                        endLon = it.end_lon,
                        gradientPct = it.gradient_pct,
                        curvature = it.curvature,
                        baseRiskLevel = it.base_risk_level,
                    )
                }
                _uiState.value = _uiState.value.copy(zones = items, isLoading = false)
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(isLoading = false, errorMessage = t.message ?: "Failed to load zones")
            }
        }
    }

    fun onVehicleTypeSelected(value: String) {
        if (_uiState.value.isDriving) return
        _uiState.value = _uiState.value.copy(vehicleType = value, errorMessage = null)
    }

    fun startDriving(context: Context) {
        if (_uiState.value.isDriving || _uiState.value.isStartingTrip) return
        _uiState.value = _uiState.value.copy(
            isStartingTrip = true,
            errorMessage = WAITING_FOR_LOCATION,
            routePoints = emptyList(),
            distanceKm = 0.0,
            riskLevel = null,
            riskScore = null,
            recommendedSpeedKmh = null,
            alertMessage = null,
            nearestZoneName = null,
            lastUpdatedLabel = null,
        )
        LocationService.start(context.applicationContext, _uiState.value.vehicleType)
        _uiState.value.currentLocation?.let { startTripAtLocation(it) }
    }

    fun stopDriving(context: Context) {
        val tripId = _uiState.value.tripId
        val location = _uiState.value.currentLocation
        if (tripId == null || location == null) {
            resetDrivingState()
            LocationService.stop(context.applicationContext)
            return
        }

        viewModelScope.launch {
            val token = tokenStore.getAccessTokenOnce()
            if (token.isNullOrBlank()) {
                _uiState.value = _uiState.value.copy(errorMessage = "You are not signed in")
                return@launch
            }

            _uiState.value = _uiState.value.copy(isEndingTrip = true, errorMessage = null)
            try {
                repository.endTrip(
                    token,
                    TripEndRequest(
                        trip_id = tripId,
                        latitude = location.latitude,
                        longitude = location.longitude,
                        distance_km = _uiState.value.distanceKm,
                        duration_minutes = tripDurationMinutes(),
                    ),
                )
                resetDrivingState()
                LocationService.stop(context.applicationContext)
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(
                    isEndingTrip = false,
                    errorMessage = t.message ?: "Failed to end trip",
                )
            }
        }
    }

    private fun startTripAtLocation(location: LocationSnapshot) {
        if (tripStartJob?.isActive == true) return
        tripStartJob = viewModelScope.launch {
            val token = tokenStore.getAccessTokenOnce()
            if (token.isNullOrBlank()) {
                _uiState.value = _uiState.value.copy(isStartingTrip = false, errorMessage = "You are not signed in")
                return@launch
            }

            try {
                val response = repository.startTrip(
                    token,
                    TripStartRequest(
                        device_id = "",
                        vehicle_type = _uiState.value.vehicleType,
                        latitude = location.latitude,
                        longitude = location.longitude,
                    ),
                )
                _uiState.value = _uiState.value.copy(
                    tripId = response.trip_id,
                    isDriving = true,
                    isStartingTrip = false,
                    startedAtMillis = System.currentTimeMillis(),
                    routePoints = listOf(DrivingPoint(location.latitude, location.longitude)),
                    errorMessage = null,
                )
                startPredictionLoop()
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(
                    isStartingTrip = false,
                    errorMessage = t.message ?: "Failed to start trip",
                )
            }
        }
    }

    private fun startPredictionLoop() {
        if (predictionJob?.isActive == true) return
        predictionJob = viewModelScope.launch {
            while (isActive && _uiState.value.isDriving) {
                refreshDrivingRisk()
                delay(PREDICTION_INTERVAL_MS)
            }
        }
    }

    private suspend fun refreshDrivingRisk() {
        val token = tokenStore.getAccessTokenOnce() ?: return
        val location = _uiState.value.currentLocation ?: return
        val nearestZone = nearestZone(location)
        val now = LocalDateTime.now()

        try {
            val result = repository.predict(
                token,
                PredictionRequest(
                    vehicle_type = _uiState.value.vehicleType,
                    temperature_c = 25.0,
                    humidity_pct = 70.0,
                    rainfall_mm = 0.0,
                    wind_speed_kmh = 0.0,
                    visibility_km = 10.0,
                    gradient_pct = nearestZone?.gradientPct,
                    curvature = nearestZone?.curvature,
                    hour_of_day = now.hour,
                    day_of_week = now.dayOfWeek.value % 7,
                    device_id = null,
                    latitude = location.latitude,
                    longitude = location.longitude,
                    speed_kmh = location.speedKmh,
                    heading_deg = location.bearingDeg?.toDouble(),
                ),
            )
            _uiState.value = _uiState.value.copy(
                riskLevel = result.risk_level,
                riskScore = result.risk_score,
                recommendedSpeedKmh = result.recommended_speed_kmh,
                alertMessage = result.alert_message,
                nearestZoneName = nearestZone?.name,
                lastUpdatedLabel = "Updated just now",
                errorMessage = null,
            )
        } catch (t: Throwable) {
            _uiState.value = _uiState.value.copy(errorMessage = t.message ?: "Risk prediction failed")
        }
    }

    private fun nearestZone(location: LocationSnapshot): RiskZoneItem? {
        return _uiState.value.zones.minByOrNull { zone ->
            val startDistance = distanceBetweenKm(location.latitude, location.longitude, zone.startLat, zone.startLon)
            val endDistance = distanceBetweenKm(location.latitude, location.longitude, zone.endLat, zone.endLon)
            minOf(startDistance, endDistance)
        }
    }

    private fun tripDurationMinutes(): Int? {
        val startedAt = _uiState.value.startedAtMillis ?: return null
        return ((System.currentTimeMillis() - startedAt) / 60_000L).toInt().coerceAtLeast(1)
    }

    private fun resetDrivingState() {
        predictionJob?.cancel()
        predictionJob = null
        _uiState.value = _uiState.value.copy(
            tripId = null,
            isDriving = false,
            isStartingTrip = false,
            isEndingTrip = false,
            startedAtMillis = null,
        )
    }

    override fun onCleared() {
        predictionJob?.cancel()
        super.onCleared()
    }

    private fun distanceBetweenKm(startLat: Double, startLon: Double, endLat: Double, endLon: Double): Double {
        val earthRadiusKm = 6371.0
        val dLat = Math.toRadians(endLat - startLat)
        val dLon = Math.toRadians(endLon - startLon)
        val lat1 = Math.toRadians(startLat)
        val lat2 = Math.toRadians(endLat)
        val a = sin(dLat / 2).pow(2.0) + sin(dLon / 2).pow(2.0) * cos(lat1) * cos(lat2)
        val c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return earthRadiusKm * c
    }

    companion object {
        private const val PREDICTION_INTERVAL_MS = 15_000L
        private const val WAITING_FOR_LOCATION = "Waiting for GPS fix"
    }
}
