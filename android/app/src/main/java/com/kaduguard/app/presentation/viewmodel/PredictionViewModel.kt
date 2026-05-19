package com.kaduguard.app.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.model.PredictionRequest
import com.kaduguard.app.data.model.PredictionResponse
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class PredictionUiState(
    val vehicleType: String = "car",
    val temperatureC: String = "26",
    val humidityPct: String = "78",
    val rainfallMm: String = "5",
    val windSpeedKmh: String = "15",
    val visibilityKm: String = "7",
    val gradientPct: String = "8",
    val curvature: String = "0.4",
    val roadSurface: String = "asphalt",
    val laneCount: String = "2",
    val hourOfDay: String = "14",
    val dayOfWeek: String = "2",
    val isHoliday: Boolean = false,
    val trafficDensity: String = "0.5",
    val deviceId: String = "",
    val latitude: String = "7.2500",
    val longitude: String = "80.5333",
    val speedKmh: String = "35",
    val headingDeg: String = "140",
    val accelMs2: String = "0.0",
    val rpm: String = "2200",
    val isLoading: Boolean = false,
    val isAutoRefreshEnabled: Boolean = true,
    val lastUpdatedLabel: String? = null,
    val riskLevel: String? = null,
    val riskScore: Double? = null,
    val probabilities: Map<String, Double> = emptyMap(),
    val recommendedSpeedKmh: Int? = null,
    val alertMessage: String? = null,
    val errorMessage: String? = null,
)

@HiltViewModel
class PredictionViewModel @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
) : ViewModel() {
    private val _uiState = MutableStateFlow(PredictionUiState())
    val uiState: StateFlow<PredictionUiState> = _uiState.asStateFlow()

    private var autoRefreshJob: Job? = null

    fun onVehicleTypeChange(value: String) = updateState { it.copy(vehicleType = value, errorMessage = null) }
    fun onTemperatureChange(value: String) = updateState { it.copy(temperatureC = value, errorMessage = null) }
    fun onHumidityChange(value: String) = updateState { it.copy(humidityPct = value, errorMessage = null) }
    fun onRainfallChange(value: String) = updateState { it.copy(rainfallMm = value, errorMessage = null) }
    fun onWindSpeedChange(value: String) = updateState { it.copy(windSpeedKmh = value, errorMessage = null) }
    fun onVisibilityChange(value: String) = updateState { it.copy(visibilityKm = value, errorMessage = null) }
    fun onGradientChange(value: String) = updateState { it.copy(gradientPct = value, errorMessage = null) }
    fun onCurvatureChange(value: String) = updateState { it.copy(curvature = value, errorMessage = null) }
    fun onRoadSurfaceChange(value: String) = updateState { it.copy(roadSurface = value, errorMessage = null) }
    fun onLaneCountChange(value: String) = updateState { it.copy(laneCount = value, errorMessage = null) }
    fun onHourOfDayChange(value: String) = updateState { it.copy(hourOfDay = value, errorMessage = null) }
    fun onDayOfWeekChange(value: String) = updateState { it.copy(dayOfWeek = value, errorMessage = null) }
    fun onHolidayChange(value: Boolean) = updateState { it.copy(isHoliday = value, errorMessage = null) }
    fun onTrafficDensityChange(value: String) = updateState { it.copy(trafficDensity = value, errorMessage = null) }
    fun onDeviceIdChange(value: String) = updateState { it.copy(deviceId = value, errorMessage = null) }
    fun onLatitudeChange(value: String) = updateState { it.copy(latitude = value, errorMessage = null) }
    fun onLongitudeChange(value: String) = updateState { it.copy(longitude = value, errorMessage = null) }
    fun onSpeedChange(value: String) = updateState { it.copy(speedKmh = value, errorMessage = null) }
    fun onHeadingChange(value: String) = updateState { it.copy(headingDeg = value, errorMessage = null) }
    fun onAccelChange(value: String) = updateState { it.copy(accelMs2 = value, errorMessage = null) }
    fun onRpmChange(value: String) = updateState { it.copy(rpm = value, errorMessage = null) }

    fun toggleAutoRefresh() {
        val enabled = !_uiState.value.isAutoRefreshEnabled
        updateState { it.copy(isAutoRefreshEnabled = enabled) }
        if (enabled) {
            startAutoRefresh()
        } else {
            stopAutoRefresh()
        }
    }

    fun startAutoRefresh() {
        if (autoRefreshJob?.isActive == true) return
        autoRefreshJob = viewModelScope.launch {
            while (isActive && _uiState.value.isAutoRefreshEnabled) {
                refreshPrediction()
                delay(30_000L)
            }
        }
    }

    fun stopAutoRefresh() {
        autoRefreshJob?.cancel()
        autoRefreshJob = null
    }

    suspend fun loadInitialPrediction() {
        if (_uiState.value.riskLevel == null) {
            refreshPrediction()
        }
    }

    fun refreshPrediction() {
        viewModelScope.launch {
            val token = tokenStore.getAccessTokenOnce()
            if (token.isNullOrBlank()) {
                updateState { it.copy(errorMessage = "You are not signed in") }
                return@launch
            }

            val request = buildRequest() ?: return@launch

            updateState { it.copy(isLoading = true, errorMessage = null) }
            try {
                val result = repository.predict(token, request)
                updateState {
                    it.copy(
                        isLoading = false,
                        lastUpdatedLabel = "Updated just now",
                        riskLevel = result.risk_level,
                        riskScore = result.risk_score,
                        probabilities = result.probabilities,
                        recommendedSpeedKmh = result.recommended_speed_kmh,
                        alertMessage = result.alert_message,
                        errorMessage = null,
                    )
                }
            } catch (throwable: Throwable) {
                updateState {
                    it.copy(
                        isLoading = false,
                        errorMessage = throwable.message ?: "Prediction request failed",
                    )
                }
            }
        }
    }

    private fun buildRequest(): PredictionRequest? {
        val state = _uiState.value
        return try {
            PredictionRequest(
                vehicle_type = state.vehicleType,
                temperature_c = state.temperatureC.toDouble(),
                humidity_pct = state.humidityPct.toDouble(),
                rainfall_mm = state.rainfallMm.toDouble(),
                wind_speed_kmh = state.windSpeedKmh.toDouble(),
                visibility_km = state.visibilityKm.toDouble(),
                gradient_pct = state.gradientPct.toDouble(),
                curvature = state.curvature.toDouble(),
                road_surface = state.roadSurface,
                lane_count = state.laneCount.toInt(),
                hour_of_day = state.hourOfDay.toInt(),
                day_of_week = state.dayOfWeek.toInt(),
                is_holiday = state.isHoliday,
                traffic_density = state.trafficDensity.toDouble(),
                device_id = state.deviceId.ifBlank { null },
                latitude = state.latitude.toDoubleOrNull(),
                longitude = state.longitude.toDoubleOrNull(),
                speed_kmh = state.speedKmh.toDoubleOrNull(),
                heading_deg = state.headingDeg.toDoubleOrNull(),
                accel_m_s2 = state.accelMs2.toDoubleOrNull(),
                rpm = state.rpm.toIntOrNull(),
            )
        } catch (throwable: Throwable) {
            updateState { it.copy(errorMessage = "Please check the input values") }
            null
        }
    }

    private fun updateState(transform: (PredictionUiState) -> PredictionUiState) {
        _uiState.value = transform(_uiState.value)
    }
}
