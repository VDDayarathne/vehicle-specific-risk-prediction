package com.kaduguard.app.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class RiskZoneItem(
    val segmentId: String,
    val name: String,
    val startLat: Double,
    val startLon: Double,
    val endLat: Double,
    val endLon: Double,
    val baseRiskLevel: String,
)

data class MapUiState(
    val zones: List<RiskZoneItem> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
)

@HiltViewModel
class MapViewModel @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
) : ViewModel() {
    private val _uiState = MutableStateFlow(MapUiState())
    val uiState: StateFlow<MapUiState> = _uiState.asStateFlow()

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
                        baseRiskLevel = it.base_risk_level,
                    )
                }
                _uiState.value = _uiState.value.copy(zones = items, isLoading = false)
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(isLoading = false, errorMessage = t.message ?: "Failed to load zones")
            }
        }
    }
}
