package com.kaduguard.app.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.model.TripHistoryResponse
import com.kaduguard.app.data.model.TripSummaryItem
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.launch

data class TripHistoryUiState(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val totalTrips: Int = 0,
    val avgRisk: Double = 0.0,
    val totalDistanceKm: Double = 0.0,
    val trips: List<TripSummaryItem> = emptyList(),
)

@HiltViewModel
class TripHistoryViewModel @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
) : ViewModel() {
    private val _uiState = MutableStateFlow(TripHistoryUiState())
    val uiState: StateFlow<TripHistoryUiState> = _uiState.asStateFlow()

    fun loadHistory() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            try {
                val token = tokenStore.getAccessTokenOnce()
                if (token.isNullOrBlank()) {
                    _uiState.value = _uiState.value.copy(isLoading = false, errorMessage = "Not authenticated")
                    return@launch
                }
                val resp: TripHistoryResponse = repository.tripSummary(token)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    totalTrips = resp.total_trips,
                    avgRisk = resp.avg_risk,
                    totalDistanceKm = resp.total_distance_km,
                    trips = resp.trips,
                )
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(isLoading = false, errorMessage = t.message ?: "Failed to load history")
            }
        }
    }
}
