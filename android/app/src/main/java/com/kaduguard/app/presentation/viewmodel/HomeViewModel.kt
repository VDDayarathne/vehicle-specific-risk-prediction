package com.kaduguard.app.presentation.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.location.LocationService
import com.kaduguard.app.data.location.LocationTracker
import com.kaduguard.app.data.model.TripEndRequest
import com.kaduguard.app.data.model.TripStartRequest
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.launch

data class HomeUiState(
	val userEmail: String? = null,
	val isLoggingOut: Boolean = false,
	val isOnTrip: Boolean = false,
	val tripId: String? = null,
)

@HiltViewModel
class HomeViewModel @Inject constructor(
	private val tokenStore: AuthTokenStore,
	private val repository: KaduGuardRepository,
) : ViewModel() {
	private val _uiState = MutableStateFlow(HomeUiState())
	val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

	init {
		viewModelScope.launch {
			tokenStore.userEmail.collect { email ->
				_uiState.value = _uiState.value.copy(userEmail = email)
			}
		}
	}

	fun logout(onComplete: () -> Unit = {}) {
		viewModelScope.launch {
			_uiState.value = _uiState.value.copy(isLoggingOut = true)
			tokenStore.clear()
			_uiState.value = _uiState.value.copy(isLoggingOut = false, userEmail = null)
			onComplete()
		}
	}

	fun startTrip(context: Context) {
		viewModelScope.launch {
			val token = tokenStore.getAccessTokenOnce() ?: return@launch
			val loc = LocationTracker.latestLocation.firstOrNull()
			val request = TripStartRequest(
				device_id = "",
				vehicle_type = "car",
				latitude = loc?.latitude ?: 0.0,
				longitude = loc?.longitude ?: 0.0,
			)
			try {
				val res = repository.startTrip(token, request)
				_uiState.value = _uiState.value.copy(isOnTrip = true, tripId = res.trip_id)
				LocationService.start(context)
			} catch (_: Throwable) {
				// ignore
			}
		}
	}

	fun endTrip(context: Context) {
		viewModelScope.launch {
			val token = tokenStore.getAccessTokenOnce() ?: return@launch
			val loc = LocationTracker.latestLocation.firstOrNull()
			val tripId = _uiState.value.tripId ?: return@launch
			val request = TripEndRequest(
				trip_id = tripId,
				latitude = loc?.latitude ?: 0.0,
				longitude = loc?.longitude ?: 0.0,
				distance_km = null,
				duration_minutes = null,
			)
			try {
				repository.endTrip(token, request)
				_uiState.value = _uiState.value.copy(isOnTrip = false, tripId = null)
				LocationService.stop(context)
			} catch (_: Throwable) {
				// ignore
			}
		}
	}
}
