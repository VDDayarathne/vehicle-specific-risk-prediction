package com.kaduguard.app.presentation.viewmodel

import android.content.Context
import android.provider.Settings
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.model.LoginRequest
import com.kaduguard.app.data.model.RegisterRequest
import com.kaduguard.app.domain.repository.KaduGuardRepository
import com.google.firebase.messaging.FirebaseMessaging
import com.kaduguard.app.data.notifications.DeviceRegistrationManager
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.launch
import retrofit2.HttpException

enum class AuthMode {
    Login,
    Register,
}

data class AuthUiState(
    val mode: AuthMode = AuthMode.Login,
    val email: String = "",
    val password: String = "",
    val phone: String = "",
    val vehicleType: String = "car",
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isAuthenticated: Boolean = false,
    val savedEmail: String? = null,
)

@HiltViewModel
class AuthViewModel @Inject constructor(
    private val repository: KaduGuardRepository,
    private val tokenStore: AuthTokenStore,
    private val deviceRegistrationManager: DeviceRegistrationManager,
    @ApplicationContext private val context: Context,
) : ViewModel() {
    private val _uiState = MutableStateFlow(AuthUiState())
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val email = tokenStore.userEmail.firstOrNull()
            if (!email.isNullOrBlank()) {
                _uiState.value = _uiState.value.copy(email = email, savedEmail = email)
            }
        }
    }

    fun onEmailChange(value: String) {
        updateState { it.copy(email = value, errorMessage = null) }
    }

    fun onPasswordChange(value: String) {
        updateState { it.copy(password = value, errorMessage = null) }
    }

    fun onPhoneChange(value: String) {
        updateState { it.copy(phone = value, errorMessage = null) }
    }

    fun onVehicleTypeChange(value: String) {
        updateState { it.copy(vehicleType = value, errorMessage = null) }
    }

    fun toggleMode() {
        updateState {
            it.copy(
                mode = if (it.mode == AuthMode.Login) AuthMode.Register else AuthMode.Login,
                errorMessage = null,
            )
        }
    }

    fun submit() {
        val state = _uiState.value
        val email = state.email.trim()
        val password = state.password

        if (email.isBlank() || password.isBlank()) {
            updateState { it.copy(errorMessage = "Email and password are required") }
            return
        }

        if (state.mode == AuthMode.Register && state.vehicleType.isBlank()) {
            updateState { it.copy(errorMessage = "Vehicle type is required") }
            return
        }

        viewModelScope.launch {
            updateState { it.copy(isLoading = true, errorMessage = null) }
            try {
                val mode = state.mode
                Log.d("AuthViewModel", "Starting ${mode.name} with email: $email")
                
                val tokens = when (mode) {
                    AuthMode.Login -> {
                        Log.d("AuthViewModel", "Attempting login...")
                        repository.login(LoginRequest(email = email, password = password))
                    }
                    AuthMode.Register -> {
                        Log.d("AuthViewModel", "Attempting registration with vehicle_type: ${state.vehicleType}")
                        repository.register(
                            RegisterRequest(
                                phone = state.phone.ifBlank { null },
                                email = email,
                                password = password,
                                vehicle_type = state.vehicleType,
                            )
                        )
                    }
                }
                
                Log.d("AuthViewModel", "${mode.name} successful! Got tokens")
                tokenStore.saveTokens(tokens.access_token, tokens.refresh_token)
                tokenStore.saveUserEmail(email)
                registerDeviceToken()
                updateState { it.copy(isLoading = false, isAuthenticated = true) }
            } catch (httpException: HttpException) {
                val errorBody = httpException.response()?.errorBody()?.string() ?: "Unknown error"
                val errorMessage = "Server Error (${httpException.code()}): $errorBody"
                Log.e("AuthViewModel", "HTTP Error: $errorMessage", httpException)
                updateState {
                    it.copy(
                        isLoading = false,
                        errorMessage = errorMessage,
                    )
                }
            } catch (throwable: Throwable) {
                val errorMessage = "Error: ${throwable.message ?: "Unknown error"}"
                Log.e("AuthViewModel", errorMessage, throwable)
                updateState {
                    it.copy(
                        isLoading = false,
                        errorMessage = errorMessage,
                    )
                }
            }
        }
    }

    private fun updateState(transform: (AuthUiState) -> AuthUiState) {
        _uiState.value = transform(_uiState.value)
    }

    private fun registerDeviceToken() {
        FirebaseMessaging.getInstance().token
            .addOnSuccessListener { fcmToken ->
                viewModelScope.launch {
                    try {
                        deviceRegistrationManager.registerDevice(context, fcmToken)
                    } catch (_: Throwable) {
                        // Device registration is best-effort and should not block login.
                    }
                }
            }
    }
}
