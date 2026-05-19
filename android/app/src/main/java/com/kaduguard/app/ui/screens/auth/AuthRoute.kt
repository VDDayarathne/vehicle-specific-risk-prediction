package com.kaduguard.app.ui.screens.auth

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.presentation.viewmodel.AuthViewModel

@Composable
fun AuthRoute(
    onAuthenticated: () -> Unit,
    viewModel: AuthViewModel = hiltViewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val tokenStore = remember(context) { AuthTokenStore(context.applicationContext) }
    val accessToken by tokenStore.accessToken.collectAsState(initial = null)

    LaunchedEffect(accessToken, uiState.isAuthenticated) {
        if (!accessToken.isNullOrBlank() || uiState.isAuthenticated) {
            onAuthenticated()
        }
    }

    LoginScreen(
        uiState = uiState,
        onEmailChange = viewModel::onEmailChange,
        onPasswordChange = viewModel::onPasswordChange,
        onPhoneChange = viewModel::onPhoneChange,
        onVehicleTypeChange = viewModel::onVehicleTypeChange,
        onSubmit = viewModel::submit,
        onToggleMode = viewModel::toggleMode,
    )
}
