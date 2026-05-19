package com.kaduguard.app.ui.screens.home

import android.Manifest
import android.content.pm.PackageManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import com.kaduguard.app.data.location.LocationService
import com.kaduguard.app.presentation.viewmodel.HomeViewModel

@Composable
fun HomeRoute(
    onOpenPrediction: () -> Unit,
    onOpenMap: () -> Unit,
    onOpenHistory: () -> Unit,
    onLogout: () -> Unit,
    viewModel: HomeViewModel = hiltViewModel(),
) {
    val context = LocalContext.current
    val uiState = viewModel.uiState.collectAsState().value
    val canTrackLocation = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    HomeScreen(
        userEmail = uiState.userEmail,
        isLoggingOut = uiState.isLoggingOut,
        isTrackingLocation = canTrackLocation,
        onOpenPrediction = onOpenPrediction,
        onStartLocationTracking = { viewModel.startTrip(context) },
        onStopLocationTracking = { viewModel.endTrip(context) },
        onOpenMap = onOpenMap,
        onOpenHistory = onOpenHistory,
        onLogout = { viewModel.logout(onLogout) },
    )
}
