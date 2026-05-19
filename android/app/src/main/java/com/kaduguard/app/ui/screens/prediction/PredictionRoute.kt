package com.kaduguard.app.ui.screens.prediction

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.hilt.navigation.compose.hiltViewModel
import com.kaduguard.app.presentation.viewmodel.PredictionViewModel

@Composable
fun PredictionRoute(
    onBack: () -> Unit,
    viewModel: PredictionViewModel = hiltViewModel(),
) {
    val uiState = viewModel.uiState.collectAsState().value

    LaunchedEffect(Unit) {
        viewModel.startAutoRefresh()
        viewModel.loadInitialPrediction()
    }

    DisposableEffect(Unit) {
        onDispose {
            viewModel.stopAutoRefresh()
        }
    }

    PredictionScreen(
        uiState = uiState,
        onBack = onBack,
        onPredictNow = viewModel::refreshPrediction,
        onToggleAutoRefresh = viewModel::toggleAutoRefresh,
        onVehicleTypeChange = viewModel::onVehicleTypeChange,
        onTemperatureChange = viewModel::onTemperatureChange,
        onHumidityChange = viewModel::onHumidityChange,
        onRainfallChange = viewModel::onRainfallChange,
        onWindSpeedChange = viewModel::onWindSpeedChange,
        onVisibilityChange = viewModel::onVisibilityChange,
        onGradientChange = viewModel::onGradientChange,
        onCurvatureChange = viewModel::onCurvatureChange,
        onRoadSurfaceChange = viewModel::onRoadSurfaceChange,
        onLaneCountChange = viewModel::onLaneCountChange,
        onHourOfDayChange = viewModel::onHourOfDayChange,
        onDayOfWeekChange = viewModel::onDayOfWeekChange,
        onHolidayChange = viewModel::onHolidayChange,
        onTrafficDensityChange = viewModel::onTrafficDensityChange,
        onDeviceIdChange = viewModel::onDeviceIdChange,
        onLatitudeChange = viewModel::onLatitudeChange,
        onLongitudeChange = viewModel::onLongitudeChange,
        onSpeedChange = viewModel::onSpeedChange,
        onHeadingChange = viewModel::onHeadingChange,
        onAccelChange = viewModel::onAccelChange,
        onRpmChange = viewModel::onRpmChange,
    )
}
