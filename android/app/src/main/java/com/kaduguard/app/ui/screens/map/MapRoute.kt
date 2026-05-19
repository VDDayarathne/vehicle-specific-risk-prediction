package com.kaduguard.app.ui.screens.map

import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.hilt.navigation.compose.hiltViewModel
import com.kaduguard.app.presentation.viewmodel.MapViewModel

@Composable
fun MapRoute(
    onBack: () -> Unit,
    viewModel: MapViewModel = hiltViewModel(),
) {
    val uiState = viewModel.uiState.collectAsState().value

    LaunchedEffect(Unit) {
        viewModel.loadZones()
    }

    MapScreen(uiState = uiState, onRetry = viewModel::loadZones)
}
