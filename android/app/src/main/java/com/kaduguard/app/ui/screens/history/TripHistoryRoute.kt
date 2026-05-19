package com.kaduguard.app.ui.screens.history

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.hilt.navigation.compose.hiltViewModel
import com.kaduguard.app.presentation.viewmodel.TripHistoryViewModel

@Composable
fun TripHistoryRoute(
    onBack: () -> Unit,
    viewModel: TripHistoryViewModel = hiltViewModel(),
) {
    val uiState = viewModel.uiState.collectAsState().value

    LaunchedEffect(Unit) {
        viewModel.loadHistory()
    }

    TripHistoryScreen(
        uiState = uiState,
        onBack = onBack,
        onRefresh = viewModel::loadHistory,
    )
}
