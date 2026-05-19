package com.kaduguard.app.ui.screens.map

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.google.maps.android.compose.CameraPositionState
import com.google.maps.android.compose.GoogleMap
import com.google.maps.android.compose.Marker
import com.google.maps.android.compose.MarkerState
import com.google.maps.android.compose.Polyline
import com.google.maps.android.compose.rememberCameraPositionState
import com.google.android.gms.maps.model.LatLng
import com.kaduguard.app.presentation.viewmodel.MapUiState

@Composable
fun MapScreen(
    uiState: MapUiState,
    onRetry: () -> Unit,
) {
    val cameraPositionState: CameraPositionState = rememberCameraPositionState()

    Box(modifier = Modifier.fillMaxSize()) {
        GoogleMap(
            modifier = Modifier.fillMaxSize(),
            cameraPositionState = cameraPositionState,
        ) {
            uiState.zones.forEach { zone ->
                val start = LatLng(zone.startLat, zone.startLon)
                val end = LatLng(zone.endLat, zone.endLon)
                // marker at start
                Marker(state = MarkerState(position = start), title = zone.name)
                // simple polyline between start and end
                Polyline(points = listOf(start, end))
            }
        }

        if (uiState.isLoading) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
        }

        uiState.errorMessage?.let { msg ->
            Text(text = msg, color = MaterialTheme.colorScheme.error, modifier = Modifier.align(Alignment.TopCenter))
        }
    }
}
