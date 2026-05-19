package com.kaduguard.app.ui.screens.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.kaduguard.app.ui.components.HeroCard
import com.kaduguard.app.ui.components.PolishedBackground
import com.kaduguard.app.ui.components.StatCard

@Composable
fun HomeScreen(
    userEmail: String? = null,
    isLoggingOut: Boolean = false,
    isTrackingLocation: Boolean = false,
    onOpenPrediction: () -> Unit = {},
    onOpenMap: () -> Unit = {},
    onStartLocationTracking: () -> Unit = {},
    onStopLocationTracking: () -> Unit = {},
    onOpenHistory: () -> Unit = {},
    onLogout: () -> Unit = {},
) {
    PolishedBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            HeroCard(
                title = "Live Risk Dashboard",
                subtitle = userEmail?.let { "Signed in as $it" } ?: "Track trips, inspect danger zones, and get alerts before risk spikes.",
                tint = MaterialTheme.colorScheme.primary,
            )

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                StatCard(title = "Status", value = if (isTrackingLocation) "Tracking" else "Idle", accent = MaterialTheme.colorScheme.tertiary, modifier = Modifier.weight(1f))
                StatCard(title = "Mode", value = "Driver", accent = MaterialTheme.colorScheme.secondary, modifier = Modifier.weight(1f))
            }

            Button(onClick = onOpenPrediction, modifier = Modifier.fillMaxWidth()) { Text(text = "Open Live Prediction") }
            Button(onClick = onOpenMap, modifier = Modifier.fillMaxWidth()) { Text(text = "Open Risk Map") }
            Button(onClick = onOpenHistory, modifier = Modifier.fillMaxWidth()) { Text(text = "Trip History") }
            Button(
                onClick = onStartLocationTracking,
                enabled = !isTrackingLocation,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = if (isTrackingLocation) "GPS Tracking Active" else "Start GPS Tracking")
            }
            Button(
                onClick = onStopLocationTracking,
                enabled = isTrackingLocation,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = "Stop GPS Tracking")
            }
            Button(
                onClick = onLogout,
                enabled = !isLoggingOut,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(text = if (isLoggingOut) "Signing out..." else "Sign Out")
            }
        }
    }
}
