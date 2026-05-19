package com.kaduguard.app.ui.screens.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.kaduguard.app.presentation.viewmodel.TripHistoryUiState
import com.kaduguard.app.ui.components.HeroCard
import com.kaduguard.app.ui.components.PolishedBackground
import com.kaduguard.app.ui.components.StatCard

@Composable
fun TripHistoryScreen(
    uiState: TripHistoryUiState,
    onBack: () -> Unit,
    onRefresh: () -> Unit,
) {
    PolishedBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            HeroCard(
                title = "Trip History",
                subtitle = "Summary of your recent drives and risk exposure.",
                tint = MaterialTheme.colorScheme.primary,
            )

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                StatCard(title = "Trips", value = uiState.totalTrips.toString(), accent = MaterialTheme.colorScheme.primary, modifier = Modifier.weight(1f))
                StatCard(title = "Avg Risk", value = "${(uiState.avgRisk * 100).toInt()}%", accent = MaterialTheme.colorScheme.tertiary, modifier = Modifier.weight(1f))
            }
            StatCard(title = "Distance", value = "${uiState.totalDistanceKm} km", accent = MaterialTheme.colorScheme.secondary, modifier = Modifier.fillMaxWidth())

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                Button(onClick = onRefresh, modifier = Modifier.weight(1f), enabled = !uiState.isLoading) {
                    Text(if (uiState.isLoading) "Refreshing..." else "Refresh")
                }
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) {
                    Text("Back")
                }
            }

            if (!uiState.errorMessage.isNullOrBlank()) {
                Card(
                    colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = uiState.errorMessage ?: "",
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.padding(16.dp),
                    )
                }
            }

            if (uiState.isLoading) {
                CircularProgressIndicator()
            }

            if (uiState.trips.isEmpty() && !uiState.isLoading) {
                Text(text = "No trips found yet.")
            }

            uiState.trips.forEach { trip ->
                Card(
                    colors = CardDefaults.elevatedCardColors(),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(text = trip.vehicle_type.uppercase(), style = MaterialTheme.typography.titleMedium)
                        Text(text = "Start: ${trip.start_time}")
                        Text(text = "End: ${trip.end_time ?: "Ongoing"}")
                        Text(text = "Distance: ${trip.distance_km} km")
                        Text(text = "Risk: ${(trip.avg_risk_score * 100).toInt()}% avg / ${(trip.max_risk_score * 100).toInt()}% max")
                        Text(text = "High / Medium / Low: ${trip.high_risk_count} / ${trip.medium_risk_count} / ${trip.low_risk_count}")
                    }
                }
            }
        }
    }
}

@Composable
private fun SummaryCard(title: String, value: String, modifier: Modifier = Modifier) {
    Card(
        colors = CardDefaults.elevatedCardColors(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp),
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = title, style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.padding(top = 4.dp))
            Text(text = value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        }
    }
}
