package com.kaduguard.app.ui.screens.prediction

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.kaduguard.app.presentation.viewmodel.PredictionUiState
import com.kaduguard.app.ui.components.HeroCard
import com.kaduguard.app.ui.components.PolishedBackground
import com.kaduguard.app.ui.components.ScreenSection

@Composable
fun PredictionScreen(
    uiState: PredictionUiState,
    onBack: () -> Unit,
    onPredictNow: () -> Unit,
    onToggleAutoRefresh: () -> Unit,
    onVehicleTypeChange: (String) -> Unit,
    onTemperatureChange: (String) -> Unit,
    onHumidityChange: (String) -> Unit,
    onRainfallChange: (String) -> Unit,
    onWindSpeedChange: (String) -> Unit,
    onVisibilityChange: (String) -> Unit,
    onGradientChange: (String) -> Unit,
    onCurvatureChange: (String) -> Unit,
    onRoadSurfaceChange: (String) -> Unit,
    onLaneCountChange: (String) -> Unit,
    onHourOfDayChange: (String) -> Unit,
    onDayOfWeekChange: (String) -> Unit,
    onHolidayChange: (Boolean) -> Unit,
    onTrafficDensityChange: (String) -> Unit,
    onDeviceIdChange: (String) -> Unit,
    onLatitudeChange: (String) -> Unit,
    onLongitudeChange: (String) -> Unit,
    onSpeedChange: (String) -> Unit,
    onHeadingChange: (String) -> Unit,
    onAccelChange: (String) -> Unit,
    onRpmChange: (String) -> Unit,
) {
    val riskColor = when (uiState.riskLevel) {
        "High" -> Color(0xFFE63946)
        "Medium" -> Color(0xFFF4A261)
        "Low" -> Color(0xFF2A9D8F)
        else -> MaterialTheme.colorScheme.primary
    }

    PolishedBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            HeroCard(
                title = "Live Prediction",
                subtitle = uiState.lastUpdatedLabel
                    ?: "Tap Predict Now or let auto-refresh keep updating.",
                tint = riskColor,
            )

            Card(
                colors = CardDefaults.elevatedCardColors(),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 6.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(text = "Auto refresh")
                        Switch(
                            checked = uiState.isAutoRefreshEnabled,
                            onCheckedChange = { onToggleAutoRefresh() })
                    }
                    Button(
                        onClick = onPredictNow,
                        enabled = !uiState.isLoading,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        if (uiState.isLoading) {
                            CircularProgressIndicator(
                                modifier = Modifier.width(18.dp),
                                strokeWidth = 2.dp
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text("Predicting...")
                        } else {
                            Text("Predict Now")
                        }
                    }
                    Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) {
                        Text("Back")
                    }
                }
            }

            uiState.riskLevel?.let { risk ->
                Card(
                    colors = CardDefaults.elevatedCardColors(),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 6.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(text = "Current Risk", style = MaterialTheme.typography.titleMedium)
                        Text(
                            text = risk,
                            style = MaterialTheme.typography.headlineLarge,
                            color = riskColor
                        )
                        LinearProgressIndicator(
                            progress = ((uiState.riskScore ?: 0.0).coerceIn(0.0, 1.0)).toFloat(),
                            color = riskColor,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(text = "Risk score: ${(uiState.riskScore ?: 0.0).times(100).toInt()}%")
                        Text(text = uiState.alertMessage ?: "")
                        Text(text = "Recommended speed: ${uiState.recommendedSpeedKmh ?: "N/A"} km/h")
                    }
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

            ScreenSection(title = "Prediction Inputs") {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    PredictionField("Vehicle type", uiState.vehicleType, onVehicleTypeChange)
                    PredictionField(
                        "Temperature (°C)",
                        uiState.temperatureC,
                        onTemperatureChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Humidity (%)",
                        uiState.humidityPct,
                        onHumidityChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Rainfall (mm)",
                        uiState.rainfallMm,
                        onRainfallChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Wind speed (km/h)",
                        uiState.windSpeedKmh,
                        onWindSpeedChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Visibility (km)",
                        uiState.visibilityKm,
                        onVisibilityChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Gradient (%)",
                        uiState.gradientPct,
                        onGradientChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Curvature",
                        uiState.curvature,
                        onCurvatureChange,
                        KeyboardType.Decimal
                    )
                    PredictionField("Road surface", uiState.roadSurface, onRoadSurfaceChange)
                    PredictionField(
                        "Lane count",
                        uiState.laneCount,
                        onLaneCountChange,
                        KeyboardType.Number
                    )
                    PredictionField(
                        "Hour of day",
                        uiState.hourOfDay,
                        onHourOfDayChange,
                        KeyboardType.Number
                    )
                    PredictionField(
                        "Day of week",
                        uiState.dayOfWeek,
                        onDayOfWeekChange,
                        KeyboardType.Number
                    )
                    PredictionField(
                        "Traffic density",
                        uiState.trafficDensity,
                        onTrafficDensityChange,
                        KeyboardType.Decimal
                    )
                    PredictionField("Device ID", uiState.deviceId, onDeviceIdChange)
                    PredictionField(
                        "Latitude",
                        uiState.latitude,
                        onLatitudeChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Longitude",
                        uiState.longitude,
                        onLongitudeChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Speed (km/h)",
                        uiState.speedKmh,
                        onSpeedChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Heading (deg)",
                        uiState.headingDeg,
                        onHeadingChange,
                        KeyboardType.Decimal
                    )
                    PredictionField(
                        "Acceleration",
                        uiState.accelMs2,
                        onAccelChange,
                        KeyboardType.Decimal
                    )
                    PredictionField("RPM", uiState.rpm, onRpmChange, KeyboardType.Number)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Switch(checked = uiState.isHoliday, onCheckedChange = onHolidayChange)
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(text = "Holiday")
                    }
                }
            }

            if (uiState.probabilities.isNotEmpty()) {
                Card(
                    colors = CardDefaults.elevatedCardColors(),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(
                            text = "Class probabilities",
                            style = MaterialTheme.typography.titleMedium
                        )
                        uiState.probabilities.forEach { (label, probability) ->
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(text = "$label: ${(probability * 100).toInt()}%")
                                LinearProgressIndicator(
                                    progress = probability.toFloat().coerceIn(0f, 1f),
                                    modifier = Modifier.fillMaxWidth(),
                                    color = when (label) {
                                        "High" -> Color(0xFFE63946)
                                        "Medium" -> Color(0xFFF4A261)
                                        else -> Color(0xFF2A9D8F)
                                    },
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PredictionField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    keyboardType: KeyboardType = KeyboardType.Text,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
    )
}
