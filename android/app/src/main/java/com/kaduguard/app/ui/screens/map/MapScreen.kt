package com.kaduguard.app.ui.screens.map

import android.annotation.SuppressLint
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.kaduguard.app.presentation.viewmodel.DRIVING_VEHICLE_TYPES
import com.kaduguard.app.presentation.viewmodel.MapUiState
import org.json.JSONArray
import org.json.JSONObject

@Composable
fun MapScreen(
    uiState: MapUiState,
    onRetry: () -> Unit,
    onVehicleTypeSelected: (String) -> Unit,
    onStartDriving: () -> Unit,
    onStopDriving: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize()) {
        OpenStreetMapView(uiState = uiState, modifier = Modifier.fillMaxSize())

        if (uiState.isLoading) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
        }

        DrivingControlSheet(
            uiState = uiState,
            onRetry = onRetry,
            onVehicleTypeSelected = onVehicleTypeSelected,
            onStartDriving = onStartDriving,
            onStopDriving = onStopDriving,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
        )
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
private fun OpenStreetMapView(uiState: MapUiState, modifier: Modifier = Modifier) {
    val html = buildLeafletHtml(uiState)

    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                )
                webViewClient = WebViewClient()
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                loadDataWithBaseURL(MAP_BASE_URL, html, "text/html", "UTF-8", null)
            }
        },
        update = { webView ->
            webView.loadDataWithBaseURL(MAP_BASE_URL, html, "text/html", "UTF-8", null)
        },
    )
}

@Composable
@OptIn(ExperimentalLayoutApi::class)
private fun DrivingControlSheet(
    uiState: MapUiState,
    onRetry: () -> Unit,
    onVehicleTypeSelected: (String) -> Unit,
    onStartDriving: () -> Unit,
    onStopDriving: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.96f)),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 10.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Driving Risk", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text(
                        text = if (uiState.isDriving) "Trip active" else "Select vehicle type before start",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                RiskBadge(riskLevel = uiState.riskLevel)
            }

            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                DRIVING_VEHICLE_TYPES.forEach { type ->
                    FilterChip(
                        selected = uiState.vehicleType == type,
                        enabled = !uiState.isDriving && !uiState.isStartingTrip,
                        onClick = { onVehicleTypeSelected(type) },
                        label = { Text(text = type.replaceFirstChar { it.uppercase() }) },
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                Button(
                    onClick = onStartDriving,
                    enabled = !uiState.isDriving && !uiState.isStartingTrip,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = if (uiState.isStartingTrip) "Starting..." else "Start")
                }
                OutlinedButton(
                    onClick = onStopDriving,
                    enabled = (uiState.isDriving || uiState.isStartingTrip) && !uiState.isEndingTrip,
                    modifier = Modifier.weight(1f),
                ) {
                    Text(text = if (uiState.isEndingTrip) "Stopping..." else "End")
                }
            }

            RiskSummary(uiState = uiState)

            uiState.errorMessage?.let { message ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedButton(onClick = onRetry) { Text(text = "Retry") }
                }
            }
        }
    }
}

@Composable
private fun RiskSummary(uiState: MapUiState) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        MetricPill(
            label = "Score",
            value = uiState.riskScore?.let { "%.2f".format(it) } ?: "--",
            modifier = Modifier.weight(1f),
        )
        MetricPill(
            label = "Speed",
            value = uiState.currentLocation?.speedKmh?.let { "%.0f km/h".format(it) } ?: "--",
            modifier = Modifier.weight(1f),
        )
        MetricPill(
            label = "Distance",
            value = "%.2f km".format(uiState.distanceKm),
            modifier = Modifier.weight(1f),
        )
    }

    Spacer(modifier = Modifier.height(2.dp))

    Text(
        text = uiState.alertMessage ?: "Risk prediction starts after GPS locks onto your trip.",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurface,
    )
    Text(
        text = listOfNotNull(
            uiState.nearestZoneName?.let { "Nearest: $it" },
            uiState.recommendedSpeedKmh?.let { "Recommended: $it km/h" },
            uiState.lastUpdatedLabel,
        ).joinToString("  "),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun RiskBadge(riskLevel: String?) {
    val label = riskLevel ?: "Idle"
    AssistChip(
        onClick = {},
        label = { Text(text = label.uppercase()) },
        leadingIcon = {
            Box(
                modifier = Modifier
                    .width(10.dp)
                    .height(10.dp)
                    .background(riskColor(label), RoundedCornerShape(5.dp)),
            )
        },
    )
}

@Composable
private fun MetricPill(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
    ) {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(text = value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
    }
}

private fun riskColor(level: String?): Color {
    return when (level?.lowercase()) {
        "high" -> Color(0xFFD32F2F)
        "medium" -> Color(0xFFF9A825)
        "low" -> Color(0xFF2E7D32)
        else -> Color(0xFF607D8B)
    }
}

private fun riskHex(level: String?): String {
    return when (level?.lowercase()) {
        "high" -> "#d32f2f"
        "medium" -> "#f9a825"
        "low" -> "#2e7d32"
        else -> "#607d8b"
    }
}

private fun buildLeafletHtml(uiState: MapUiState): String {
    val centerLat = uiState.currentLocation?.latitude
        ?: uiState.zones.firstOrNull()?.startLat
        ?: DEFAULT_LATITUDE
    val centerLon = uiState.currentLocation?.longitude
        ?: uiState.zones.firstOrNull()?.startLon
        ?: DEFAULT_LONGITUDE

    val zonesJson = JSONArray().apply {
        uiState.zones.forEach { zone ->
            put(
                JSONObject()
                    .put("name", zone.name)
                    .put("startLat", zone.startLat)
                    .put("startLon", zone.startLon)
                    .put("endLat", zone.endLat)
                    .put("endLon", zone.endLon)
                    .put("color", riskHex(zone.baseRiskLevel)),
            )
        }
    }

    val routeJson = JSONArray().apply {
        uiState.routePoints.forEach { point ->
            put(JSONArray().put(point.latitude).put(point.longitude))
        }
    }

    val currentJson = uiState.currentLocation?.let {
        JSONObject()
            .put("lat", it.latitude)
            .put("lon", it.longitude)
            .put("speed", it.speedKmh?.let { speed -> "%.0f km/h".format(speed) } ?: "")
    } ?: JSONObject.NULL

    return """
        <!doctype html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.1.0/dist/maplibre-gl.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script src="https://unpkg.com/maplibre-gl@5.1.0/dist/maplibre-gl.js"></script>
            <script src="https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.0.22/leaflet-maplibre-gl.js"></script>
            <style>
                html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; background: #eef2f1; }
                .leaflet-control-attribution { font-size: 10px; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                const center = [$centerLat, $centerLon];
                const zones = $zonesJson;
                const route = $routeJson;
                const current = $currentJson;
                const map = L.map('map', { zoomControl: true }).setView(center, current ? 15 : 13);

                L.maplibreGL({
                    style: 'https://tiles.openfreemap.org/styles/liberty'
                }).addTo(map);

                zones.forEach(zone => {
                    const line = [[zone.startLat, zone.startLon], [zone.endLat, zone.endLon]];
                    L.polyline(line, { color: zone.color, weight: 6, opacity: 0.82 }).addTo(map);
                    L.marker([zone.startLat, zone.startLon]).addTo(map).bindPopup(zone.name);
                });

                if (route.length > 1) {
                    L.polyline(route, { color: '#1976d2', weight: 7, opacity: 0.9 }).addTo(map);
                }

                if (current) {
                    L.circleMarker([current.lat, current.lon], {
                        radius: 9,
                        color: '#ffffff',
                        weight: 3,
                        fillColor: '#1976d2',
                        fillOpacity: 1
                    }).addTo(map).bindPopup(current.speed ? 'Current position<br>' + current.speed : 'Current position');
                }
            </script>
        </body>
        </html>
    """.trimIndent()
}

private const val MAP_BASE_URL = "https://tiles.openfreemap.org/"
private const val DEFAULT_LATITUDE = 7.2536
private const val DEFAULT_LONGITUDE = 80.5275
