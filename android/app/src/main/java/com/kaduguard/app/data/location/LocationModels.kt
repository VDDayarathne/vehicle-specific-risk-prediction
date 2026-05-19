package com.kaduguard.app.data.location

import android.location.Location

data class LocationSnapshot(
    val latitude: Double,
    val longitude: Double,
    val accuracyMeters: Float? = null,
    val speedKmh: Double? = null,
    val bearingDeg: Float? = null,
    val altitudeMeters: Double? = null,
    val timestampMillis: Long = System.currentTimeMillis(),
) {
    companion object {
        fun fromLocation(location: Location): LocationSnapshot {
            return LocationSnapshot(
                latitude = location.latitude,
                longitude = location.longitude,
                accuracyMeters = if (location.hasAccuracy()) location.accuracy else null,
                speedKmh = if (location.hasSpeed()) location.speed * 3.6 else null,
                bearingDeg = if (location.hasBearing()) location.bearing else null,
                altitudeMeters = if (location.hasAltitude()) location.altitude else null,
                timestampMillis = location.time.takeIf { it > 0 } ?: System.currentTimeMillis(),
            )
        }
    }
}
