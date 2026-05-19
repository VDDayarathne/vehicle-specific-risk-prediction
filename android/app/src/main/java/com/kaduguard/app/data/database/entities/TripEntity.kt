package com.kaduguard.app.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "trips")
data class TripEntity(
    @PrimaryKey val tripId: String,
    val driverId: String,
    val startTime: Long,
    val endTime: Long? = null,
    val startLatitude: Double,
    val startLongitude: Double,
    val endLatitude: Double? = null,
    val endLongitude: Double? = null,
    val distanceKm: Double = 0.0,
    val durationMinutes: Int? = null,
    val avgRiskScore: Double = 0.0,
    val maxRiskScore: Double = 0.0,
    val highRiskCount: Int = 0,
    val mediumRiskCount: Int = 0,
    val lowRiskCount: Int = 0,
    val vehicleType: String,
    val notes: String? = null,
)
