package com.kaduguard.app.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "predictions")
data class PredictionEntity(
    @PrimaryKey val predictionId: String,
    val timestamp: Long,
    val vehicleType: String,
    val riskLevel: String,
    val riskScore: Double,
    val recommendedSpeedKmh: Int?,
    val alertMessage: String,
    val latitude: Double? = null,
    val longitude: Double? = null,
    val temperatureC: Double? = null,
    val humidityPct: Double? = null,
    val rainfallMm: Double? = null,
    val windSpeedKmh: Double? = null,
    val visibilityKm: Double? = null,
    val cached: Boolean = true,
)
