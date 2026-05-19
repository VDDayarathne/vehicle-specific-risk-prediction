package com.kaduguard.app.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "risk_zones")
data class RiskZoneEntity(
    @PrimaryKey val segmentId: String,
    val name: String,
    val startLatitude: Double,
    val startLongitude: Double,
    val endLatitude: Double,
    val endLongitude: Double,
    val gradientPct: Double,
    val curvature: Double,
    val baseRiskLevel: String,
    val updatedAt: Long,
)
