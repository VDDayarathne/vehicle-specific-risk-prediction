package com.kaduguard.app.data.database.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.kaduguard.app.data.database.entities.RiskZoneEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface RiskZoneDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAll(zones: List<RiskZoneEntity>)

    @Query("SELECT * FROM risk_zones ORDER BY baseRiskLevel DESC, name ASC")
    fun observeRiskZones(): Flow<List<RiskZoneEntity>>

    @Query("SELECT * FROM risk_zones ORDER BY updatedAt DESC")
    suspend fun getAll(): List<RiskZoneEntity>

    @Query("DELETE FROM risk_zones")
    suspend fun clearAll()
}
