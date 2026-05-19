package com.kaduguard.app.data.database.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.kaduguard.app.data.database.entities.PredictionEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface PredictionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(prediction: PredictionEntity)

    @Query("SELECT * FROM predictions ORDER BY timestamp DESC")
    fun observeRecentPredictions(): Flow<List<PredictionEntity>>

    @Query("SELECT * FROM predictions WHERE cached = 1 ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getCachedPredictions(limit: Int = 20): List<PredictionEntity>

    @Query("DELETE FROM predictions")
    suspend fun clearAll()
}
