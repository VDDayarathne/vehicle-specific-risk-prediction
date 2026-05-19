package com.kaduguard.app.data.database.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.kaduguard.app.data.database.entities.TripEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface TripDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(trip: TripEntity)

    @Update
    suspend fun update(trip: TripEntity)

    @Query("SELECT * FROM trips ORDER BY startTime DESC")
    fun observeAllTrips(): Flow<List<TripEntity>>

    @Query("SELECT * FROM trips WHERE tripId = :tripId LIMIT 1")
    suspend fun getTripById(tripId: String): TripEntity?

    @Query("SELECT * FROM trips WHERE driverId = :driverId ORDER BY startTime DESC LIMIT :limit")
    suspend fun getRecentTrips(driverId: String, limit: Int = 20): List<TripEntity>

    @Delete
    suspend fun delete(trip: TripEntity)

    @Query("DELETE FROM trips")
    suspend fun clearAll()
}
