package com.kaduguard.app.data.database.dao

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.kaduguard.app.data.database.entities.OfflineQueueEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface OfflineQueueDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun enqueue(item: OfflineQueueEntity)

    @Query("SELECT * FROM offline_queue ORDER BY createdAt ASC")
    fun observeQueue(): Flow<List<OfflineQueueEntity>>

    @Query("SELECT * FROM offline_queue ORDER BY createdAt ASC")
    suspend fun getAll(): List<OfflineQueueEntity>

    @Delete
    suspend fun delete(item: OfflineQueueEntity)

    @Query("DELETE FROM offline_queue")
    suspend fun clearAll()
}
