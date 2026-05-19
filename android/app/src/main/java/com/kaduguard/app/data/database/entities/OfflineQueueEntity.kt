package com.kaduguard.app.data.database.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "offline_queue")
data class OfflineQueueEntity(
    @PrimaryKey val requestId: String,
    val requestType: String,
    val payloadJson: String,
    val createdAt: Long,
    val retryCount: Int = 0,
    val lastError: String? = null,
)
