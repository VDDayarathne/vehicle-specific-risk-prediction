package com.kaduguard.app.data.database

import androidx.room.Database
import androidx.room.RoomDatabase
import com.kaduguard.app.data.database.dao.OfflineQueueDao
import com.kaduguard.app.data.database.dao.PredictionDao
import com.kaduguard.app.data.database.dao.RiskZoneDao
import com.kaduguard.app.data.database.dao.TripDao
import com.kaduguard.app.data.database.entities.OfflineQueueEntity
import com.kaduguard.app.data.database.entities.PredictionEntity
import com.kaduguard.app.data.database.entities.RiskZoneEntity
import com.kaduguard.app.data.database.entities.TripEntity

@Database(
    entities = [
        TripEntity::class,
        PredictionEntity::class,
        RiskZoneEntity::class,
        OfflineQueueEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class KaduGuardDatabase : RoomDatabase() {
    abstract fun tripDao(): TripDao
    abstract fun predictionDao(): PredictionDao
    abstract fun riskZoneDao(): RiskZoneDao
    abstract fun offlineQueueDao(): OfflineQueueDao
}
