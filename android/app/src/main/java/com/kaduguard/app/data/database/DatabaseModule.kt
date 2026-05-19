package com.kaduguard.app.data.database

import android.content.Context
import androidx.room.Room
import com.kaduguard.app.data.database.dao.OfflineQueueDao
import com.kaduguard.app.data.database.dao.PredictionDao
import com.kaduguard.app.data.database.dao.RiskZoneDao
import com.kaduguard.app.data.database.dao.TripDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    private const val DATABASE_NAME = "kaduguard.db"

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): KaduGuardDatabase {
        return Room.databaseBuilder(context, KaduGuardDatabase::class.java, DATABASE_NAME)
            .fallbackToDestructiveMigration()
            .build()
    }

    @Provides
    fun provideTripDao(database: KaduGuardDatabase): TripDao = database.tripDao()

    @Provides
    fun providePredictionDao(database: KaduGuardDatabase): PredictionDao = database.predictionDao()

    @Provides
    fun provideRiskZoneDao(database: KaduGuardDatabase): RiskZoneDao = database.riskZoneDao()

    @Provides
    fun provideOfflineQueueDao(database: KaduGuardDatabase): OfflineQueueDao = database.offlineQueueDao()
}
