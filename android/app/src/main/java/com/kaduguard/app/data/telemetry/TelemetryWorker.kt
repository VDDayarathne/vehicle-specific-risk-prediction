package com.kaduguard.app.data.telemetry

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.WorkerParameters
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.repository.KaduGuardRepositoryImpl
import com.kaduguard.app.data.database.KaduGuardDatabase
import com.kaduguard.app.data.database.entities.OfflineQueueEntity
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import com.squareup.moshi.Types
import kotlinx.coroutines.runBlocking

class TelemetryWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {
    companion object {
        const val KEY_LAT = "lat"
        const val KEY_LON = "lon"
        const val KEY_SPEED = "speed"
        const val KEY_HEADING = "heading"
        const val KEY_DEVICE_ID = "device_id"
    }

    override suspend fun doWork(): Result {
        val lat = inputData.getDouble(KEY_LAT, Double.NaN)
        val lon = inputData.getDouble(KEY_LON, Double.NaN)
        val speed = inputData.getDouble(KEY_SPEED, Double.NaN)
        val heading = inputData.getDouble(KEY_HEADING, Double.NaN)
        val deviceId = inputData.getString(KEY_DEVICE_ID)

        if (lat.isNaN() || lon.isNaN()) return Result.failure()

        val authStore = AuthTokenStore(applicationContext)
        val token = authStore.getAccessTokenOnce()

        val repo = KaduGuardRepositoryImpl(applicationContext)

        val payload = mutableMapOf<String, Any>(
            "latitude" to lat,
            "longitude" to lon,
            "timestamp" to System.currentTimeMillis(),
        )
        if (!speed.isNaN()) payload["speed_kmh"] = speed
        if (!heading.isNaN()) payload["heading_deg"] = heading
        if (!deviceId.isNullOrBlank()) payload["device_id"] = deviceId

        return try {
            if (token.isNullOrBlank()) {
                // offline: enqueue into Room queue
                persistToQueue(applicationContext, payload)
                enqueueSync(applicationContext)
                Result.success()
            } else {
                repo.sendTelemetry(token, payload)
                Result.success()
            }
        } catch (t: Throwable) {
            // on failure, persist to offline queue for later sync
            try {
                persistToQueue(applicationContext, payload, t.message)
                enqueueSync(applicationContext)
            } catch (_: Throwable) {
            }
            Result.success()
        }
    }

    private fun persistToQueue(context: Context, payload: Map<String, Any>, lastError: String? = null) {
        val moshi = Moshi.Builder().add(KotlinJsonAdapterFactory()).build()
        val type = Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java)
        val adapter = moshi.adapter<Map<String, Any>>(type)
        val json = adapter.toJson(payload)

        val db = KaduGuardDatabase::class.java.let {
            androidx.room.Room.databaseBuilder(context, it, "kaduguard.db").build()
        }

        val entity = OfflineQueueEntity(
            requestId = java.util.UUID.randomUUID().toString(),
            requestType = "telemetry",
            payloadJson = json,
            createdAt = System.currentTimeMillis(),
            retryCount = 0,
            lastError = lastError,
        )

        runBlocking { db.offlineQueueDao().enqueue(entity) }
    }

    private fun enqueueSync(context: Context) {
        val work = OneTimeWorkRequestBuilder<SyncWorker>().build()
        WorkManager.getInstance(context).enqueue(work)
    }
}
