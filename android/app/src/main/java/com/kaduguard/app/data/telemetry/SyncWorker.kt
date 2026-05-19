package com.kaduguard.app.data.telemetry

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.kaduguard.app.data.database.KaduGuardDatabase
import com.kaduguard.app.data.database.entities.OfflineQueueEntity
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.repository.KaduGuardRepositoryImpl
import com.squareup.moshi.Moshi
import com.squareup.moshi.Types
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory

class SyncWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {
    private val MAX_RETRIES = 5

    override suspend fun doWork(): Result {
        val db = KaduGuardDatabase::class.java.let {
            androidx.room.Room.databaseBuilder(applicationContext, it, "kaduguard.db").build()
        }
        val items = db.offlineQueueDao().getAll()
        if (items.isEmpty()) return Result.success()

        val authStore = AuthTokenStore(applicationContext)
        val token = authStore.getAccessTokenOnce() ?: return Result.retry()
        val repo = KaduGuardRepositoryImpl(applicationContext)

        val moshi = Moshi.Builder().add(KotlinJsonAdapterFactory()).build()
        val type = Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java)
        val adapter = moshi.adapter<Map<String, Any>>(type)

        var anyFailures = false
        for (item in items) {
            try {
                val payload = adapter.fromJson(item.payloadJson) ?: emptyMap()
                repo.sendTelemetry(token, payload)
                db.offlineQueueDao().delete(item)
            } catch (t: Throwable) {
                anyFailures = true
                val newRetry = item.retryCount + 1
                if (newRetry >= MAX_RETRIES) {
                    db.offlineQueueDao().delete(item)
                } else {
                    val updated = item.copy(retryCount = newRetry, lastError = t.message)
                    db.offlineQueueDao().enqueue(updated)
                    db.offlineQueueDao().delete(item)
                }
            }
        }

        return if (anyFailures) Result.retry() else Result.success()
    }
}
