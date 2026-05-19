package com.kaduguard.app.data.location

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.kaduguard.app.R
import androidx.work.Data
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.kaduguard.app.data.telemetry.TelemetryWorker

class LocationService : Service() {
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var locationCallback: LocationCallback? = null

    override fun onCreate() {
        super.onCreate()
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Starting GPS tracking"))
        startLocationUpdates()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startLocationUpdates()
            ACTION_STOP -> stopTracking()
            else -> startLocationUpdates()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopTracking()
        super.onDestroy()
    }

    private fun startLocationUpdates() {
        if (!hasLocationPermission()) {
            stopSelf()
            return
        }

        if (locationCallback != null) return

        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, LOCATION_INTERVAL_MS)
            .setMinUpdateIntervalMillis(MIN_UPDATE_INTERVAL_MS)
            .setWaitForAccurateLocation(false)
            .build()

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val location = result.lastLocation ?: return
                LocationTracker.update(LocationSnapshot.fromLocation(location))
                updateForegroundNotification(location.latitude, location.longitude)

                // enqueue a telemetry worker to deliver this fix
                try {
                    val data = Data.Builder()
                        .putDouble(TelemetryWorker.KEY_LAT, location.latitude)
                        .putDouble(TelemetryWorker.KEY_LON, location.longitude)
                        .putDouble(TelemetryWorker.KEY_SPEED, if (location.hasSpeed()) location.speed * 3.6 else Double.NaN)
                        .putDouble(TelemetryWorker.KEY_HEADING, if (location.hasBearing()) location.bearing.toDouble() else Double.NaN)
                        .putString(TelemetryWorker.KEY_DEVICE_ID, null)
                        .build()

                    val work = OneTimeWorkRequestBuilder<TelemetryWorker>()
                        .setInputData(data)
                        .build()

                    WorkManager.getInstance(applicationContext).enqueue(work)
                } catch (_: Throwable) {
                    // ignore enqueue failures for now
                }
            }
        }

        fusedLocationClient.requestLocationUpdates(
            request,
            locationCallback!!,
            mainLooper,
        )
    }

    private fun stopTracking() {
        locationCallback?.let { fusedLocationClient.removeLocationUpdates(it) }
        locationCallback = null
        LocationTracker.update(null)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun hasLocationPermission(): Boolean {
        val fine = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val coarse = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        return fine || coarse
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "KaduGuard GPS Tracking",
            NotificationManager.IMPORTANCE_LOW,
        )
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(contentText: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("KaduGuard GPS active")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateForegroundNotification(latitude: Double, longitude: Double) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification("Tracking at %.5f, %.5f".format(latitude, longitude)))
    }

    companion object {
        const val ACTION_START = "com.kaduguard.app.action.START_LOCATION"
        const val ACTION_STOP = "com.kaduguard.app.action.STOP_LOCATION"

        private const val CHANNEL_ID = "kaduguard_location_channel"
        private const val NOTIFICATION_ID = 1001
        private const val LOCATION_INTERVAL_MS = 10_000L
        private const val MIN_UPDATE_INTERVAL_MS = 5_000L

        fun start(context: Context) {
            val intent = Intent(context, LocationService::class.java).apply { action = ACTION_START }
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, LocationService::class.java).apply { action = ACTION_STOP }
            context.startService(intent)
        }
    }
}
