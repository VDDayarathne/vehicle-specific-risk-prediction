package com.kaduguard.app.data.location

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

object LocationTracker {
    private val _latestLocation = MutableStateFlow<LocationSnapshot?>(null)
    val latestLocation: StateFlow<LocationSnapshot?> = _latestLocation.asStateFlow()

    fun update(location: LocationSnapshot?) {
        _latestLocation.value = location
    }
}
