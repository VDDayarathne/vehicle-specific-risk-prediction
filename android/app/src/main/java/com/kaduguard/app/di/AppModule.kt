package com.kaduguard.app.di

import android.content.Context
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.data.network.api.KaduGuardApiService
import com.kaduguard.app.data.network.ApiClient
import com.kaduguard.app.data.repository.KaduGuardRepositoryImpl
import com.kaduguard.app.domain.repository.KaduGuardRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Provides
    @Singleton
    fun provideApiService(@ApplicationContext context: Context): KaduGuardApiService = ApiClient.createService(context)

    @Provides
    @Singleton
    fun provideRepository(@ApplicationContext context: Context): KaduGuardRepository = KaduGuardRepositoryImpl(context)

    @Provides
    @Singleton
    fun provideAuthTokenStore(@ApplicationContext context: Context): AuthTokenStore = AuthTokenStore(context)
}
