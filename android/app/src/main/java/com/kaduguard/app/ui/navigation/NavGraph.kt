package com.kaduguard.app.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.kaduguard.app.data.local.AuthTokenStore
import com.kaduguard.app.ui.screens.auth.AuthRoute
import com.kaduguard.app.ui.screens.home.HomeRoute
import com.kaduguard.app.ui.screens.history.TripHistoryRoute
import com.kaduguard.app.ui.screens.prediction.PredictionRoute

object Routes {
    const val STARTUP = "startup"
    const val LOGIN = "login"
    const val HOME = "home"
    const val PREDICTION = "prediction"
    const val MAP = "map"
    const val HISTORY = "history"
}

@Composable
fun KaduGuardNavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = Routes.STARTUP,
    ) {
        composable(Routes.STARTUP) {
            val context = LocalContext.current
            val tokenStore = remember(context) { AuthTokenStore(context.applicationContext) }
            val accessToken by tokenStore.accessToken.collectAsState(initial = null)

            LaunchedEffect(accessToken) {
                if (!accessToken.isNullOrBlank()) {
                    navController.navigate(Routes.HOME) {
                        popUpTo(Routes.STARTUP) { inclusive = true }
                    }
                } else {
                    navController.navigate(Routes.LOGIN) {
                        popUpTo(Routes.STARTUP) { inclusive = true }
                    }
                }
            }
        }
        composable(Routes.LOGIN) {
            AuthRoute(onAuthenticated = {
                navController.navigate(Routes.HOME) {
                    popUpTo(Routes.LOGIN) { inclusive = true }
                }
            })
        }
        composable(Routes.HOME) {
            HomeRoute(
                onOpenPrediction = {
                    navController.navigate(Routes.PREDICTION)
                },
                onOpenMap = {
                    navController.navigate(Routes.MAP)
                },
                onOpenHistory = {
                    navController.navigate(Routes.HISTORY)
                },
                onLogout = {
                    navController.navigate(Routes.LOGIN) {
                        popUpTo(Routes.HOME) { inclusive = true }
                    }
                },
            )
        }
        composable(Routes.PREDICTION) {
            PredictionRoute(onBack = {
                navController.navigate(Routes.HOME) {
                    popUpTo(Routes.PREDICTION) { inclusive = true }
                }
            })
        }
        composable(Routes.MAP) {
            // Back will return to home
            com.kaduguard.app.ui.screens.map.MapRoute(onBack = {
                navController.navigate(Routes.HOME) {
                    popUpTo(Routes.MAP) { inclusive = true }
                }
            })
        }
        composable(Routes.HISTORY) {
            TripHistoryRoute(onBack = {
                navController.navigate(Routes.HOME) {
                    popUpTo(Routes.HISTORY) { inclusive = true }
                }
            })
        }
    }
}
