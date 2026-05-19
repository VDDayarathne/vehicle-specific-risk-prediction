package com.kaduguard.app.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val BrandRed = Color(0xFFB71C1C)
private val BrandAmber = Color(0xFFF9A825)
private val BrandMint = Color(0xFF2A9D8F)
private val BrandInk = Color(0xFF102A43)
private val BrandPaper = Color(0xFFF7F4EF)

private val LightColors = lightColorScheme(
    primary = BrandInk,
    onPrimary = Color.White,
    secondary = BrandAmber,
    onSecondary = Color(0xFF1F1B16),
    tertiary = BrandMint,
    onTertiary = Color.White,
    background = BrandPaper,
    onBackground = BrandInk,
    surface = Color.White,
    onSurface = BrandInk,
    error = BrandRed,
    onError = Color.White,
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFFD9E2EC),
    onPrimary = Color(0xFF102A43),
    secondary = BrandAmber,
    onSecondary = Color(0xFF1F1B16),
    tertiary = BrandMint,
    onTertiary = Color(0xFF041F1A),
    background = Color(0xFF081622),
    onBackground = Color(0xFFE6EEF5),
    surface = Color(0xFF102A43),
    onSurface = Color(0xFFE6EEF5),
    error = Color(0xFFFF6B6B),
    onError = Color(0xFF1A0000),
)

@Composable
fun KaduGuardTheme(
    useDarkTheme: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colors = if (useDarkTheme) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colors,
        typography = Typography,
        content = content,
    )
}
