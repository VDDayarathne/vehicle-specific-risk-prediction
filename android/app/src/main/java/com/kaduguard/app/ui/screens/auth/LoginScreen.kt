package com.kaduguard.app.ui.screens.auth

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.kaduguard.app.presentation.viewmodel.AuthMode
import com.kaduguard.app.presentation.viewmodel.AuthUiState
import com.kaduguard.app.ui.components.HeroCard
import com.kaduguard.app.ui.components.PolishedBackground
import com.kaduguard.app.ui.components.ScreenSection

@Composable
fun LoginScreen(
    uiState: AuthUiState,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onPhoneChange: (String) -> Unit,
    onVehicleTypeChange: (String) -> Unit,
    onSubmit: () -> Unit,
    onToggleMode: () -> Unit,
) {
    PolishedBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp, Alignment.CenterVertically),
        ) {
            HeroCard(
                title = "KaduGuard",
                subtitle = "Drive with live risk guidance, alerts, and trip history in one place.",
                tint = MaterialTheme.colorScheme.primary,
            )

            ScreenSection(title = if (uiState.mode == AuthMode.Login) "Sign In" else "Create Account") {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = uiState.email,
                        onValueChange = onEmailChange,
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Email") },
                        singleLine = true,
                    )
                    OutlinedTextField(
                        value = uiState.password,
                        onValueChange = onPasswordChange,
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Password") },
                        singleLine = true,
                        visualTransformation = PasswordVisualTransformation(),
                    )
                    if (uiState.mode == AuthMode.Register) {
                        OutlinedTextField(
                            value = uiState.phone,
                            onValueChange = onPhoneChange,
                            modifier = Modifier.fillMaxWidth(),
                            label = { Text("Phone number") },
                            singleLine = true,
                        )
                        OutlinedTextField(
                            value = uiState.vehicleType,
                            onValueChange = onVehicleTypeChange,
                            modifier = Modifier.fillMaxWidth(),
                            label = { Text("Vehicle type") },
                            singleLine = true,
                        )
                    }
                    Button(
                        onClick = onSubmit,
                        enabled = !uiState.isLoading,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(if (uiState.isLoading) "Please wait..." else if (uiState.mode == AuthMode.Login) "Sign In" else "Create Account")
                    }
                    Text(
                        text = if (uiState.mode == AuthMode.Login) "New driver? Create an account." else "Already registered? Sign in.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(top = 4.dp),
                        color = MaterialTheme.colorScheme.primary,
                        maxLines = 2,
                        onTextLayout = null,
                    )
                    Button(onClick = onToggleMode, modifier = Modifier.fillMaxWidth()) {
                        Text(if (uiState.mode == AuthMode.Login) "Switch to Register" else "Switch to Sign In")
                    }
                }
            }

            if (!uiState.errorMessage.isNullOrBlank()) {
                Text(
                    text = uiState.errorMessage ?: "",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}