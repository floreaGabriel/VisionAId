package com.florea_gabriel.impairedhelpapp.presentation.permissions

import android.Manifest
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.florea_gabriel.impairedhelpapp.presentation.home.HomeScreen
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale

/**
 * CameraPermissionScreen: Entry point that handles camera permission before showing the app.
 *
 * Flow:
 * 1. Check if camera permission is granted
 * 2. If granted -> Show HomeScreen with navigation
 * 3. If denied -> Show accessible permission request UI
 *
 * Accessibility features:
 * - Large button (72dp height) for easy touch targeting
 * - High contrast colors
 * - Clear, readable text (20sp)
 * - TalkBack support with content descriptions
 */
@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraPermissionScreen() {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    when {
        // Permission granted - show main app with navigation
        cameraPermissionState.status.isGranted -> {
            HomeScreen()
        }

        // Permission not granted - show request UI
        else -> {
            PermissionRequestUI(
                shouldShowRationale = cameraPermissionState.status.shouldShowRationale,
                onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
            )
        }
    }
}

/**
 * PermissionRequestUI: Accessible UI for requesting camera permission.
 *
 * Designed for users with visual impairments:
 * - Large, high-contrast button
 * - Clear explanatory text
 * - Proper TalkBack descriptions
 *
 * @param shouldShowRationale Whether to show rationale (user previously denied)
 * @param onRequestPermission Callback to request permission
 */
@Composable
private fun PermissionRequestUI(
    shouldShowRationale: Boolean,
    onRequestPermission: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFF1A1A2E)  // Dark background for contrast
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Title
            Text(
                text = "Camera Permission Required",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .padding(bottom = 24.dp)
                    .semantics {
                        contentDescription = "Camera Permission Required"
                    }
            )

            // Explanation text
            val explanationText = if (shouldShowRationale) {
                "Camera access is required to detect objects and measure distances. " +
                "Please grant permission to use this app."
            } else {
                "This app uses the camera to detect objects around you and measure " +
                "distances to help you navigate safely."
            }

            Text(
                text = explanationText,
                fontSize = 20.sp,
                color = Color.White.copy(alpha = 0.9f),
                textAlign = TextAlign.Center,
                lineHeight = 28.sp,
                modifier = Modifier
                    .padding(bottom = 48.dp)
                    .semantics {
                        contentDescription = explanationText
                    }
            )

            // Large, accessible permission button
            Button(
                onClick = onRequestPermission,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(72.dp)  // Large touch target
                    .semantics {
                        contentDescription = "Grant camera permission button"
                    },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF4CAF50),  // Green for positive action
                    contentColor = Color.White
                )
            ) {
                Text(
                    text = "Grant Camera Permission",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
