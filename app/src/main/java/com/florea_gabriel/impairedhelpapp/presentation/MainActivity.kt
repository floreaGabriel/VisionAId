package com.florea_gabriel.impairedhelpapp.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.florea_gabriel.impairedhelpapp.presentation.onboarding.OnboardingFlow
import com.florea_gabriel.impairedhelpapp.presentation.onboarding.OnboardingPrefs
import com.florea_gabriel.impairedhelpapp.presentation.permissions.CameraPermissionScreen
import com.florea_gabriel.impairedhelpapp.ui.theme.VisionAIdTheme

/**
 * MainActivity: Entry point of the VisionAId application.
 *
 * App Flow:
 * 1. MainActivity starts
 * 2. Onboarding (welcome + features overview) shown only on first install
 * 3. CameraPermissionScreen checks for camera permission
 * 4. If granted -> HomeScreen with bottom navigation is shown
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Edge-to-edge with dark status bar
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.parseColor("#0F1419")),
            navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.parseColor("#0F1419"))
        )

        // Hide navigation bar (immersive mode)
        val windowInsetsController = WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        windowInsetsController.hide(WindowInsetsCompat.Type.navigationBars())

        setContent {
            VisionAIdTheme {
                Box(
                        modifier = Modifier
                                .fillMaxSize()
                                .background(Color(0xFF0F1419))
                ) {
                    Surface(
                            modifier = Modifier.fillMaxSize().safeDrawingPadding(),
                            color = MaterialTheme.colorScheme.background
                    ) {
                        val context = LocalContext.current
                        var onboardingDone by remember {
                            mutableStateOf(OnboardingPrefs.isOnboardingCompleted(context))
                        }

                        if (!onboardingDone) {
                            OnboardingFlow(onFinished = { onboardingDone = true })
                        } else {
                            CameraPermissionScreen()
                        }
                    }
                }
            }
        }
    }
}
