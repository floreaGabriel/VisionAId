package com.florea_gabriel.impairedhelpapp.presentation.home

import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.florea_gabriel.impairedhelpapp.presentation.camera.CameraScreen

private val BOTTOM_NAV_HEIGHT = 80.dp

@Composable
fun LiveDetectionScreen(
        isScreenVisible: Boolean = true,
        triggerSceneDescription: Boolean,
        onSceneDescriptionComplete: () -> Unit,
        onSceneDescription: (String) -> Unit
) {
        Box(modifier = Modifier.fillMaxSize()) {
                CameraScreen(
                        isScreenVisible = isScreenVisible,
                        bottomPadding = BOTTOM_NAV_HEIGHT,
                        triggerSceneDescription = triggerSceneDescription,
                        onSceneDescriptionComplete = onSceneDescriptionComplete,
                        onSceneDescription = onSceneDescription
                )
        }
}
