package com.florea_gabriel.impairedhelpapp.presentation.camera.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas

/**
 * PerformanceMetricsOverlay: Displays real-time performance information.
 *
 * Shows FPS, detection count, and optional depth inference time.
 * Useful for debugging and thesis documentation.
 *
 * @param fps Current frames per second
 * @param detectionCount Number of objects currently detected
 * @param depthInferenceTime Depth model inference time in milliseconds (optional)
 * @param showDepthMetrics Whether to show depth-related metrics
 */
@Composable
fun PerformanceMetrics(
    fps: Float,
    detectionCount: Int,
    depthInferenceTime: Long = 0L,
    showDepthMetrics: Boolean = false
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val textPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.YELLOW
            textSize = 48f
            textAlign = android.graphics.Paint.Align.LEFT
            isAntiAlias = true
            isFakeBoldText = true
            setShadowLayer(4f, 2f, 2f, android.graphics.Color.BLACK)
        }

        val cyanPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.CYAN
            textSize = 48f
            textAlign = android.graphics.Paint.Align.LEFT
            isAntiAlias = true
            isFakeBoldText = true
            setShadowLayer(4f, 2f, 2f, android.graphics.Color.BLACK)
        }

        drawIntoCanvas { canvas ->
            var yOffset = 180f
            val xOffset = 20f
            val lineHeight = 55f

            // FPS
            canvas.nativeCanvas.drawText(
                "FPS: ${"%.1f".format(fps)}",
                xOffset,
                yOffset,
                textPaint
            )
            yOffset += lineHeight

            // Detection count
            canvas.nativeCanvas.drawText(
                "Detections: $detectionCount",
                xOffset,
                yOffset,
                textPaint
            )
            yOffset += lineHeight

            // Depth inference time (if showing depth metrics)
            if (showDepthMetrics) {
                canvas.nativeCanvas.drawText(
                    "Depth: ${depthInferenceTime}ms",
                    xOffset,
                    yOffset,
                    cyanPaint
                )
            }
        }
    }
}
