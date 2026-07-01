package com.florea_gabriel.impairedhelpapp.presentation.camera.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import com.florea_gabriel.impairedhelpapp.data.model.DepthResult

/**
 * DepthOverlay: Displays depth map visualization as a heatmap overlay.
 *
 * Color mapping (normalized depth):
 * - Blue (0.0): Far objects
 * - Cyan (0.25): Medium-far
 * - Green (0.5): Medium distance
 * - Yellow (0.75): Medium-close
 * - Red (1.0): Close objects
 *
 * @param depthBitmap Pre-computed depth heatmap bitmap
 * @param alpha Transparency level (0.0-1.0), default 0.7 for semi-transparent overlay
 * @param modifier Modifier for the image
 */
@Composable
fun DepthOverlay(
    depthBitmap: Bitmap?,
    alpha: Float = 0.7f,
    modifier: Modifier = Modifier
) {
    if (depthBitmap != null) {
        Image(
            bitmap = depthBitmap.asImageBitmap(),
            contentDescription = "Depth Map - Shows distance to objects with colors",
            modifier = modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.3f)),
            alpha = alpha
        )
    }
}

/**
 * Converts DepthResult to a colored heatmap Bitmap.
 *
 * Uses normalized depth values (0.0-1.0) where:
 * - Higher values (close to 1.0) = Close objects = Red/Yellow colors
 * - Lower values (close to 0.0) = Far objects = Blue/Cyan colors
 *
 * @param depthResult Depth estimation result containing normalized depth map
 * @return Bitmap with heatmap color visualization
 */
fun depthResultToHeatmapBitmap(depthResult: DepthResult): Bitmap {
    val width = depthResult.width
    val height = depthResult.height
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    val pixels = IntArray(width * height)

    for (y in 0 until height) {
        for (x in 0 until width) {
            val normalizedDepth = depthResult.depthMap[y][x]
            pixels[y * width + x] = depthToHeatmapColor(normalizedDepth)
        }
    }

    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
}

/**
 * Converts normalized depth value to heatmap color.
 *
 * Color gradient: Blue -> Cyan -> Green -> Yellow -> Red
 *
 * @param normalizedDepth Depth value in range [0.0, 1.0]
 *                        0.0 = far (blue), 1.0 = close (red)
 * @return ARGB color integer
 */
private fun depthToHeatmapColor(normalizedDepth: Float): Int {
    val depth = normalizedDepth.coerceIn(0f, 1f)

    val r: Int
    val g: Int
    val b: Int

    when {
        // Blue to Cyan (0.0 - 0.25)
        depth < 0.25f -> {
            val t = depth / 0.25f
            r = 0
            g = (255 * t).toInt()
            b = 255
        }
        // Cyan to Green (0.25 - 0.5)
        depth < 0.5f -> {
            val t = (depth - 0.25f) / 0.25f
            r = 0
            g = 255
            b = (255 * (1 - t)).toInt()
        }
        // Green to Yellow (0.5 - 0.75)
        depth < 0.75f -> {
            val t = (depth - 0.5f) / 0.25f
            r = (255 * t).toInt()
            g = 255
            b = 0
        }
        // Yellow to Red (0.75 - 1.0)
        else -> {
            val t = (depth - 0.75f) / 0.25f
            r = 255
            g = (255 * (1 - t)).toInt()
            b = 0
        }
    }

    return android.graphics.Color.argb(255, r, g, b)
}
