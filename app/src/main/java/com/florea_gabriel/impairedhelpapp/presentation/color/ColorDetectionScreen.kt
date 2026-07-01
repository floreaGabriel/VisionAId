package com.florea_gabriel.impairedhelpapp.presentation.color

import android.graphics.Color as AndroidColor
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "ColorDetection"

@Composable
fun ColorDetectionScreen(
    isScreenVisible: Boolean,
    bottomPadding: Dp = 0.dp,
    onColorDetected: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var colorName by remember { mutableStateOf("") }
    var colorRgb by remember { mutableStateOf(Triple(128, 128, 128)) }

    val isActive = remember { AtomicBoolean(true) }
    val currentIsVisible by rememberUpdatedState(isScreenVisible)
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    var lastColorName by remember { mutableStateOf("") }
    var lastProcessTime by remember { mutableLongStateOf(0L) }

    Box(modifier = Modifier.fillMaxSize()) {

        // Camera preview
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                }

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor) analyzer@{ imageProxy ->
                        if (!isActive.get() || !currentIsVisible) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val now = System.currentTimeMillis()
                        if (now - lastProcessTime < 1000L) {
                            imageProxy.close()
                            return@analyzer
                        }
                        lastProcessTime = now

                        try {
                            val w = imageProxy.width
                            val h = imageProxy.height

                            // Sample center 30x30 area from YUV_420_888
                            val yPlane = imageProxy.planes[0]
                            val uPlane = imageProxy.planes[1]
                            val vPlane = imageProxy.planes[2]

                            val yBuffer = yPlane.buffer
                            val uBuffer = uPlane.buffer
                            val vBuffer = vPlane.buffer

                            val yRowStride = yPlane.rowStride
                            val uvRowStride = uPlane.rowStride
                            val uvPixelStride = uPlane.pixelStride

                            val centerX = w / 2
                            val centerY = h / 2
                            val sampleSize = 15 // half of 30

                            var totalR = 0L
                            var totalG = 0L
                            var totalB = 0L
                            var count = 0

                            for (dy in -sampleSize until sampleSize) {
                                for (dx in -sampleSize until sampleSize) {
                                    val px = centerX + dx
                                    val py = centerY + dy
                                    if (px < 0 || px >= w || py < 0 || py >= h) continue

                                    val yVal = (yBuffer.get(py * yRowStride + px).toInt() and 0xFF)
                                    val uvX = px / 2
                                    val uvY = py / 2
                                    val uvIndex = uvY * uvRowStride + uvX * uvPixelStride
                                    val uVal = (uBuffer.get(uvIndex).toInt() and 0xFF) - 128
                                    val vVal = (vBuffer.get(uvIndex).toInt() and 0xFF) - 128

                                    // YUV to RGB
                                    var r = (yVal + 1.370705 * vVal).toInt()
                                    var g = (yVal - 0.337633 * uVal - 0.698001 * vVal).toInt()
                                    var b = (yVal + 1.732446 * uVal).toInt()
                                    r = r.coerceIn(0, 255)
                                    g = g.coerceIn(0, 255)
                                    b = b.coerceIn(0, 255)

                                    totalR += r
                                    totalG += g
                                    totalB += b
                                    count++
                                }
                            }

                            if (count > 0) {
                                val avgR = (totalR / count).toInt()
                                val avgG = (totalG / count).toInt()
                                val avgB = (totalB / count).toInt()

                                colorRgb = Triple(avgR, avgG, avgB)
                                val detected = getColorNameRomanian(avgR, avgG, avgB)
                                colorName = detected

                                if (detected != lastColorName) {
                                    lastColorName = detected
                                    onColorDetected(detected)
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Color analysis error: ${e.message}")
                        } finally {
                            imageProxy.close()
                        }
                    }

                    // Guard: don't bind if screen was already disposed during async provider fetch
                    if (!isActive.get()) return@addListener

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera bind error: ${e.message}")
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Crosshair overlay
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cx = size.width / 2f
            val cy = size.height / 2f
            val lineLen = 30.dp.toPx()
            val circleR = 30.dp.toPx()
            val strokeW = 2.dp.toPx()
            val crosshairColor = Color.White

            // Horizontal line
            drawLine(crosshairColor, Offset(cx - lineLen, cy), Offset(cx + lineLen, cy), strokeWidth = strokeW)
            // Vertical line
            drawLine(crosshairColor, Offset(cx, cy - lineLen), Offset(cx, cy + lineLen), strokeWidth = strokeW)
            // Circle
            drawCircle(crosshairColor, radius = circleR, center = Offset(cx, cy), style = Stroke(width = strokeW))
        }

        // Bottom color info
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp + bottomPadding, start = 24.dp, end = 24.dp)
                .fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(16.dp))
                    .padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                // Color swatch
                Box(
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            Color(colorRgb.first / 255f, colorRgb.second / 255f, colorRgb.third / 255f),
                            RoundedCornerShape(12.dp)
                        )
                )

                Spacer(modifier = Modifier.width(16.dp))

                // Color name
                Text(
                    text = colorName.ifEmpty { "Se detectează..." },
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }

    // Cleanup
    DisposableEffect(Unit) {
        onDispose {
            isActive.set(false)
            cameraExecutor.shutdown()
            cameraExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Exception) {}
        }
    }
}

/**
 * Maps RGB to a Romanian color name using HSV ranges.
 */
private fun getColorNameRomanian(r: Int, g: Int, b: Int): String {
    val hsv = FloatArray(3)
    AndroidColor.RGBToHSV(r, g, b, hsv)
    val h = hsv[0]          // 0..360
    val s = hsv[1] * 100f   // 0..100
    val v = hsv[2] * 100f   // 0..100

    // Achromatic colors
    if (v < 15f) return "Negru"
    if (s < 15f && v > 80f) return "Alb"
    if (s < 15f) return "Gri"

    // Chromatic — base name from hue
    val baseName = when {
        h < 15f || h >= 345f -> "Roșu"
        h < 40f  -> "Portocaliu"
        h < 70f  -> "Galben"
        h < 160f -> "Verde"
        h < 200f -> "Cyan"
        h < 260f -> "Albastru"
        h < 290f -> "Violet"
        else     -> "Roz"
    }

    // Lightness nuance
    return when {
        v > 75f && s > 20f -> "$baseName deschis"
        v < 40f            -> "$baseName închis"
        else               -> baseName
    }
}
