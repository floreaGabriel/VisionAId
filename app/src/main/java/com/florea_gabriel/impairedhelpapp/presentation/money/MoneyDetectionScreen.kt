package com.florea_gabriel.impairedhelpapp.presentation.money

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.florea_gabriel.impairedhelpapp.data.model.Detection
import com.florea_gabriel.impairedhelpapp.ml.detector.MoneyDetector
import com.florea_gabriel.impairedhelpapp.ml.processor.ImageProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "MoneyDetection"

/**
 * State machine for one-at-a-time banknote counting:
 * WAITING    — no bill in frame, ready for next
 * DETECTED   — bill detected & confirmed (stable for N frames), added to total
 * COOLDOWN   — bill was counted, waiting for frame to clear before accepting next
 */
private enum class ScanState { WAITING, DETECTED, COOLDOWN }

@Composable
fun MoneyDetectionScreen(
    isScreenVisible: Boolean,
    bottomPadding: Dp = 0.dp,
    onSpeakRequest: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val isActive = remember { AtomicBoolean(true) }
    val currentIsVisible by rememberUpdatedState(isScreenVisible)
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Current frame detections (for overlay)
    var currentDetections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var lastProcessTime by remember { mutableLongStateOf(0L) }
    var previewWidth by remember { mutableIntStateOf(1) }
    var previewHeight by remember { mutableIntStateOf(1) }

    // Counting state
    var runningTotal by remember { mutableIntStateOf(0) }
    var countedBills by remember { mutableStateOf<List<Int>>(emptyList()) }
    var scanState by remember { mutableStateOf(ScanState.WAITING) }
    var lastDetectedLabel by remember { mutableStateOf("") }
    var stableFrameCount by remember { mutableIntStateOf(0) }
    var emptyFrameCount by remember { mutableIntStateOf(0) }
    var lastCountedValue by remember { mutableIntStateOf(0) }

    // Stability thresholds
    val stableFramesNeeded = 3  // consecutive frames with same detection to confirm
    val emptyFramesNeeded = 3  // consecutive empty frames before accepting next bill

    var moneyDetector by remember { mutableStateOf<MoneyDetector?>(null) }

    // Initialize model on background thread to avoid blocking UI
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            moneyDetector = MoneyDetector(context)
        }
        onSpeakRequest("Pune câte o bancnotă în fața camerei. Voi anunța valoarea și totalul.")
    }

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
                        .setTargetResolution(Size(1280, 720))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor) analyzer@{ imageProxy ->
                        if (!isActive.get() || !currentIsVisible) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val detector = moneyDetector
                        if (detector == null) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val now = System.currentTimeMillis()
                        if (now - lastProcessTime < 150L) {
                            imageProxy.close()
                            return@analyzer
                        }
                        lastProcessTime = now

                        try {
                            val bitmap = try {
                                ImageProcessor.imageProxyToBitmap(imageProxy)
                            } catch (e: Exception) {
                                null
                            }
                            if (bitmap != null) {
                                previewWidth = bitmap.width
                                previewHeight = bitmap.height
                                val results = detector.detect(bitmap)
                                currentDetections = results
                                bitmap.recycle()

                                // State machine logic
                                val bestDetection = results.maxByOrNull { it.confidence }

                                when (scanState) {
                                    ScanState.WAITING -> {
                                        if (bestDetection != null) {
                                            if (bestDetection.label == lastDetectedLabel) {
                                                stableFrameCount++
                                            } else {
                                                lastDetectedLabel = bestDetection.label
                                                stableFrameCount = 1
                                            }
                                            // Confirmed: same label for N consecutive frames
                                            if (stableFrameCount >= stableFramesNeeded) {
                                                val value = MoneyDetector.labelToValue(bestDetection.label)
                                                runningTotal += value
                                                countedBills = countedBills + value
                                                lastCountedValue = value
                                                scanState = ScanState.DETECTED
                                                stableFrameCount = 0

                                                val totalText = if (runningTotal == value) {
                                                    "${formatValue(value)}. Total: ${formatValue(runningTotal)}"
                                                } else {
                                                    "${formatValue(value)}. Total: ${formatValue(runningTotal)}"
                                                }
                                                onSpeakRequest(totalText)
                                            }
                                        } else {
                                            // No detection — reset stability counter
                                            stableFrameCount = 0
                                            lastDetectedLabel = ""
                                        }
                                    }

                                    ScanState.DETECTED -> {
                                        // Wait for bill to be removed
                                        if (bestDetection == null) {
                                            emptyFrameCount++
                                            if (emptyFrameCount >= emptyFramesNeeded) {
                                                scanState = ScanState.WAITING
                                                emptyFrameCount = 0
                                                lastDetectedLabel = ""
                                            }
                                        } else {
                                            emptyFrameCount = 0
                                        }
                                    }

                                    ScanState.COOLDOWN -> {
                                        // Not used currently, but available for future extensions
                                        scanState = ScanState.WAITING
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Analysis error: ${e.message}")
                        } finally {
                            imageProxy.close()
                        }
                    }

                    // Guard: don't bind if screen was already disposed during async provider fetch
                    if (!isActive.get()) return@addListener

                    // Bind after layout so previewView.viewPort is available: a shared
                    // ViewPort crops the analysis stream to exactly the region the
                    // preview displays (WYSIWYG) — otherwise boxes drift near edges
                    previewView.post {
                        if (!isActive.get()) return@post
                        try {
                            cameraProvider.unbindAll()
                            val viewPort = previewView.viewPort
                            if (viewPort != null) {
                                val group = UseCaseGroup.Builder()
                                    .setViewPort(viewPort)
                                    .addUseCase(preview)
                                    .addUseCase(imageAnalysis)
                                    .build()
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    group
                                )
                            } else {
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview,
                                    imageAnalysis
                                )
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Camera bind error: ${e.message}")
                        }
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Bounding box overlay
        if (currentDetections.isNotEmpty()) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val canvasW = size.width
                val canvasH = size.height
                val scaleX = canvasW / previewWidth.toFloat()
                val scaleY = canvasH / previewHeight.toFloat()
                val scale = maxOf(scaleX, scaleY)
                val offsetX = (canvasW - previewWidth * scale) / 2f
                val offsetY = (canvasH - previewHeight * scale) / 2f

                val boxColor = when (scanState) {
                    ScanState.WAITING -> Color(0xFFFFA000) // Orange — scanning
                    ScanState.DETECTED -> Color(0xFF4CAF50) // Green — counted
                    ScanState.COOLDOWN -> Color.Gray
                }

                for (det in currentDetections) {
                    val box = det.boundingBox
                    val left = box.left * scale + offsetX
                    val top = box.top * scale + offsetY
                    val right = box.right * scale + offsetX
                    val bottom = box.bottom * scale + offsetY

                    drawRect(
                        color = boxColor,
                        topLeft = Offset(left, top),
                        size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                        style = Stroke(width = 4.dp.toPx())
                    )

                    val labelText = formatLabel(det.label) + " ${(det.confidence * 100).toInt()}%"
                    val paint = android.graphics.Paint().apply {
                        color = android.graphics.Color.WHITE
                        textSize = 16.dp.toPx()
                        isAntiAlias = true
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                    }
                    val textWidth = paint.measureText(labelText)
                    val textHeight = paint.textSize

                    drawRect(
                        color = boxColor,
                        topLeft = Offset(left, top - textHeight - 8.dp.toPx()),
                        size = androidx.compose.ui.geometry.Size(
                            textWidth + 12.dp.toPx(),
                            textHeight + 8.dp.toPx()
                        )
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        labelText,
                        left + 6.dp.toPx(),
                        top - 6.dp.toPx(),
                        paint
                    )
                }
            }
        }

        // Status indicator at top
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 48.dp, start = 16.dp, end = 16.dp)
                .fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            val statusText = when (scanState) {
                ScanState.WAITING -> if (countedBills.isEmpty()) "Arată o bancnotă" else "Arată următoarea bancnotă"
                ScanState.DETECTED -> "Numărată! Ia bancnota"
                ScanState.COOLDOWN -> "Pregătit..."
            }
            val statusColor = when (scanState) {
                ScanState.WAITING -> Color(0xFFFFA000)
                ScanState.DETECTED -> Color(0xFF4CAF50)
                ScanState.COOLDOWN -> Color.Gray
            }
            Text(
                text = statusText,
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .background(statusColor.copy(alpha = 0.85f), RoundedCornerShape(12.dp))
                    .padding(horizontal = 20.dp, vertical = 10.dp)
            )
        }

        // Bottom panel — running total + buttons
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp + bottomPadding, start = 16.dp, end = 16.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Running total display
            Text(
                text = if (runningTotal > 0) "Total: ${formatValue(runningTotal)}" else "Total: 0 lei",
                color = Color.White,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(12.dp))
                    .padding(horizontal = 20.dp, vertical = 14.dp)
            )

            // Bill breakdown
            if (countedBills.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                val breakdown = countedBills
                    .groupBy { it }
                    .entries
                    .sortedByDescending { it.key }
                    .joinToString("  ") { (value, items) -> "${items.size}x${value}" }
                Text(
                    text = breakdown,
                    color = Color.White.copy(alpha = 0.9f),
                    fontSize = 16.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 16.dp, vertical = 6.dp)
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Buttons row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // "Spune totalul" button
                Button(
                    onClick = {
                        if (runningTotal == 0) {
                            onSpeakRequest("Nu ați numărat nicio bancnotă")
                        } else {
                            val breakdown = countedBills
                                .groupBy { it }
                                .entries
                                .sortedByDescending { it.key }
                                .joinToString(", ") { (value, items) ->
                                    "${items.size} de ${formatValue(value)}"
                                }
                            onSpeakRequest("Aveți ${formatValue(runningTotal)}. $breakdown")
                        }
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    ),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp)
                ) {
                    Text(
                        text = "Spune totalul",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                // Reset button
                OutlinedButton(
                    onClick = {
                        runningTotal = 0
                        countedBills = emptyList()
                        scanState = ScanState.WAITING
                        lastDetectedLabel = ""
                        stableFrameCount = 0
                        emptyFrameCount = 0
                        onSpeakRequest("Resetat. Arată prima bancnotă.")
                    },
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp)
                ) {
                    Text(
                        text = "Resetare",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
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
            val det = moneyDetector
            Thread {
                Thread.sleep(300)
                det?.close()
            }.start()
        }
    }
}

/**
 * Format value with "lei" / "leu" (1 leu, rest lei).
 */
private fun formatValue(value: Int): String {
    return if (value == 1) "$value leu" else "$value lei"
}

/**
 * Convert "100_ron" → "100 RON" for display.
 */
private fun formatLabel(label: String): String {
    return label.replace("_", " ").uppercase()
}
