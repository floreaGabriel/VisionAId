package com.florea_gabriel.impairedhelpapp.presentation.camera

import android.graphics.Bitmap
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Vibration
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.florea_gabriel.impairedhelpapp.data.model.DepthResult
import com.florea_gabriel.impairedhelpapp.ml.captioning.ImageCaptioningManager
import com.florea_gabriel.impairedhelpapp.ml.detector.DepthEstimator
import com.florea_gabriel.impairedhelpapp.ml.processor.ImageProcessor
import com.florea_gabriel.impairedhelpapp.presentation.camera.components.PerformanceMetrics
import com.florea_gabriel.impairedhelpapp.presentation.camera.components.depthResultToHeatmapBitmap
import com.florea_gabriel.impairedhelpapp.ui.theme.Amber
import com.florea_gabriel.impairedhelpapp.ui.theme.AppDimensions
import com.florea_gabriel.impairedhelpapp.ui.theme.Indigo
import com.florea_gabriel.impairedhelpapp.ui.theme.IndigoDark
import com.florea_gabriel.impairedhelpapp.ui.theme.Rose
import com.florea_gabriel.impairedhelpapp.utils.Constants
import com.florea_gabriel.impairedhelpapp.utils.ProximityFeedback
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

private const val TAG = "CameraScreen"

// Center band of the depth map scanned for obstacles (fraction of the full frame).
// Full height is used again because the floor is removed geometrically (see
// getCenterObstacleDistance), not by cropping the bottom of the frame.
private const val CENTER_REGION_WIDTH_FRACTION = 0.25f
private const val CENTER_REGION_HEIGHT_FRACTION = 1.0f

// ── Floor-vs-obstacle separation (values in CALIBRATED meters) ────────────────
// An obstacle is a (near) vertical surface, so its depth stays ~constant over a
// vertical span. The floor is a depth ramp, so its depth drifts out of this
// tolerance over the same span and never forms a long enough constant run.
private const val FLAT_TOL_M = 0.08f                 // max depth spread within one vertical run
private const val VERTICAL_RUN_FRACTION = 0.15f      // a run must span ≥ this fraction of the band height
private const val MIN_OBSTACLE_COLUMNS_FRACTION = 0.10f // ≥ this fraction of columns must agree (cluster)
private const val OBSTACLE_DISTANCE_PERCENTILE = 0.10f  // report this percentile of obstacle depths, not the min

// Auto Gemini guidance on sustained danger
private const val AUTO_GUIDANCE_SUSTAINED_MS = 1500L   // danger must persist this long
private const val AUTO_GUIDANCE_COOLDOWN_MS = 10_000L  // min interval between Gemini calls

@Composable
fun CameraScreen(
        isScreenVisible: Boolean = true,
        bottomPadding: Dp = 0.dp,
        triggerSceneDescription: Boolean = false,
        onSceneDescriptionComplete: (() -> Unit)? = null,
        onSceneDescription: ((String) -> Unit)? = null,
        modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val currentIsScreenVisible by rememberUpdatedState(isScreenVisible)

    // ========== ML Models ==========
    var depthEstimator by remember { mutableStateOf<DepthEstimator?>(null) }
    val proximityFeedback = remember { ProximityFeedback(context) }
    val imageCaptioningManager = remember { ImageCaptioningManager() }

    var lastCapturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isDescribingScene by remember { mutableStateOf(false) }
    val descriptionScope = rememberCoroutineScope()

    // Initialize depth estimator
    LaunchedEffect(Unit) {
        Log.d(TAG, "Initializing DepthEstimator...")
        kotlinx.coroutines.withContext(Dispatchers.Default) {
            depthEstimator = DepthEstimator(context)
        }
        Log.d(TAG, "DepthEstimator initialized successfully")
    }

    // Handle voice command trigger for scene description
    LaunchedEffect(triggerSceneDescription) {
        if (triggerSceneDescription && !isDescribingScene && lastCapturedBitmap != null) {
            isDescribingScene = true
            try {
                val result = imageCaptioningManager.describeScene(lastCapturedBitmap!!)
                result.fold(
                        onSuccess = { description ->
                            onSceneDescription?.invoke(description)
                        },
                        onFailure = { error ->
                            onSceneDescription?.invoke(
                                    "Nu am putut descrie scena. ${error.message}"
                            )
                        }
                )
            } finally {
                isDescribingScene = false
                onSceneDescriptionComplete?.invoke()
            }
        }
    }

    // ========== State ==========
    var cachedDepthResult by remember { mutableStateOf<DepthResult?>(null) }

    var fps by remember { mutableStateOf(0f) }
    var lastFrameTime by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var depthInferenceTime by remember { mutableLongStateOf(0L) }

    var showDepthMap by remember { mutableStateOf(false) }
    var depthMapBitmap by remember { mutableStateOf<Bitmap?>(null) }

    var enableProximityAlert by remember { mutableStateOf(true) }
    val currentEnableProximityAlert by rememberUpdatedState(enableProximityAlert)

    var frameCount by remember { mutableStateOf(0) }
    val depthSkipFrames = 3
    var isProcessing by remember { mutableStateOf(false) }

    // Auto-guidance state: track sustained-danger window and cooldown
    val dangerStartTime = remember { java.util.concurrent.atomic.AtomicLong(0L) }
    val lastGuidanceCallTime = remember { java.util.concurrent.atomic.AtomicLong(0L) }
    val isAutoGuidanceInFlight = remember { java.util.concurrent.atomic.AtomicBoolean(false) }

    // ========== Camera Setup ==========
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val mlScope = remember { CoroutineScope(SupervisorJob() + Dispatchers.Default) }
    val isActive = remember { java.util.concurrent.atomic.AtomicBoolean(true) }

    val previewView = remember {
        PreviewView(context).apply {
            layoutParams =
                    ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                    )
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(lifecycleOwner) {
        Log.d(TAG, "Setting up camera...")

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener(
                {
                    try {
                        val cameraProvider = cameraProviderFuture.get()
                        cameraProvider.unbindAll()

                        val preview =
                                Preview.Builder().build().also {
                                    it.setSurfaceProvider(previewView.surfaceProvider)
                                }

                        val imageAnalyzer =
                                ImageAnalysis.Builder()
                                        .setBackpressureStrategy(
                                                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                        )
                                        .build()
                                        .also { analysis ->
                                            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                                                if (!isActive.get() || !currentIsScreenVisible) {
                                                    imageProxy.close()
                                                    return@setAnalyzer
                                                }

                                                val currentTime = System.currentTimeMillis()
                                                val timeDiff = currentTime - lastFrameTime
                                                lastFrameTime = currentTime
                                                fps = if (timeDiff > 0) 1000f / timeDiff else 0f

                                                if (isProcessing) {
                                                    imageProxy.close()
                                                    return@setAnalyzer
                                                }

                                                isProcessing = true
                                                frameCount++

                                                val shouldRunDepth = frameCount % depthSkipFrames == 0
                                                val currentDepthEstimator = depthEstimator

                                                mlScope.launch(Dispatchers.Default) {
                                                    try {
                                                        val bitmap =
                                                                ImageProcessor.imageProxyToBitmap(
                                                                        imageProxy
                                                                )
                                                        imageProxy.close()

                                                        lastCapturedBitmap = bitmap

                                                        // Run depth estimation
                                                        var depthResult: DepthResult? = cachedDepthResult

                                                        if (currentDepthEstimator != null && shouldRunDepth) {
                                                            depthResult =
                                                                    currentDepthEstimator
                                                                            .estimateDepth(bitmap)
                                                                            ?: cachedDepthResult
                                                        }

                                                        // Generate depth visualization if enabled
                                                        val newDepthBitmap =
                                                                if (showDepthMap && depthResult != null) {
                                                                    depthResultToHeatmapBitmap(depthResult)
                                                                } else {
                                                                    null
                                                                }

                                                        // Proximity alert from center region of depth map
                                                        if (currentEnableProximityAlert && depthResult != null) {
                                                            val centerDistance = getCenterObstacleDistance(depthResult)
                                                            if (proximityFeedback.shouldTriggerAlert(centerDistance)) {
                                                                proximityFeedback.triggerWarning(centerDistance)
                                                            }

                                                            // Auto Gemini guidance on sustained danger (gated by cooldown)
                                                            maybeTriggerAutoGuidance(
                                                                centerDistance = centerDistance,
                                                                bitmap = bitmap,
                                                                dangerStartTime = dangerStartTime,
                                                                lastGuidanceCallTime = lastGuidanceCallTime,
                                                                inFlight = isAutoGuidanceInFlight,
                                                                isDescribingScene = isDescribingScene,
                                                                scope = mlScope,
                                                                captioningManager = imageCaptioningManager,
                                                                onGuidance = { text -> onSceneDescription?.invoke(text) }
                                                            )
                                                        }

                                                        launch(Dispatchers.Main) {
                                                            if (depthResult != null && shouldRunDepth) {
                                                                cachedDepthResult = depthResult
                                                                depthInferenceTime = depthResult.inferenceTime
                                                            }
                                                            if (newDepthBitmap != null) {
                                                                depthMapBitmap = newDepthBitmap
                                                            }
                                                            isProcessing = false
                                                        }
                                                    } catch (e: Exception) {
                                                        Log.e(TAG, "Error processing frame: ${e.message}")
                                                        imageProxy.close()
                                                        isProcessing = false
                                                    }
                                                }
                                            }
                                        }

                        cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalyzer
                        )
                        Log.d(TAG, "Camera initialized successfully")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error setting up camera: ${e.message}", e)
                    }
                },
                ContextCompat.getMainExecutor(context)
        )

        onDispose {
            Log.d(TAG, "Disposing CameraScreen...")
            try {
                isActive.set(false)
                cameraExecutor.shutdown()
                cameraExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)
                mlScope.cancel()
                cameraProviderFuture.get().unbindAll()
                val depth = depthEstimator
                val prox = proximityFeedback
                val caption = imageCaptioningManager
                Thread {
                    Thread.sleep(500)
                    depth?.close()
                    prox.release()
                    caption.release()
                    Log.d(TAG, "CameraScreen ML models closed (deferred)")
                }.start()
                Log.d(TAG, "CameraScreen disposed successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error disposing camera: ${e.message}")
            }
        }
    }

    // ========== UI Layers ==========
    Box(modifier = modifier.fillMaxSize()) {
        // Layer 1: Camera Preview
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        // Layer 2: Depth Map Overlay
        if (showDepthMap && depthMapBitmap != null) {
            Image(
                    bitmap = depthMapBitmap!!.asImageBitmap(),
                    contentDescription = "Depth Map Visualization",
                    modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.3f)),
                    alpha = 0.7f
            )
        }

        // Layer 3: Performance Metrics
        PerformanceMetrics(
                fps = fps,
                detectionCount = 0,
                depthInferenceTime = depthInferenceTime,
                showDepthMetrics = true
        )

        // Layer 4: Toggle Buttons (bottom-right)
        Column(
                modifier =
                        Modifier.align(Alignment.BottomEnd)
                                .padding(end = AppDimensions.spaceM, bottom = 120.dp + bottomPadding),
                verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
        ) {
            // Scene Description Button
            FloatingActionButton(
                    onClick = {
                        if (!isDescribingScene && lastCapturedBitmap != null) {
                            isDescribingScene = true
                            descriptionScope.launch {
                                try {
                                    val result =
                                            imageCaptioningManager.describeScene(
                                                    lastCapturedBitmap!!
                                            )
                                    result.fold(
                                            onSuccess = { description ->
                                                onSceneDescription?.invoke(description)
                                            },
                                            onFailure = { error ->
                                                onSceneDescription?.invoke(
                                                        "Nu am putut descrie scena. ${error.message}"
                                                )
                                            }
                                    )
                                } finally {
                                    isDescribingScene = false
                                }
                            }
                        }
                    },
                    modifier =
                            Modifier.size(56.dp)
                                    .then(
                                            if (isDescribingScene) {
                                                Modifier.border(2.dp, Color.White, CircleShape)
                                            } else {
                                                Modifier
                                            }
                                    )
                                    .semantics {
                                        contentDescription =
                                                "Descrie camera - apasă pentru a auzi o descriere a încăperii"
                                    },
                    shape = CircleShape,
                    containerColor =
                            if (isDescribingScene) IndigoDark else Indigo
            ) {
                if (isDescribingScene) {
                    CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = Color.White,
                            strokeWidth = 2.dp
                    )
                } else {
                    Icon(
                            imageVector = Icons.Default.Description,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(28.dp)
                    )
                }
            }

            // Depth Map Toggle
            FloatingActionButton(
                    onClick = { showDepthMap = !showDepthMap },
                    modifier =
                            Modifier.size(56.dp)
                                    .then(
                                            if (showDepthMap) {
                                                Modifier.border(2.dp, Color.White, CircleShape)
                                            } else {
                                                Modifier
                                            }
                                    )
                                    .semantics {
                                        contentDescription =
                                                if (showDepthMap) "Ascunde harta de adâncime"
                                                else "Arată harta de adâncime"
                                    },
                    shape = CircleShape,
                    containerColor = if (showDepthMap) Amber else Color.Black.copy(alpha = 0.6f)
            ) {
                Icon(
                        imageVector = Icons.Default.Layers,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(28.dp)
                )
            }

            // Proximity Alert Toggle
            FloatingActionButton(
                    onClick = { enableProximityAlert = !enableProximityAlert },
                    modifier =
                            Modifier.size(56.dp)
                                    .then(
                                            if (enableProximityAlert) {
                                                Modifier.border(2.dp, Color.White, CircleShape)
                                            } else {
                                                Modifier
                                            }
                                    )
                                    .semantics {
                                        contentDescription =
                                                if (enableProximityAlert) "Oprește alertele"
                                                else "Pornește alertele"
                                    },
                    shape = CircleShape,
                    containerColor =
                            if (enableProximityAlert) Rose else Color.Black.copy(alpha = 0.6f)
            ) {
                Icon(
                        imageVector = Icons.Default.Vibration,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(28.dp)
                )
            }
        }
    }
}

/**
 * Distance to the closest real obstacle in the user's forward path.
 *
 * Replaces a naive "minimum metric depth in the center band" (which alarmed on
 * the single closest pixel) with a floor-aware, noise-robust estimate.
 *
 *  (B) Floor removal — an obstacle (table, wall, person) is a near-vertical
 *      surface, so along a column its depth stays ~constant over a vertical
 *      span. The floor, by contrast, is a depth ramp that recedes toward the
 *      top of the frame, so its depth drifts out of [FLAT_TOL_M] over the span
 *      and never forms a long enough constant run. We therefore keep only
 *      vertical runs of near-constant, near depth and ignore everything else.
 *      This rejects the ground when the phone is tilted down WITHOUT cropping
 *      the bottom of the frame (so low obstacles like a knee-height table are
 *      still detected).
 *
 *  (A) Robust aggregation — instead of the single closest pixel we require a
 *      cluster of columns to agree, then report a low percentile of the
 *      detected depths. Isolated int8 quantization spikes can no longer trigger
 *      the alarm.
 *
 * Algorithm, per column of the center band (calibrated meters throughout):
 *  1. Scan top→bottom, growing a "vertical run" while each pixel is near
 *     (< [ProximityFeedback.THRESHOLD_CAUTION]) and within [FLAT_TOL_M] of the
 *     run's anchor depth. The floor ramp breaks this tolerance and is dropped.
 *  2. A run that spans at least [VERTICAL_RUN_FRACTION] of the band height is
 *     accepted as a vertical surface; its mean depth is recorded and the column
 *     is marked as containing an obstacle.
 *  3. Only fire if at least [MIN_OBSTACLE_COLUMNS_FRACTION] of the columns agree
 *     (cluster gate); otherwise return no-obstacle.
 *  4. Report the [OBSTACLE_DISTANCE_PERCENTILE] percentile of all recorded run
 *     depths as the reported distance (robust to the few closest outliers).
 *
 * Tuning (all constants are near the top of this file):
 *  - Floor still leaking through → raise [VERTICAL_RUN_FRACTION] and/or
 *    [MIN_OBSTACLE_COLUMNS_FRACTION], or lower [FLAT_TOL_M].
 *  - Low/real obstacles missed → do the opposite.
 *
 * @return calibrated distance in meters to the nearest obstacle, or
 *         [Float.MAX_VALUE] when no obstacle is present (no alert).
 */
private fun getCenterObstacleDistance(depthResult: DepthResult): Float {
    val regionW = (depthResult.width * CENTER_REGION_WIDTH_FRACTION).toInt().coerceAtLeast(1)
    val regionH = (depthResult.height * CENTER_REGION_HEIGHT_FRACTION).toInt().coerceAtLeast(1)
    val startX = (depthResult.width - regionW) / 2
    val startY = (depthResult.height - regionH) / 2
    val endX = startX + regionW
    val endY = startY + regionH
    val cal = Constants.DEPTH_CALIBRATION_FACTOR
    val minRun = (regionH * VERTICAL_RUN_FRACTION).toInt().coerceAtLeast(4)

    val obstacleDepths = ArrayList<Float>()
    var columnsWithObstacle = 0

    for (x in startX until endX) {
        var runLen = 0
        var runSum = 0f
        var runRef = -1f            // anchor depth of the current vertical run
        var columnHit = false

        for (y in startY until endY) {
            val raw = depthResult.metricDepthMap[y][x]
            val d = if (raw > 0f) raw * cal else -1f
            val near = d in 0f..ProximityFeedback.THRESHOLD_CAUTION
            // A pixel continues the run only if it stays within tolerance of the
            // run's anchor — the floor ramp drifts away and breaks the run.
            val continues = near && runRef > 0f && kotlin.math.abs(d - runRef) <= FLAT_TOL_M

            if (continues) {
                runLen++
                runSum += d
            } else {
                if (runLen >= minRun) {
                    obstacleDepths.add(runSum / runLen)
                    columnHit = true
                }
                if (near) { runLen = 1; runSum = d; runRef = d }
                else { runLen = 0; runSum = 0f; runRef = -1f }
            }
        }
        if (runLen >= minRun) {
            obstacleDepths.add(runSum / runLen)
            columnHit = true
        }
        if (columnHit) columnsWithObstacle++
    }

    // (A) Require a cluster of columns, then report a low percentile (not the min).
    val minColumns = (regionW * MIN_OBSTACLE_COLUMNS_FRACTION).toInt().coerceAtLeast(3)
    if (columnsWithObstacle < minColumns || obstacleDepths.isEmpty()) return Float.MAX_VALUE

    obstacleDepths.sort()
    val idx = (obstacleDepths.size * OBSTACLE_DISTANCE_PERCENTILE).toInt()
        .coerceIn(0, obstacleDepths.size - 1)
    return obstacleDepths[idx]
}

/**
 * Sustained-danger + cooldown gate for auto Gemini guidance.
 * Fires `describeObstacleAndGuide` only when danger persists ≥ AUTO_GUIDANCE_SUSTAINED_MS
 * and at least AUTO_GUIDANCE_COOLDOWN_MS has elapsed since the last call.
 * Resets the danger window as soon as the path clears, so brief phone movements
 * past a near object don't accumulate toward a trigger.
 */
private fun maybeTriggerAutoGuidance(
    centerDistance: Float,
    bitmap: Bitmap,
    dangerStartTime: java.util.concurrent.atomic.AtomicLong,
    lastGuidanceCallTime: java.util.concurrent.atomic.AtomicLong,
    inFlight: java.util.concurrent.atomic.AtomicBoolean,
    isDescribingScene: Boolean,
    scope: CoroutineScope,
    captioningManager: ImageCaptioningManager,
    onGuidance: (String) -> Unit
) {
    val now = System.currentTimeMillis()
    val inDanger = centerDistance < ProximityFeedback.THRESHOLD_WARNING

    if (!inDanger) {
        dangerStartTime.set(0L)
        return
    }

    // Mark when the danger window started
    dangerStartTime.compareAndSet(0L, now)
    val sustainedFor = now - dangerStartTime.get()

    if (sustainedFor < AUTO_GUIDANCE_SUSTAINED_MS) return
    if (now - lastGuidanceCallTime.get() < AUTO_GUIDANCE_COOLDOWN_MS) return
    if (isDescribingScene) return
    if (!inFlight.compareAndSet(false, true)) return

    // Take cooldown timestamp now so concurrent frames are gated even before the call returns
    lastGuidanceCallTime.set(now)

    scope.launch(Dispatchers.IO) {
        try {
            val result = captioningManager.describeObstacleAndGuide(bitmap)
            result.fold(
                onSuccess = { text -> onGuidance(text) },
                onFailure = { /* swallow: proximity beep already alerts the user */ }
            )
        } finally {
            inFlight.set(false)
        }
    }
}
