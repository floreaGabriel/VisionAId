package com.florea_gabriel.impairedhelpapp.presentation.search

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.AudioManager
import android.media.Image
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.ar.core.*
import com.google.ar.core.CameraConfigFilter
import io.github.sceneview.ar.ARScene
import io.github.sceneview.ar.arcore.createAnchorOrNull
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.ar.rememberARCameraNode
import io.github.sceneview.math.Position
import io.github.sceneview.node.CubeNode
import io.github.sceneview.rememberCollisionSystem
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberMaterialLoader
import io.github.sceneview.rememberModelLoader
import io.github.sceneview.rememberNodes
import io.github.sceneview.rememberOnGestureListener
import io.github.sceneview.rememberView
import kotlinx.coroutines.*
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity

private const val TAG = "ARSearchScreen"

// Arrival is judged on HORIZONTAL distance (ground plane), so an object directly
// below/under the user still counts as reached — see guidance block.
private const val ARRIVAL_HORIZONTAL_DISTANCE_M = 0.6f
// Delay before auto-returning to the object list, so the arrival message is heard.
private const val ARRIVAL_RETURN_DELAY_MS = 2500L
// Time the empty AR scene keeps rendering after nodes are removed, so Filament frees
// the marker's GPU resources BEFORE the engine is destroyed (avoids PreconditionPanic).
private const val AR_TEARDOWN_DRAIN_MS = 350L

/**
 * AR-enabled search screen using SceneView.
 *
 * Features:
 * - Uses ARScene for camera and AR rendering
 * - Extracts frames from AR session for object detection
 * - Places 3D markers that stay fixed in space
 * - Provides spatial audio guidance to the marker
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ARSearchScreen(
    objectName: String,
    // Driven by the parent: when it flips to true (back / top-bar back / arrival),
    // this screen runs a graceful AR teardown and then calls [onTornDown].
    shuttingDown: Boolean,
    // Called once the user has reached the object (after the spoken announcement).
    onArrived: () -> Unit,
    // Called after the AR scene has been safely dismantled — parent may now navigate.
    onTornDown: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val configuration = LocalConfiguration.current
    val density = LocalDensity.current

    // Screen dimensions in pixels
    val screenWidthPx = with(density) { configuration.screenWidthDp.dp.roundToPx() }
    val screenHeightPx = with(density) { configuration.screenHeightDp.dp.roundToPx() }

    // SceneView components
    val engine = rememberEngine()
    val view = rememberView(engine)
    val modelLoader = rememberModelLoader(engine)
    val materialLoader = rememberMaterialLoader(engine)
    val cameraNode = rememberARCameraNode(engine)
    val collisionSystem = rememberCollisionSystem(view)
    val childNodes = rememberNodes()

    // Lifecycle guard — prevents processing after dispose starts
    val isActive = remember { AtomicBoolean(true) }
    val isProcessingAtomic = remember { AtomicBoolean(false) }

    // AR state
    var anchorNode by remember { mutableStateOf<AnchorNode?>(null) }
    var trackingStatus by remember { mutableStateOf("Initializare...") }
    var planeDetected by remember { mutableStateOf(false) }
    var currentFrame by remember { mutableStateOf<Frame?>(null) }

    // Search state
    var searchState by remember { mutableStateOf(SearchState.SCANNING) }
    var objectSimilarity by remember { mutableStateOf(0f) }
    var emaConfidence by remember { mutableStateOf(0f) }
    var lastDetectionTime by remember { mutableStateOf(0L) }
    var markerPlaced by remember { mutableStateOf(false) }
    var shouldPlaceMarker by remember { mutableStateOf(false) }
    // Save detected object position for marker placement
    var detectedObjectX by remember { mutableStateOf(0.5f) }
    var detectedObjectY by remember { mutableStateOf(0.5f) }

    // Object finder with YOLO + CLIP pipeline
    var objectFinder by remember { mutableStateOf<FullObjectMatcher?>(null) }

    // State transition tracking for TTS cues
    var previousSearchState by remember { mutableStateOf(SearchState.SCANNING) }
    var spokeEmaAlmostSure by remember { mutableStateOf(false) }

    // Sonification (Geiger-counter beeps during FOUND)
    var beepJob by remember { mutableStateOf<Job?>(null) }

    // Audio/Haptic feedback
    val vibrator = remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            manager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }
    val toneGenerator = remember { ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100) }

    // TTS
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }
    var lastSpokenTime by remember { mutableStateOf(0L) }

    // Guidance to marker
    var guidanceText by remember { mutableStateOf("Caut $objectName...") }
    var distanceToMarker by remember { mutableStateOf<Float?>(null) }
    var angleToMarker by remember { mutableStateOf<Float?>(null) }
    var lastVibrationTime by remember { mutableStateOf(0L) }
    var isMarkerInFront by remember { mutableStateOf(false) }

    // Initialize TTS
    LaunchedEffect(Unit) {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale("ro", "RO")
                tts?.setSpeechRate(1.15f)
                tts?.setPitch(0.8f)
                tts?.speak("Caut $objectName. Mișcă telefonul încet.", TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }
    }

    // Initialize object matcher with YOLO + CLIP
    LaunchedEffect(objectName) {
        withContext(Dispatchers.IO) {
            try {
                objectFinder = FullObjectMatcher(context, objectName)
                objectFinder?.initialize()
                Log.d(TAG, "Full object matcher (YOLO+CLIP) initialized for: $objectName")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize object matcher: ${e.message}")
            }
        }
    }

    // Stable coroutine scope for frame processing (survives recomposition)
    val processingScope = remember { CoroutineScope(Dispatchers.Default + SupervisorJob()) }
    var isProcessing by remember { mutableStateOf(false) }

    // Graceful AR teardown. The parent never removes this screen abruptly; instead it
    // flips [shuttingDown] (on back, top-bar back, or arrival) and we dismantle the AR
    // scene in three phases. Removing the composable while the marker is still attached
    // destroys the Filament engine with live GPU resources → uncatchable native
    // PreconditionPanic (SIGABRT). Here we instead empty the scene first, let SceneView
    // render a few frames to release those resources, and only then signal the parent.
    LaunchedEffect(shuttingDown) {
        if (shuttingDown) {
            isActive.set(false)          // stop frame processing immediately
            tts?.stop()
            beepJob?.cancel()
            beepJob = null

            val node = anchorNode
            anchorNode = null
            if (node != null) {
                // Phase 1: stop rendering the marker (remove it from the scene list),
                // then give the render thread one frame to drop it.
                try { childNodes.remove(node) } catch (_: Exception) {}
                delay(80)
                // Phase 2: detach the ARCore anchor and EXPLICITLY destroy the marker's
                // child renderables (the cube). Node.destroy() does NOT recurse to
                // children, so the cube must be destroyed by hand. This has to happen
                // BEFORE the MaterialLoader disposes on exit, otherwise Filament aborts:
                // "destroying MaterialInstance 'Transparent Colored' which is still in
                // use by Renderable" → uncatchable native SIGABRT.
                try { node.anchor?.detach() } catch (_: Exception) {}
                try { node.childNodes.toList().forEach { runCatching { it.destroy() } } } catch (_: Exception) {}
                try { node.destroy() } catch (_: Exception) {}
            }
            // Phase 3: let the now-empty scene render a few frames before the parent
            // removes us from composition (which destroys the Filament engine).
            delay(AR_TEARDOWN_DRAIN_MS)
            onTornDown()
        }
    }

    // Frame processing - only start new processing if not already running
    LaunchedEffect(currentFrame) {
        if (!isActive.get()) return@LaunchedEffect
        val frame = currentFrame ?: return@LaunchedEffect
        val finder = objectFinder ?: return@LaunchedEffect

        // Try to place marker if flagged (using FRESH frame)
        if (shouldPlaceMarker && !markerPlaced) {
            Log.d(TAG, "Attempting marker placement at object pos=(${String.format("%.2f", detectedObjectX)},${String.format("%.2f", detectedObjectY)})")

            // Convert detected object position (normalized 0-1) to screen pixels
            // Use object position, not screen center — places marker where the object actually is
            val hitX = detectedObjectX * screenWidthPx
            val hitY = detectedObjectY * screenHeightPx
            Log.d(TAG, "HitTest at screen (${hitX.toInt()}, ${hitY.toInt()}) of ${screenWidthPx}x${screenHeightPx}")

            // Try regular hit test at object position
            var hitResult = frame.hitTest(hitX, hitY).firstOrNull { hit ->
                val trackable = hit.trackable
                when {
                    trackable is Plane && trackable.isPoseInPolygon(hit.hitPose) -> true
                    else -> hit.distance < 3.0f
                }
            }

            // Fallback 1: try screen center
            if (hitResult == null) {
                hitResult = frame.hitTest(screenWidthPx / 2f, screenHeightPx / 2f).firstOrNull { hit ->
                    hit.trackable is Plane
                }
            }

            // Fallback 2: InstantPlacement at object position
            if (hitResult == null) {
                Log.d(TAG, "No regular hit, trying InstantPlacement...")
                try {
                    hitResult = frame.hitTestInstantPlacement(hitX, hitY, 1.5f).firstOrNull()
                    if (hitResult != null) {
                        Log.d(TAG, "InstantPlacement hit at ${hitResult.distance}m")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "InstantPlacement failed: ${e.message}")
                }
            }

            if (hitResult != null) {
                val anchor = hitResult.createAnchorOrNull()
                if (anchor != null) {
                    anchorNode = AnchorNode(
                        engine = engine,
                        anchor = anchor
                    ).apply {
                        addChildNode(
                            CubeNode(
                                engine = engine,
                                size = Position(0.08f, 0.08f, 0.08f),
                                materialInstance = materialLoader.createColorInstance(
                                    color = Color.Green.copy(alpha = 0.9f)
                                )
                            )
                        )
                    }
                    childNodes += anchorNode!!
                    markerPlaced = true
                    shouldPlaceMarker = false
                    searchState = SearchState.LOCKED

                    tts?.speak("Obiect găsit și marcat!", TextToSpeech.QUEUE_FLUSH, null, null)
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 300)
                    Log.i(TAG, "Marker placed successfully!")
                }
            } else {
                Log.w(TAG, "Still no hit result, will retry next frame")
            }
        }

        // Update marker guidance (fast, no CLIP needed)
        if (markerPlaced) {
            anchorNode?.let { node ->
                val anchor = node.anchor ?: return@let
                if (anchor.trackingState == TrackingState.TRACKING) {
                    val cameraPose = frame.camera.pose
                    val anchorPose = anchor.pose

                    // Vector from camera to anchor in WORLD space
                    val worldDx = anchorPose.tx() - cameraPose.tx()
                    val worldDy = anchorPose.ty() - cameraPose.ty()
                    val worldDz = anchorPose.tz() - cameraPose.tz()
                    val distance = kotlin.math.sqrt(worldDx * worldDx + worldDy * worldDy + worldDz * worldDz)
                    distanceToMarker = distance

                    // HORIZONTAL (ground-plane) distance — ignores the vertical
                    // offset, so an object directly below the user still counts as
                    // reached and left/right is judged only on the walking plane.
                    val horizDistance = kotlin.math.sqrt(worldDx * worldDx + worldDz * worldDz)

                    // Bearing is computed in the WORLD-HORIZONTAL plane, NOT in the
                    // tilting camera-local frame. The previous camera-local angle was
                    // only correct with the phone held upright: when the phone pitched
                    // toward the floor, the marker's vertical offset leaked into the
                    // left/right axis and the direction flipped. Here we project the
                    // camera's forward (-Z) onto the ground plane and measure the signed
                    // angle to the marker's horizontal bearing → immune to tilt/roll.
                    val cameraRotation = cameraPose.rotationQuaternion
                    val fwdWorld = rotateVectorByQuaternion(
                        floatArrayOf(0f, 0f, -1f),
                        cameraRotation
                    )
                    var fwdX = fwdWorld[0]
                    var fwdZ = fwdWorld[2]
                    val fwdLen = kotlin.math.sqrt(fwdX * fwdX + fwdZ * fwdZ)
                    if (fwdLen > 1e-4f) { fwdX /= fwdLen; fwdZ /= fwdLen }

                    // Signed horizontal angle: + = marker to the right, − = left.
                    val dot = fwdX * worldDx + fwdZ * worldDz       // ahead component
                    val cross = fwdX * worldDz - fwdZ * worldDx     // sideways component
                    val angleDegrees = Math.toDegrees(
                        kotlin.math.atan2(cross.toDouble(), dot.toDouble())
                    ).toFloat()
                    angleToMarker = angleDegrees

                    // Marker ahead of the horizontal heading? (positive ahead component)
                    val isInFront = dot > 0f
                    isMarkerInFront = isInFront
                    val absAngle = kotlin.math.abs(angleDegrees)

                    // Arrival on horizontal distance — object below you still counts.
                    val arrived = horizDistance < ARRIVAL_HORIZONTAL_DISTANCE_M

                    // Update guidance text. Inside the arrival radius we stop giving
                    // left/right (the bearing is degenerate that close) and announce arrival.
                    guidanceText = when {
                        arrived -> "Ai ajuns!"
                        !isInFront -> "Întoarce-te - markerul e în spate"
                        absAngle > 60 -> if (angleDegrees > 0) "Întoarce-te mult la dreapta" else "Întoarce-te mult la stânga"
                        absAngle > 30 -> if (angleDegrees > 0) "La dreapta" else "La stânga"
                        absAngle > 10 -> if (angleDegrees > 0) "Puțin la dreapta" else "Puțin la stânga"
                        else -> "Înainte - ${String.format("%.1f", distance)}m"
                    }

                    // VIBRATION: Only when marker is in front AND close to center
                    val now = System.currentTimeMillis()
                    if (isInFront && absAngle < 45) {
                        // Vibration intensity increases as you get closer
                        // Distance 3m = weak, Distance 0.5m = strong
                        val proximityFactor = (1f - (distance / 3f).coerceIn(0f, 1f))
                        val intensity = (proximityFactor * 200 + 55).toInt().coerceIn(1, 255)

                        // Vibrate every 200-500ms based on proximity (closer = faster)
                        val vibrationInterval = (500 - proximityFactor * 300).toLong().coerceIn(150, 500)

                        if (now - lastVibrationTime > vibrationInterval) {
                            lastVibrationTime = now
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                vibrator.vibrate(VibrationEffect.createOneShot(50, intensity))
                            } else {
                                @Suppress("DEPRECATION")
                                vibrator.vibrate(50)
                            }
                        }
                    }
                    // No vibration when marker is behind or too far to the side

                    // Voice guidance every 4 seconds
                    if (now - lastSpokenTime > 4000) {
                        lastSpokenTime = now
                        tts?.speak(guidanceText, TextToSpeech.QUEUE_FLUSH, null, null)
                    }

                    // ARRIVED: announce, then auto-return to the object list.
                    if (arrived && searchState != SearchState.ARRIVED) {
                        searchState = SearchState.ARRIVED
                        lastSpokenTime = now
                        toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 500)
                        tts?.speak("Ai ajuns la $objectName!", TextToSpeech.QUEUE_FLUSH, null, "arrived")

                        // Strong vibration burst for arrival
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                            vibrator.vibrate(VibrationEffect.createOneShot(300, 255))
                        }

                        // After the message is heard, ask the parent to exit. This flips
                        // `shuttingDown`, which runs the graceful AR teardown before navigating.
                        scope.launch {
                            delay(ARRIVAL_RETURN_DELAY_MS)
                            onArrived()
                        }
                    }
                }
            }
            return@LaunchedEffect
        }

        // Skip if already processing, not tracking, or shutting down
        if (isProcessing) return@LaunchedEffect
        if (!isActive.get()) return@LaunchedEffect
        if (frame.camera.trackingState != TrackingState.TRACKING) return@LaunchedEffect

        val now = System.currentTimeMillis()
        if (now - lastDetectionTime < 200) return@LaunchedEffect // Rate limit to 5fps (fast path skips CLIP)
        lastDetectionTime = now

        // Extract bitmap BEFORE launching coroutine (must be on AR thread)
        val bitmap: Bitmap?
        try {
            val image = frame.acquireCameraImage()
            bitmap = imageToBitmap(image)
            image.close()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to acquire camera image: ${e.message}")
            return@LaunchedEffect
        }

        if (bitmap == null) return@LaunchedEffect

        // Launch YOLO + CLIP processing in stable scope
        isProcessing = true
        isProcessingAtomic.set(true)
        processingScope.launch {
            try {
                // Bail out early if screen is being disposed
                if (!isActive.get()) {
                    bitmap.recycle()
                    return@launch
                }

                val matchResult = finder.matchObject(bitmap)
                bitmap.recycle()

                // Don't update UI if we're shutting down
                if (!isActive.get()) return@launch

                val similarity = matchResult.similarity
                Log.d(TAG, "Match result: ${matchResult.label} = ${String.format("%.3f", similarity)}")

                // Update UI state on main thread — EMA confidence tracker
                withContext(Dispatchers.Main) {
                    objectSimilarity = similarity

                    val meetsThreshold = similarity > (objectFinder?.matchThreshold ?: 0.60f)
                    val isSkippable = matchResult.noUsableCandidates || similarity == 0f
                    val oldState = searchState

                    // EMA update: confidence rises fast on hits, decays slowly on misses
                    val alpha = 0.4f  // smoothing factor
                    when {
                        meetsThreshold -> {
                            // Hit: boost confidence (weighted by similarity²)
                            val weightedSim = similarity * similarity
                            emaConfidence = alpha * weightedSim + (1 - alpha) * emaConfidence
                            searchState = SearchState.FOUND
                            // Save object position for marker placement
                            detectedObjectX = matchResult.centerX
                            detectedObjectY = matchResult.centerY
                            Log.d(TAG, "✅ HIT ${matchResult.label} sim=${"%.3f".format(similarity)} ema=${"%.3f".format(emaConfidence)}")

                            // Vibrate on HIT — object is in frame right now
                            // Intensity proportional to EMA (stronger = more confident)
                            val intensity = (emaConfidence * 255).toInt().coerceIn(80, 255)
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                vibrator.vibrate(VibrationEffect.createOneShot(50, intensity))
                            } else {
                                @Suppress("DEPRECATION")
                                vibrator.vibrate(50)
                            }

                            // "Almost sure" TTS cue at EMA 0.35
                            if (emaConfidence >= 0.35f && !spokeEmaAlmostSure) {
                                spokeEmaAlmostSure = true
                                tts?.speak("Aproape sigur", TextToSpeech.QUEUE_FLUSH, null, null)
                            }

                            // Lock when EMA confidence is high enough
                            if (emaConfidence >= 0.55f && !shouldPlaceMarker) {
                                Log.i(TAG, "🎯 EMA confidence ${"%.3f".format(emaConfidence)} ≥ 0.55! Placing marker...")
                                shouldPlaceMarker = true
                            }
                        }
                        isSkippable -> {
                            // Neutral: very gentle decay (person-only, 0 detections)
                            emaConfidence *= 0.95f
                            Log.d(TAG, "⏸️ Skip (noUsable=${matchResult.noUsableCandidates}) ema=${"%.3f".format(emaConfidence)}")
                        }
                        else -> {
                            // Miss: moderate decay (doesn't destroy progress like consecutive reset)
                            emaConfidence *= 0.75f
                            Log.d(TAG, "✗ MISS ${matchResult.label} sim=${"%.3f".format(similarity)} ema=${"%.3f".format(emaConfidence)}")
                            if (emaConfidence < 0.15f && searchState == SearchState.FOUND) {
                                searchState = SearchState.LOST
                            }
                            // Reset "almost sure" flag on decay
                            if (emaConfidence < 0.30f) {
                                spokeEmaAlmostSure = false
                            }
                        }
                    }

                    // M5a: TTS on state transitions
                    if (searchState != oldState) {
                        when {
                            oldState == SearchState.SCANNING && searchState == SearchState.FOUND -> {
                                tts?.speak("Posibil detectat", TextToSpeech.QUEUE_ADD, null, null)
                            }
                            oldState == SearchState.FOUND && searchState == SearchState.LOST -> {
                                tts?.speak("Pierdut", TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                            oldState == SearchState.LOST && searchState == SearchState.FOUND -> {
                                tts?.speak("Regăsit", TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        }

                        // M5b: Manage beep job on state transitions
                        if (searchState == SearchState.FOUND && !markerPlaced) {
                            // Start confidence beeps — faster beeps = higher confidence
                            if (beepJob == null || beepJob?.isActive != true) {
                                beepJob = scope.launch {
                                    while (isActive.get()) {
                                        // Interval: EMA 0.5+ → 150ms (fast), EMA ~0.2 → 800ms (slow)
                                        val interval = (800 - emaConfidence * 1200).toLong().coerceIn(150, 800)
                                        try {
                                            toneGenerator.startTone(ToneGenerator.TONE_DTMF_1, 60)
                                        } catch (_: Exception) { }
                                        delay(interval)
                                    }
                                }
                            }
                        } else if (searchState != SearchState.FOUND) {
                            // Stop beeps when leaving FOUND state
                            beepJob?.cancel()
                            beepJob = null
                        }

                        previousSearchState = searchState
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Frame processing error: ${e.message}")
            } finally {
                isProcessing = false
                isProcessingAtomic.set(false)
            }
        }
    }

    // Cleanup — waits for native ONNX inference to finish before releasing models.
    // OrtSession.run() is a blocking JNI call that can't be cancelled by coroutine
    // cancellation. Closing the session while run() is active destroys a mutex
    // that native code still holds → SIGABRT. So we wait on a background thread.
    DisposableEffect(Unit) {
        onDispose {
            isActive.set(false)
            beepJob?.cancel()
            // AR nodes are already released in the graceful teardown (LaunchedEffect on
            // `shuttingDown`); here we only release the heavy ML/native resources.

            // Capture refs before composable state is torn down
            val finderRef = objectFinder
            val ttsRef = tts
            val scopeRef = processingScope

            // Release on a background thread that waits for in-flight inference
            Thread {
                // Poll until ONNX inference finishes (max 3s)
                var waited = 0
                while (isProcessingAtomic.get() && waited < 3000) {
                    Thread.sleep(50)
                    waited += 50
                }
                // Now safe: no native code is using the session
                scopeRef.cancel()
                finderRef?.release()
                ttsRef?.shutdown()
                try { toneGenerator.release() } catch (_: Exception) {}
                Log.d(TAG, "Cleanup complete (waited ${waited}ms for inference)")
            }.start()
        }
    }

    // UI
    Box(modifier = Modifier.fillMaxSize().padding(top = 48.dp, bottom = 80.dp)) {
        ARScene(
            modifier = Modifier.fillMaxSize(),
            engine = engine,
            view = view,
            modelLoader = modelLoader,
            cameraNode = cameraNode,
            collisionSystem = collisionSystem,
            childNodes = childNodes,
            sessionConfiguration = { session, config ->
                // Depth & light estimation disabled — saves ~25-30% GPU
                // We don't use ARCore depth (we have our own model) and the green cube doesn't need HDR lighting
                config.depthMode = Config.DepthMode.DISABLED
                config.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
                config.lightEstimationMode = Config.LightEstimationMode.DISABLED
                config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL

                // Select 1280x720 camera — good enough for YOLO (resizes to 640x640 internally)
                // Avoids max resolution which wastes CPU on YUV→Bitmap conversion
                try {
                    val filter = CameraConfigFilter(session)
                    val configs = session.getSupportedCameraConfigs(filter)
                    val targetPixels = 1280 * 720
                    val bestConfig = configs.minByOrNull {
                        val pixels = it.imageSize.width * it.imageSize.height
                        // Prefer closest to 1280x720 that's >= 720p
                        if (pixels >= 640 * 480) kotlin.math.abs(pixels - targetPixels) else Int.MAX_VALUE
                    }
                    if (bestConfig != null) {
                        session.cameraConfig = bestConfig
                        Log.d(TAG, "📷 ARCore camera: ${bestConfig.imageSize.width}x${bestConfig.imageSize.height}")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not set camera config: ${e.message}")
                }
            },
            planeRenderer = !markerPlaced, // Hide planes once marker is placed
            onSessionUpdated = { session, updatedFrame ->
                currentFrame = updatedFrame

                // Update tracking status
                trackingStatus = when (updatedFrame.camera.trackingState) {
                    TrackingState.TRACKING -> "Tracking activ"
                    TrackingState.PAUSED -> "Tracking în pauză"
                    TrackingState.STOPPED -> "Tracking oprit"
                }

                // After marker is placed, disable plane finding to save CPU/GPU
                // We only need tracking to keep the anchor stable, not to find new planes
                if (markerPlaced && session.config.planeFindingMode != Config.PlaneFindingMode.DISABLED) {
                    val updatedConfig = session.config
                    updatedConfig.planeFindingMode = Config.PlaneFindingMode.DISABLED
                    session.configure(updatedConfig)
                    Log.d(TAG, "⚡ Plane finding disabled after marker placement (saving battery)")
                }

                // Check for planes (only if still searching)
                if (!markerPlaced) {
                    val planes = session.getAllTrackables(Plane::class.java)
                    planeDetected = planes.any { it.trackingState == TrackingState.TRACKING }
                }
            },
            onGestureListener = rememberOnGestureListener(
                onSingleTapConfirmed = { _, _ ->
                    // Manual tap to place marker (fallback)
                    if (!markerPlaced && searchState == SearchState.FOUND) {
                        val frame = currentFrame ?: return@rememberOnGestureListener
                        val centerX = screenWidthPx / 2f
                        val centerY = screenHeightPx / 2f
                        val hitResults = frame.hitTest(centerX, centerY)

                        hitResults.firstOrNull()?.createAnchorOrNull()?.let { anchor ->
                            anchorNode = AnchorNode(
                                engine = engine,
                                anchor = anchor
                            ).apply {
                                addChildNode(
                                    CubeNode(
                                        engine = engine,
                                        size = Position(0.08f, 0.08f, 0.08f),
                                        materialInstance = materialLoader.createColorInstance(
                                            color = Color.Green.copy(alpha = 0.9f)
                                        )
                                    )
                                )
                            }
                            childNodes += anchorNode!!
                            markerPlaced = true
                            searchState = SearchState.LOCKED
                            tts?.speak("Marker plasat manual.", TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                    }
                }
            )
        )

        // Status overlay (below the HomeScreen top bar)
        StatusOverlay(
            searchState = searchState,
            similarity = objectSimilarity,
            consecutiveDetections = (emaConfidence * 100).toInt(),
            trackingStatus = trackingStatus,
            planeDetected = planeDetected,
            markerPlaced = markerPlaced,
            modifier = Modifier.align(Alignment.TopCenter).padding(top = 16.dp)
        )

        // Direction Arrow (Seeing AI style) - only show when marker is placed
        if (markerPlaced && angleToMarker != null) {
            DirectionArrow(
                angle = angleToMarker!!,
                isInFront = isMarkerInFront,
                distance = distanceToMarker ?: 0f,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 120.dp)
            )
        }

        // Guidance at bottom
        GuidanceCard(
            text = guidanceText,
            distance = distanceToMarker,
            angle = angleToMarker,
            searchState = searchState,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 32.dp)
        )

        // Progress indicator when searching
        if (searchState == SearchState.FOUND && !markerPlaced) {
            LockingProgress(
                progress = (emaConfidence / 0.55f).coerceIn(0f, 1f),  // 0.55 = lock threshold
                modifier = Modifier.align(Alignment.Center)
            )
        }
    }
}

enum class SearchState {
    SCANNING, FOUND, LOCKED, LOST, ARRIVED
}

@Composable
private fun StatusOverlay(
    searchState: SearchState,
    similarity: Float,
    consecutiveDetections: Int,
    trackingStatus: String,
    planeDetected: Boolean,
    markerPlaced: Boolean,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.padding(horizontal = 16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.7f)
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            // Status row
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusIndicator(
                    isActive = trackingStatus == "Tracking activ",
                    activeColor = Color.Green
                )
                Spacer(Modifier.width(8.dp))
                Text(trackingStatus, color = Color.White, fontSize = 12.sp)
            }

            Spacer(Modifier.height(4.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusIndicator(
                    isActive = planeDetected,
                    activeColor = Color.Green,
                    inactiveColor = Color.Yellow
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (planeDetected) "Suprafață detectată" else "Caut suprafețe...",
                    color = Color.White,
                    fontSize = 12.sp
                )
            }

            Spacer(Modifier.height(4.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusIndicator(
                    isActive = markerPlaced,
                    activeColor = Color.Green,
                    inactiveColor = when (searchState) {
                        SearchState.FOUND -> Color.Yellow
                        SearchState.LOST -> Color.Red
                        else -> Color.Gray
                    }
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    when {
                        markerPlaced -> "Marker plasat! ⚓"
                        searchState == SearchState.FOUND -> "Obiect detectat (${String.format("%.0f", similarity * 100)}%)"
                        searchState == SearchState.LOST -> "Obiect pierdut"
                        else -> "Caut obiectul..."
                    },
                    color = Color.White,
                    fontSize = 12.sp
                )
            }
        }
    }
}

/**
 * Direction arrow indicator (Seeing AI style)
 * Points toward the marker location
 */
@Composable
private fun DirectionArrow(
    angle: Float,
    isInFront: Boolean,
    distance: Float,
    modifier: Modifier = Modifier
) {
    val arrowColor = when {
        distance < 0.5f -> Color.Green
        distance < 1.5f -> Color(0xFF4CAF50)
        else -> Color.White
    }

    // Pulsing animation when close
    val infiniteTransition = rememberInfiniteTransition(label = "arrow_pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (distance < 1f) 1.2f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(500),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Box(
        modifier = modifier.size(80.dp),
        contentAlignment = Alignment.Center
    ) {
        // Background circle
        Box(
            modifier = Modifier
                .size(70.dp)
                .background(Color.Black.copy(alpha = 0.6f), CircleShape)
        )

        // Arrow icon that rotates
        Icon(
            imageVector = if (isInFront) Icons.Default.Navigation else Icons.Default.Undo,
            contentDescription = "Direcție",
            tint = arrowColor,
            modifier = Modifier
                .size((40 * scale).dp)
                .graphicsLayer {
                    // Rotate arrow to point toward marker
                    // angle: positive = right, negative = left
                    // Navigation icon points UP by default, so we rotate it
                    rotationZ = if (isInFront) angle else 180f
                }
        )
    }
}

@Composable
private fun StatusIndicator(
    isActive: Boolean,
    activeColor: Color = Color.Green,
    inactiveColor: Color = Color.Gray
) {
    Box(
        modifier = Modifier
            .size(10.dp)
            .background(
                if (isActive) activeColor else inactiveColor,
                CircleShape
            )
    )
}

@Composable
private fun GuidanceCard(
    text: String,
    distance: Float?,
    angle: Float?,
    searchState: SearchState,
    modifier: Modifier = Modifier
) {
    val backgroundColor = when (searchState) {
        SearchState.ARRIVED -> Color(0xFF4CAF50)
        SearchState.LOCKED -> Color(0xFF2196F3)
        SearchState.FOUND -> Color(0xFFFFA726)
        else -> Color.Black.copy(alpha = 0.7f)
    }

    Card(
        modifier = modifier.padding(horizontal = 16.dp),
        colors = CardDefaults.cardColors(containerColor = backgroundColor)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = text,
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )

            if (distance != null && searchState == SearchState.LOCKED) {
                Spacer(Modifier.height(8.dp))
                Row {
                    Text(
                        "📍 ${String.format("%.1f", distance)}m",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                    if (angle != null) {
                        Spacer(Modifier.width(16.dp))
                        Text(
                            "🧭 ${String.format("%.0f", angle)}°",
                            color = Color.White,
                            fontSize = 14.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun LockingProgress(
    progress: Float,
    modifier: Modifier = Modifier
) {
    val animatedProgress by animateFloatAsState(
        targetValue = progress.coerceIn(0f, 1f),
        animationSpec = tween(200),
        label = "progress"
    )

    Box(
        modifier = modifier.size(120.dp),
        contentAlignment = Alignment.Center
    ) {
        CircularProgressIndicator(
            progress = { animatedProgress },
            modifier = Modifier.size(100.dp),
            color = Color.Green,
            strokeWidth = 8.dp,
            trackColor = Color.White.copy(alpha = 0.3f)
        )

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "${(animatedProgress * 100).toInt()}%",
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                "Blocare...",
                color = Color.White,
                fontSize = 12.sp
            )
        }
    }
}

/**
 * Full object matcher using YOLO + CLIP pipeline.
 *
 * Pipeline:
 * 1. YOLO detects all objects in frame with bounding boxes
 * 2. For each detection, crop the region
 * 3. CLIP extracts features from crop
 * 4. Compare with saved embeddings
 * 5. Return best match similarity
 */
private class FullObjectMatcher(
    private val context: Context,
    private val objectName: String
) {
    private var clipExtractor: com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor? = null
    private var yoloSegmenter: com.florea_gabriel.impairedhelpapp.ml.segmentation.YoloSegmenter? = null
    private var targetEmbeddings: List<FloatArray> = emptyList()
    private var centroid: FloatArray = FloatArray(0) // Mean of all embeddings, re-normalized
    private var priorityLabels: Set<String> = emptySet()
    var matchThreshold: Float = 0.60f

    // Detection result with location info
    data class MatchResult(
        val similarity: Float,
        val label: String,
        val centerX: Float,  // Normalized 0-1
        val centerY: Float,  // Normalized 0-1
        val noUsableCandidates: Boolean = false // True when all detections were filtered out (e.g. only "person too large")
    )

    suspend fun initialize() = withContext(Dispatchers.IO) {
        // Initialize CLIP
        clipExtractor = com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor(context)

        // Initialize YOLO
        yoloSegmenter = com.florea_gabriel.impairedhelpapp.ml.segmentation.YoloSegmenter(context)
        if (yoloSegmenter?.isReady() == true) {
            Log.d(TAG, "✅ YOLO initialized")
        } else {
            Log.e(TAG, "❌ YOLO failed to initialize")
        }

        // Load saved embeddings for the object
        val db = com.florea_gabriel.impairedhelpapp.data.database.AppDatabase.getInstance(context)
        val personalObject = db.personalObjectDao().findByName(objectName.lowercase())
            ?: db.personalObjectDao().getAllObjectsList().find {
                it.name.lowercase().contains(objectName.lowercase())
            }

        if (personalObject != null) {
            // Adaptive threshold from registration quality
            matchThreshold = (personalObject.recommendedThreshold ?: 0.65f).coerceIn(0.65f, 0.80f)
            Log.d(TAG, "matchThreshold=$matchThreshold for $objectName (saved=${personalObject.recommendedThreshold})")

            targetEmbeddings = com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor.loadEmbeddings(
                personalObject.embeddingBlob,
                personalObject.embeddingJson
            )
            Log.d(TAG, "Loaded ${targetEmbeddings.size} embeddings for $objectName")

            // Compute centroid (mean of all embeddings, L2-normalized)
            if (targetEmbeddings.isNotEmpty()) {
                val dim = targetEmbeddings[0].size
                val mean = FloatArray(dim)
                for (emb in targetEmbeddings) {
                    for (i in emb.indices) mean[i] += emb[i]
                }
                val n = targetEmbeddings.size.toFloat()
                for (i in mean.indices) mean[i] /= n
                // L2 normalize
                var norm = 0f
                for (v in mean) norm += v * v
                norm = kotlin.math.sqrt(norm)
                if (norm > 0) for (i in mean.indices) mean[i] /= norm
                Log.d(TAG, "Initial centroid from $n embeddings")

                // === Outlier removal: keep only embeddings close to centroid ===
                // This tightens the cluster and makes the centroid more discriminative
                val centroidSims = targetEmbeddings.map { emb ->
                    var dot = 0f; var nA = 0f; var nB = 0f
                    for (j in emb.indices) { dot += emb[j] * mean[j]; nA += emb[j] * emb[j]; nB += mean[j] * mean[j] }
                    if (nA > 0 && nB > 0) dot / (kotlin.math.sqrt(nA) * kotlin.math.sqrt(nB)) else 0f
                }
                val avgSim = centroidSims.average().toFloat()
                Log.i(TAG, "📊 DIAG: emb→centroid: min=${String.format("%.3f", centroidSims.min())} " +
                    "max=${String.format("%.3f", centroidSims.max())} avg=${String.format("%.3f", avgSim)}")

                // Keep only embeddings with centroid similarity >= average (top ~50%)
                val keepThreshold = avgSim.coerceAtLeast(0.75f)
                val filteredEmbeddings = targetEmbeddings.filterIndexed { idx, _ ->
                    centroidSims[idx] >= keepThreshold
                }

                if (filteredEmbeddings.size >= 5) {
                    targetEmbeddings = filteredEmbeddings
                    // Recompute centroid from filtered embeddings
                    val cleanMean = FloatArray(filteredEmbeddings[0].size)
                    for (emb in filteredEmbeddings) for (j in emb.indices) cleanMean[j] += emb[j]
                    var cleanNorm = 0f
                    for (v in cleanMean) cleanNorm += v * v
                    cleanNorm = kotlin.math.sqrt(cleanNorm)
                    if (cleanNorm > 0) for (i in cleanMean.indices) cleanMean[i] /= cleanNorm
                    centroid = cleanMean
                    Log.i(TAG, "📊 Cleaned: ${filteredEmbeddings.size}/${n.toInt()} embeddings kept (threshold=${String.format("%.3f", keepThreshold)})")
                } else {
                    centroid = mean
                    Log.i(TAG, "📊 Kept all embeddings (filtered would leave <5)")
                }
            }

            // Load priority labels from saved object
            personalObject.detectionKeywords?.let { keywords ->
                priorityLabels = keywords.split(",")
                    .map { it.trim().lowercase() }
                    .filter { it.isNotEmpty() }
                    .toSet()
                Log.d(TAG, "Priority labels: $priorityLabels")
            }
        } else {
            Log.w(TAG, "No saved object found for: $objectName")
        }
    }

    suspend fun matchObject(bitmap: Bitmap): MatchResult = withContext(Dispatchers.Default) {
        val clip = clipExtractor ?: return@withContext MatchResult(0f, "", 0.5f, 0.5f)
        val yolo = yoloSegmenter ?: return@withContext MatchResult(0f, "", 0.5f, 0.5f)
        if (targetEmbeddings.isEmpty()) return@withContext MatchResult(0f, "", 0.5f, 0.5f)
        if (!yolo.isReady()) return@withContext MatchResult(0f, "", 0.5f, 0.5f)

        // Step 1: YOLO detection + segmentation in ONE pass
        val detectionsWithMasks = yolo.detectAllWithMasks(bitmap)

        val bitmapArea = (bitmap.width * bitmap.height).toFloat()

        // If YOLO found nothing — try CLIP-only fallback
        if (detectionsWithMasks.isEmpty()) {
            Log.d(TAG, "━━ YOLO: 0 detections (${bitmap.width}x${bitmap.height})")
            return@withContext clipFallback(clip, bitmap)
        }

        Log.d(TAG, "━━ YOLO-seg: ${detectionsWithMasks.size} detections")
        detectionsWithMasks.forEach { dwm ->
            val det = dwm.detection
            val label = yolo.getLabel(det.classId)
            val area = ((det.box[2] - det.box[0]) * (det.box[3] - det.box[1])) / bitmapArea
            val w = (det.box[2] - det.box[0]).toInt()
            val h = (det.box[3] - det.box[1]).toInt()
            Log.d(TAG, "   $label conf=${String.format("%.2f", det.score)} area=${String.format("%.4f", area)} ${w}x${h}px seg=${dwm.segmentedCrop != null}")
        }

        // Step 2: Filter and prioritize — use bbox crop as fallback when segmented crop is null
        data class CandidateWithCrop(
            val detection: com.florea_gabriel.impairedhelpapp.ml.segmentation.YoloSegmenter.Detection,
            val crop: Bitmap,
            val isSegmented: Boolean
        )

        val neverMatchLabels = setOf("bed", "couch", "sofa", "dining table", "toilet", "chair", "person")
        val candidatesWithCrops = mutableListOf<CandidateWithCrop>()

        for (dwm in detectionsWithMasks) {
            val det = dwm.detection
            val label = yolo.getLabel(det.classId).lowercase()
            val box = det.box
            val areaRatio = ((box[2] - box[0]) * (box[3] - box[1])) / bitmapArea

            if (label in neverMatchLabels) {
                Log.d(TAG, "   ❌ $label: filtered (neverMatch)")
                dwm.segmentedCrop?.recycle()
                continue
            }
            if (areaRatio < 0.002f) {
                Log.d(TAG, "   ❌ $label: too small ${String.format("%.4f", areaRatio)}")
                dwm.segmentedCrop?.recycle()
                continue
            }
            if (areaRatio > 0.5f) {
                Log.d(TAG, "   ❌ $label: too large ${String.format("%.4f", areaRatio)}")
                dwm.segmentedCrop?.recycle()
                continue
            }

            // Use segmented crop if available, otherwise fall back to bbox crop
            val crop: Bitmap
            val isSegmented: Boolean
            if (dwm.segmentedCrop != null) {
                crop = dwm.segmentedCrop
                isSegmented = true
                Log.d(TAG, "   ✅ $label: passed filter [SEGMENTED]")
            } else {
                // Bbox crop fallback
                val left = box[0].toInt().coerceIn(0, bitmap.width - 1)
                val top = box[1].toInt().coerceIn(0, bitmap.height - 1)
                val right = box[2].toInt().coerceIn(left + 1, bitmap.width)
                val bottom = box[3].toInt().coerceIn(top + 1, bitmap.height)
                val cw = right - left
                val ch = bottom - top
                if (cw < 10 || ch < 10) {
                    Log.d(TAG, "   ❌ $label: bbox crop too small ${cw}x${ch}")
                    continue
                }
                crop = Bitmap.createBitmap(bitmap, left, top, cw, ch)
                isSegmented = false
                Log.d(TAG, "   ✅ $label: passed filter [BBOX-CROP]")
            }

            candidatesWithCrops.add(CandidateWithCrop(det, crop, isSegmented))
        }

        // Sort by priority labels and score, take top 2
        val sortedCandidates = candidatesWithCrops
            .sortedByDescending { c ->
                val label = yolo.getLabel(c.detection.classId).lowercase()
                if (priorityLabels.contains(label)) 1000f + c.detection.score else c.detection.score
            }
            .take(2)

        // Recycle crops not in top candidates
        candidatesWithCrops.forEach { c ->
            if (c !in sortedCandidates) c.crop.recycle()
        }

        if (sortedCandidates.isEmpty()) {
            // All YOLO detections filtered (neverMatch/area) — try CLIP fallback
            detectionsWithMasks.forEach { it.segmentedCrop?.recycle() }
            return@withContext clipFallback(clip, bitmap)
        }

        Log.d(TAG, "   ${sortedCandidates.size} candidates for CLIP")

        // Step 3: Batch CLIP on candidate crops
        val cropBitmaps = sortedCandidates.map { it.crop }
        val startClip = System.currentTimeMillis()
        val embeddings = clip.extractFeaturesBatch(cropBitmaps)
        val clipMs = System.currentTimeMillis() - startClip
        Log.d(TAG, "   CLIP batch: ${cropBitmaps.size} crops in ${clipMs}ms")

        // Recycle all crops
        cropBitmaps.forEach { it.recycle() }

        if (embeddings == null || embeddings.size != sortedCandidates.size) {
            return@withContext MatchResult(0f, "", 0.5f, 0.5f)
        }

        // Step 4: Combined score matching
        var bestResult = MatchResult(0f, "", 0.5f, 0.5f)

        for (i in embeddings.indices) {
            val embedding = embeddings[i]
            val det = sortedCandidates[i].detection
            val label = yolo.getLabel(det.classId)
            val box = det.box
            val normCx = (box[0] + box[2]) / 2f / bitmap.width
            val normCy = (box[1] + box[3]) / 2f / bitmap.height
            val tag = if (sortedCandidates[i].isSegmented) "SEGMENTED" else "BBOX"

            val centroidSim = if (centroid.isNotEmpty()) cosineSimilarity(embedding, centroid) else 0f
            // Use top-3 embedding matches instead of just max for more robust scoring
            val topSims = targetEmbeddings.map { cosineSimilarity(embedding, it) }
                .sortedDescending().take(3)
            val top3Avg = topSims.average().toFloat()
            val rawScore = 0.5f * centroidSim + 0.5f * top3Avg

            // YOLO-label trust: priority labels get a boost, non-priority need higher raw score
            val labelLower = label.lowercase()
            val labelMatchesTarget = priorityLabels.contains(labelLower)
            val combinedScore: Float
            val effectiveThreshold: Float

            if (labelMatchesTarget) {
                // Trusted: YOLO label matches registration → boost 10%
                combinedScore = (rawScore * 1.10f).coerceAtMost(1f)
                effectiveThreshold = matchThreshold
            } else {
                // Untrusted: could be a false positive → require higher score
                combinedScore = rawScore
                effectiveThreshold = (matchThreshold + 0.10f).coerceAtMost(0.85f)
            }

            val passesThreshold = combinedScore >= effectiveThreshold
            Log.d(TAG, "  ${label}: centroid=${String.format("%.3f", centroidSim)} top3=${String.format("%.3f", top3Avg)} combined=${String.format("%.3f", combinedScore)} thresh=${String.format("%.2f", effectiveThreshold)}${if (labelMatchesTarget) " +PRIORITY" else ""}${if (passesThreshold) " ✓" else " ✗"} [$tag]")

            if (passesThreshold && combinedScore > bestResult.similarity) {
                bestResult = MatchResult(combinedScore, label, normCx, normCy)
            }
        }

        bestResult
    }

    /**
     * CLIP-only fallback: crops center of frame and runs CLIP directly.
     * Used when YOLO fails to detect the target object.
     * Uses top-3 maxSim (more robust than centroid alone) with higher threshold.
     */
    private suspend fun clipFallback(
        clip: com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor,
        bitmap: Bitmap
    ): MatchResult {
        val fallbackThreshold = (matchThreshold + 0.05f).coerceAtMost(0.80f)

        // Center crop ~50% of frame
        val cropW = (bitmap.width * 0.5f).toInt()
        val cropH = (bitmap.height * 0.5f).toInt()
        val cropX = (bitmap.width - cropW) / 2
        val cropY = (bitmap.height - cropH) / 2
        val centerCrop = Bitmap.createBitmap(bitmap, cropX, cropY, cropW, cropH)

        val emb = clip.extractFeatures(centerCrop)
        centerCrop.recycle()

        if (emb != null) {
            val centroidSim = if (centroid.isNotEmpty()) cosineSimilarity(emb, centroid) else 0f
            val topSims = targetEmbeddings.map { cosineSimilarity(emb, it) }
                .sortedDescending().take(3)
            val top3Avg = topSims.average().toFloat()
            val score = 0.5f * centroidSim + 0.5f * top3Avg

            Log.d(TAG, "   [CLIP-FALLBACK] centroid=${String.format("%.3f", centroidSim)} top3=${String.format("%.3f", top3Avg)} score=${String.format("%.3f", score)} thresh=$fallbackThreshold")

            if (score >= fallbackThreshold) {
                Log.d(TAG, "   [CLIP-FALLBACK] ✅ HIT")
                return MatchResult(score, "clip-fallback", 0.5f, 0.5f)
            } else {
                Log.d(TAG, "   [CLIP-FALLBACK] ❌ miss")
            }
        }

        return MatchResult(0f, "", 0.5f, 0.5f, noUsableCandidates = true)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float =
        com.florea_gabriel.impairedhelpapp.utils.cosineSimilarity(a, b)

    fun release() {
        clipExtractor?.close()
        yoloSegmenter?.close()
    }
}

/**
 * Convert ARCore Image (YUV_420_888) to Bitmap — direct per-pixel conversion.
 * Uses ITU-R BT.601 coefficients. Avoids JPEG encode/decode (~15-25ms savings).
 */
private fun imageToBitmap(image: Image): Bitmap? {
    return try {
        if (image.format != ImageFormat.YUV_420_888) {
            Log.w(TAG, "Unexpected image format: ${image.format}")
            return null
        }

        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val argb = IntArray(width * height)

        for (row in 0 until height) {
            val yRowOffset = row * yRowStride
            val uvRow = row shr 1
            val uvRowOffset = uvRow * uvRowStride

            for (col in 0 until width) {
                val y = (yBuffer.get(yRowOffset + col).toInt() and 0xFF)
                val uvCol = col shr 1
                val uvOffset = uvRowOffset + uvCol * uvPixelStride

                val u = (uBuffer.get(uvOffset).toInt() and 0xFF) - 128
                val v = (vBuffer.get(uvOffset).toInt() and 0xFF) - 128

                // ITU-R BT.601
                val r = (y + 1.370f * v).toInt().coerceIn(0, 255)
                val g = (y - 0.337f * u - 0.698f * v).toInt().coerceIn(0, 255)
                val b = (y + 1.732f * u).toInt().coerceIn(0, 255)

                argb[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    } catch (e: Exception) {
        Log.e(TAG, "Image to bitmap conversion failed: ${e.message}")
        null
    }
}

/**
 * Rotate a vector by the inverse of a quaternion.
 * This transforms a world-space vector into camera-local space.
 *
 * @param vector The vector to rotate [x, y, z]
 * @param quaternion The quaternion [x, y, z, w]
 * @return The rotated vector in camera-local space
 */
/**
 * Rotate a vector by a quaternion (camera-local → world space).
 * Used to project the camera's forward axis onto the world ground plane.
 *
 * @param vector The vector to rotate [x, y, z]
 * @param quaternion The quaternion [x, y, z, w]
 * @return The rotated vector in world space
 */
private fun rotateVectorByQuaternion(vector: FloatArray, quaternion: FloatArray): FloatArray {
    val qx = quaternion[0]
    val qy = quaternion[1]
    val qz = quaternion[2]
    val qw = quaternion[3]

    val vx = vector[0]
    val vy = vector[1]
    val vz = vector[2]

    // q * v * q^(-1), with t = 2 * cross(q.xyz, v)
    val tx = 2f * (qy * vz - qz * vy)
    val ty = 2f * (qz * vx - qx * vz)
    val tz = 2f * (qx * vy - qy * vx)

    return floatArrayOf(
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx)
    )
}
