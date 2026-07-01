package com.florea_gabriel.impairedhelpapp.ml.registration

import android.content.Context
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor
import com.florea_gabriel.impairedhelpapp.ml.search.GeminiObjectLabeler
import com.florea_gabriel.impairedhelpapp.ml.segmentation.YoloSegmenter
import kotlin.math.sqrt
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * AdvancedObjectRegistration: Seeing AI-style multi-angle object registration.
 *
 * PROBLEM with current approach:
 * - Single 6-second video may not capture all angles
 * - No guidance for user on what positions to capture
 * - No blur detection (blurry frames = bad embeddings)
 * - No IMU-based keyframe selection
 *
 * SOLUTION (Seeing AI approach):
 * - Guided capture: 3 multi-scale phases (close 30cm, medium 1m, far 2-3m)
 * - IMU keyframe selection: capture when camera rotates 20°+
 * - Blur detection: reject frames with Laplacian variance < threshold
 * - Voice guidance: tell user what to do
 * - Target: 390 diverse embeddings per object (130 per scale)
 *
 * @param context Android context
 * @param onVoiceGuidance Callback for TTS messages
 */
class AdvancedObjectRegistration(
        private val context: Context,
        private val onVoiceGuidance: (String) -> Unit
) : SensorEventListener {

    companion object {
        private const val TAG = "AdvancedRegistration"

        // Capture phases - MULTI-SCALE REGISTRATION
        const val PHASE_CLOSE = 0 // Close-up (30cm) - detailed features
        const val PHASE_MEDIUM = 1 // Medium distance (1m) - normal usage + rotation
        const val PHASE_FAR = 2 // Far distance (2-3m) - small object detection
        const val PHASE_COMPLETE = 3 // All done

        // Thresholds - optimized for < 60 sec total
        const val MIN_ROTATION_DEGREES = 15f // Minimum rotation to capture new keyframe
        const val BLUR_THRESHOLD = 80.0 // Laplacian variance threshold
        const val MIN_EMBEDDINGS_PER_PHASE = 10 // Minimum before allowing skip
        const val MAX_EMBEDDINGS_PER_PHASE = 25 // Safety limit
        const val TARGET_TOTAL_EMBEDDINGS = 50 // 15 + 20 + 15

        // Phase durations (ms) - reduced for faster capture
        const val PHASE_DURATION_MS = 8000L // 8s per phase max
    }

    /** Registration state */
    enum class RegistrationState {
        IDLE, // Not started
        INITIALIZING, // Loading models
        READY, // Ready to start
        CAPTURING, // Actively capturing
        PROCESSING, // Processing embeddings
        COMPLETE, // Successfully completed
        ERROR // Error occurred
    }

    /** Capture phase info */
    data class PhaseInfo(
            val phase: Int,
            val name: String,
            val instruction: String,
            val targetEmbeddings: Int
    )

    /** Registration progress */
    data class RegistrationProgress(
            val state: RegistrationState,
            val currentPhase: Int,
            val phaseProgress: Float, // 0-1 within current phase
            val totalProgress: Float, // 0-1 overall
            val embeddingsCollected: Int,
            val currentInstruction: String,
            val errorMessage: String? = null
    )

    // ML Components
    private var clipExtractor: CLIPFeatureExtractor? = null
    private var yoloSegmenter: YoloSegmenter? = null
    private var geminiLabeler: GeminiObjectLabeler? = null

    // YOLO labels detected during registration (for offline Gemini fallback)
    private val detectedYoloLabels = mutableSetOf<String>()

    // Best frame for labeling
    private var bestFrame: Bitmap? = null
    private var bestFrameScore = 0.0

    // Sensor components
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    // Rotation tracking for keyframe selection
    private var lastKeyframeRotation = FloatArray(3)
    private var currentRotation = FloatArray(3)
    private val rotationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)
    private var accelerometerValues = FloatArray(3)
    private var magnetometerValues = FloatArray(3)
    private var lastKeyframeTime = 0L
    private var lastGyroTimestampNs = 0L

    // State
    private var _progress =
            MutableStateFlow(
                    RegistrationProgress(
                            state = RegistrationState.IDLE,
                            currentPhase = -1,
                            phaseProgress = 0f,
                            totalProgress = 0f,
                            embeddingsCollected = 0,
                            currentInstruction = ""
                    )
            )
    val progress: StateFlow<RegistrationProgress> = _progress.asStateFlow()

    // Debug mask overlay for visualization
    private val _debugOverlay = MutableStateFlow<Bitmap?>(null)
    val debugOverlay: StateFlow<Bitmap?> = _debugOverlay.asStateFlow()
    var debugMaskEnabled = false

    // Collected data
    private val phaseEmbeddings = mutableMapOf<Int, MutableList<FloatArray>>()
    private val allEmbeddings = mutableListOf<FloatArray>()

    // Coroutine scope
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // 3-PHASE GUIDED REGISTRATION for diverse embeddings
    // Phase 1: Close-up - detailed object features
    // Phase 2: Medium distance - typical viewing distance
    // Phase 3: Different background - context diversity
    private val phases =
            listOf(
                    PhaseInfo(
                            PHASE_CLOSE,
                            "Aproape",
                            "Ține obiectul la 30 cm. Rotește-l încet 360 de grade.",
                            15
                    ),
                    PhaseInfo(
                            PHASE_MEDIUM,
                            "Distanță medie",
                            "Pune obiectul jos. Îndepărtează-te la 1 metru și privește-l.",
                            20
                    ),
                    PhaseInfo(
                            PHASE_FAR,
                            "Alt fundal",
                            "Mută obiectul în alt loc din cameră. Privește-l de la distanță.",
                            15
                    )
            )

    /** Initialize ML models */
    suspend fun initialize() =
            withContext(Dispatchers.IO) {
                updateState(RegistrationState.INITIALIZING, instruction = "Se încarcă modelele...")

                try {
                    Log.d(TAG, "Initializing CLIP extractor...")
                    clipExtractor = CLIPFeatureExtractor(context)

                    Log.d(TAG, "Initializing YoloSegmenter Segmenter...")
                    yoloSegmenter = YoloSegmenter(context)
                    if (!yoloSegmenter!!.isReady()) {
                        Log.e(TAG, "❌ YoloSegmenter failed to initialize!")
                        updateState(
                                RegistrationState.ERROR,
                                error = "Failed to initialize segmenter"
                        )
                        return@withContext
                    }

                    Log.d(TAG, "Initializing GeminiLabeler...")
                    geminiLabeler = GeminiObjectLabeler()

                    // Initialize sensors
                    accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
                    gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

                    updateState(RegistrationState.READY, instruction = "Gata pentru înregistrare")
                    Log.d(TAG, "Registration system initialized")
                } catch (e: Exception) {
                    Log.e(TAG, "Initialization failed: ${e.message}", e)
                    updateState(
                            RegistrationState.ERROR,
                            error = "Eroare la inițializare: ${e.message}"
                    )
                }
            }

    /** Start sensors for keyframe detection */
    fun startSensors() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    /** Stop sensors */
    fun stopSensors() {
        sensorManager.unregisterListener(this)
    }

    /** Start registration process */
    fun startRegistration() {
        if (_progress.value.state != RegistrationState.READY) {
            Log.w(TAG, "Cannot start - not in READY state")
            return
        }

        // Clear previous data
        phaseEmbeddings.clear()
        allEmbeddings.clear()
        phases.forEach { phaseEmbeddings[it.phase] = mutableListOf() }

        // Start with first phase (close-up)
        startPhase(PHASE_CLOSE)
    }

    /** Start a specific capture phase */
    fun startPhase(phase: Int) {
        val phaseInfo = phases.getOrNull(phase)
        if (phaseInfo == null) {
            completeRegistration()
            return
        }

        Log.d(TAG, "Starting phase: ${phaseInfo.name}")

        updateState(
                state = RegistrationState.CAPTURING,
                phase = phase,
                phaseProgress = 0f,
                instruction = phaseInfo.instruction
        )

        // Voice guidance
        onVoiceGuidance("Faza ${phase + 1}: ${phaseInfo.instruction}")

        // Reset keyframe tracking
        lastKeyframeRotation = currentRotation.clone()
        lastKeyframeTime = System.currentTimeMillis()
        lastGyroTimestampNs = 0L
    }

    /**
     * Process a camera frame during capture. Uses IMU + blur detection to decide if frame is a good
     * keyframe.
     *
     * @param bitmap Camera frame
     * @param depthMap Optional depth map for background removal (PREFERRED over SAM)
     * @return true if frame was accepted as keyframe
     */
    suspend fun processFrame(bitmap: Bitmap, depthMap: FloatArray? = null): Boolean =
            withContext(Dispatchers.Default) {
                if (_progress.value.state != RegistrationState.CAPTURING) {
                    return@withContext false
                }

                val currentPhase = _progress.value.currentPhase
                val phaseInfo = phases.getOrNull(currentPhase) ?: return@withContext false
                val phaseList = phaseEmbeddings[currentPhase] ?: return@withContext false

                // Check if we've reached target for this phase - auto-advance
                if (phaseList.size >= phaseInfo.targetEmbeddings) {
                    Log.d(
                            TAG,
                            "Phase $currentPhase reached target ${phaseInfo.targetEmbeddings}, moving to next"
                    )
                    advancePhase()
                    return@withContext false
                }

                // Safety limit: don't exceed max per phase
                if (phaseList.size >= MAX_EMBEDDINGS_PER_PHASE) {
                    Log.d(TAG, "Phase $currentPhase at max limit, forcing advance")
                    advancePhase()
                    return@withContext false
                }

                // Check 1: Is this a keyframe? (enough rotation since last capture)
                val isKeyframe = shouldCaptureKeyframe()

                // Check 2: Is frame sharp enough?
                val isSharp = isFrameSharp(bitmap)

                if (!isKeyframe && phaseList.size >= MIN_EMBEDDINGS_PER_PHASE) {
                    // We have minimum, wait for rotation
                    return@withContext false
                }

                if (!isSharp) {
                    Log.d(TAG, "Frame rejected: blur detected")
                    return@withContext false
                }

                // Process frame
                try {
                    val embeddings = extractEmbeddings(bitmap, depthMap)

                    if (embeddings.isNotEmpty()) {
                        phaseList.addAll(embeddings)
                        allEmbeddings.addAll(embeddings)

                        // Update progress
                        val phaseProgress = phaseList.size.toFloat() / phaseInfo.targetEmbeddings
                        val totalProgress = allEmbeddings.size.toFloat() / TARGET_TOTAL_EMBEDDINGS

                        updateState(
                                state = RegistrationState.CAPTURING,
                                phase = currentPhase,
                                phaseProgress = phaseProgress.coerceIn(0f, 1f),
                                totalProgress = totalProgress.coerceIn(0f, 1f),
                                embeddingsCount = allEmbeddings.size,
                                instruction = phaseInfo.instruction
                        )

                        // Capture best frame for Gemini labeling (using blur score as proxy for
                        // sharpness)
                        // We want a clear, sharp image for the LLM
                        val sharpness = calculateSharpness(bitmap)
                        if (bestFrame == null || sharpness > bestFrameScore) {
                            bestFrame?.recycle()
                            bestFrame = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                            bestFrameScore = sharpness
                        }

                        // Mark this as keyframe position
                        lastKeyframeRotation = currentRotation.clone()
                        lastKeyframeTime = System.currentTimeMillis()

                        Log.d(
                                TAG,
                                "Frame accepted: +${embeddings.size} embeddings " +
                                        "(phase: ${phaseList.size}, total: ${allEmbeddings.size})"
                        )

                        return@withContext true
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Frame processing error: ${e.message}")
                }

                return@withContext false
            }

    /** Advance to next phase - just updates phase, does NOT auto-start capture */
    fun advancePhase() {
        val nextPhase = _progress.value.currentPhase + 1
        if (nextPhase >= phases.size) {
            completeRegistration()
        } else {
            // Only update the phase number, let UI show instruction and trigger startPhase
            val phaseInfo = phases[nextPhase]
            Log.d(TAG, "Phase ${nextPhase - 1} complete, pending phase: ${phaseInfo.name}")

            updateState(
                    state = RegistrationState.READY, // READY, not CAPTURING - wait for user
                    phase = nextPhase,
                    phaseProgress = 0f,
                    instruction = "Pregătește-te pentru: ${phaseInfo.name}"
            )
        }
    }

    /** Complete registration and return all embeddings */
    private fun completeRegistration() {
        if (allEmbeddings.size < MIN_EMBEDDINGS_PER_PHASE) {
            updateState(
                    state = RegistrationState.ERROR,
                    error = "Prea puține embeddings: ${allEmbeddings.size}"
            )
            return
        }

        Log.d(TAG, "Registration complete: ${allEmbeddings.size} total embeddings")

        // Log phase breakdown
        phases.forEach { phase ->
            val count = phaseEmbeddings[phase.phase]?.size ?: 0
            Log.d(TAG, "  Phase ${phase.name}: $count embeddings")
        }

        updateState(
                state = RegistrationState.COMPLETE,
                phase = PHASE_COMPLETE,
                totalProgress = 1f,
                embeddingsCount = allEmbeddings.size,
                instruction = "Înregistrare completă! ${allEmbeddings.size} caracteristici."
        )

        onVoiceGuidance("Înregistrare completă cu ${allEmbeddings.size} caracteristici")
    }

    /** Get all collected embeddings */
    fun getAllEmbeddings(): List<FloatArray> = allEmbeddings.toList()

    /**
     * Extract embeddings from a frame using depth masking (preferred) or SAM + CLIP at multiple
     * scales.
     *
     * @param bitmap Original frame
     * @param depthMap Optional depth map for fast background removal
     */
    /** Get the best frame captured so far (for thumbnail/labeling) */
    fun getBestFrame(): Bitmap? = bestFrame?.copy(Bitmap.Config.ARGB_8888, true)

    /** Get YOLO labels detected during registration (for offline Gemini fallback) */
    fun getDetectedYoloLabels(): Set<String> = detectedYoloLabels.toSet()

    /** Generate semantic labels using Gemini */
    suspend fun generateSemanticLabels(bitmap: Bitmap): GeminiObjectLabeler.LabelingResult? {
        return geminiLabeler?.labelObject(bitmap)?.getOrNull()
    }

    /**
     * Extract embeddings from a frame using YOLO segmentation (preferred).
     *
     * @param bitmap Original frame
     * @param depthMap Optional depth map (Ignored, using YOLO-seg now)
     */
    private suspend fun extractEmbeddings(
            bitmap: Bitmap,
            depthMap: FloatArray? = null
    ): List<FloatArray> {
        val clip = clipExtractor ?: return emptyList()
        val yolo = yoloSegmenter

        val embeddings = mutableListOf<FloatArray>()

        try {
            // Step 1: Segment object using YOLO-seg (class-aware)
            val processedBitmap: Bitmap

            if (yolo != null && yolo.isReady()) {
                if (debugMaskEnabled) {
                    // Debug mode: get both segmented and overlay
                    val (segmented, overlay) = yolo.segmentWithOverlay(bitmap)
                    processedBitmap = segmented
                    _debugOverlay.value = overlay
                    if (segmented != bitmap) {
                        Log.v(TAG, "✅ YoloSegmenter segmentation applied (debug mode)")
                    }
                } else {
                    // Normal mode
                    val segmented = yolo.segment(bitmap)
                    processedBitmap = segmented
                    if (segmented != bitmap) {
                        Log.v(TAG, "✅ YoloSegmenter segmentation applied")
                    } else {
                        Log.w(TAG, "⚠️ YoloSegmenter returned original - no object detected")
                    }
                }
                // Track detected YOLO label for offline keyword fallback
                yolo.lastDetectedLabel?.let { label ->
                    if (label != "unknown") detectedYoloLabels.add(label)
                }
            } else {
                Log.w(TAG, "⚠️ YoloSegmenter NOT READY")
                processedBitmap = bitmap
            }

            // Step 2: Extract single embedding from processed bitmap
            // Multi-scale is now achieved through physical distance (3 capture phases)

            // Save debug samples only when debug mode is explicitly enabled.
            if (debugMaskEnabled) {
                debugFrameCounter++
                if (debugFrameCounter % 15 == 0) { // Save every ~15th frame (approx 1 per second)
                    saveDebugImage(
                            processedBitmap,
                            "phase_${_progress.value.currentPhase}_frame_$debugFrameCounter"
                    )
                }
            }

            val embedding = clip.extractFeatures(processedBitmap)

            if (embedding != null) {
                embeddings.add(embedding)
                // Log embedding info for verification
                val norm = kotlin.math.sqrt(embedding.map { it * it }.sum())
                Log.d(
                        TAG,
                        "✅ CLIP Embedding: dim=${embedding.size}, norm=${"%.4f".format(norm)}, sample=[${embedding.take(3).map { "%.4f".format(it) }.joinToString(", ")}...]"
                )
            } else {
                Log.e(TAG, "❌ Failed to extract embedding from frame")
            }

            // Clean up
            if (processedBitmap != bitmap && !processedBitmap.isRecycled) {
                processedBitmap.recycle()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Embedding extraction error: ${e.message}")
        }

        return embeddings
    }

    private fun calculateSharpness(bitmap: Bitmap): Double {
        // Reuse isFrameSharp logic but return score
        val width = bitmap.width
        val height = bitmap.height
        val sampleSize = minOf(100, width / 4, height / 4)
        val startX = (width - sampleSize) / 2
        val startY = (height - sampleSize) / 2

        var sum = 0.0
        var sumSquared = 0.0
        var count = 0

        val pixels = IntArray(sampleSize * sampleSize)
        bitmap.getPixels(pixels, 0, sampleSize, startX, startY, sampleSize, sampleSize)

        for (y in 1 until sampleSize - 1) {
            for (x in 1 until sampleSize - 1) {
                val center = getGray(pixels[y * sampleSize + x])
                val up = getGray(pixels[(y - 1) * sampleSize + x])
                val down = getGray(pixels[(y + 1) * sampleSize + x])
                val left = getGray(pixels[y * sampleSize + (x - 1)])
                val right = getGray(pixels[y * sampleSize + (x + 1)])

                val laplacian = (4 * center - up - down - left - right).toDouble()
                sum += laplacian
                sumSquared += laplacian * laplacian
                count++
            }
        }

        if (count == 0) return 0.0
        val mean = sum / count
        return (sumSquared / count) - (mean * mean)
    }

    // DEBUG: Counter for saved frames
    private var debugFrameCounter = 0

    /** DEBUG: Save processed bitmap to storage to visually verify background removal */
    private fun saveDebugImage(bitmap: Bitmap, name: String) {
        scope.launch(Dispatchers.IO) {
            try {
                // Save to:
                // /Android/data/com.florea_gabriel.impairedhelpapp/files/Pictures/Debug_Registration
                val dir =
                        java.io.File(
                                context.getExternalFilesDir(
                                        android.os.Environment.DIRECTORY_PICTURES
                                ),
                                "Debug_Registration"
                        )
                if (!dir.exists()) dir.mkdirs()

                val file = java.io.File(dir, "debug_${System.currentTimeMillis()}_$name.jpg")
                val stream = java.io.FileOutputStream(file)

                // Save as JPEG with high quality
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
                stream.flush()
                stream.close()

                Log.d(TAG, "📸 DEBUG IMAGE SAVED: ${file.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save debug image: ${e.message}")
            }
        }
    }

    /** Check if camera has rotated enough to capture a new keyframe */
    private fun shouldCaptureKeyframe(): Boolean {
        val timeSinceLastKeyframe = System.currentTimeMillis() - lastKeyframeTime

        // Always capture if enough time passed (even without rotation)
        if (timeSinceLastKeyframe > 1500L) {
            return true
        }

        // Check rotation difference
        val rotationDiff = calculateRotationDifference(currentRotation, lastKeyframeRotation)

        return rotationDiff >= MIN_ROTATION_DEGREES
    }

    /** Calculate rotation difference in degrees */
    private fun calculateRotationDifference(rot1: FloatArray, rot2: FloatArray): Float {
        if (rot1.size < 3 || rot2.size < 3) return 0f

        // Calculate difference for each axis (azimuth is most important)
        var diffAzimuth = Math.toDegrees((rot1[0] - rot2[0]).toDouble()).toFloat()
        if (diffAzimuth > 180) diffAzimuth -= 360
        if (diffAzimuth < -180) diffAzimuth += 360

        val diffPitch = Math.toDegrees((rot1[1] - rot2[1]).toDouble()).toFloat()
        val diffRoll = Math.toDegrees((rot1[2] - rot2[2]).toDouble()).toFloat()

        // Combined rotation magnitude
        return sqrt(
                diffAzimuth * diffAzimuth +
                        diffPitch * diffPitch * 0.5f +
                        diffRoll * diffRoll * 0.3f
        )
    }

    /** Check if frame is sharp (not blurry) */
    private fun isFrameSharp(bitmap: Bitmap): Boolean {
        // Simple Laplacian variance blur detection
        val width = bitmap.width
        val height = bitmap.height

        // Sample center region
        val sampleSize = minOf(100, width / 4, height / 4)
        val startX = (width - sampleSize) / 2
        val startY = (height - sampleSize) / 2

        var sum = 0.0
        var sumSquared = 0.0
        var count = 0

        val pixels = IntArray(sampleSize * sampleSize)
        bitmap.getPixels(pixels, 0, sampleSize, startX, startY, sampleSize, sampleSize)

        // Calculate Laplacian approximation
        for (y in 1 until sampleSize - 1) {
            for (x in 1 until sampleSize - 1) {
                val center = getGray(pixels[y * sampleSize + x])
                val up = getGray(pixels[(y - 1) * sampleSize + x])
                val down = getGray(pixels[(y + 1) * sampleSize + x])
                val left = getGray(pixels[y * sampleSize + (x - 1)])
                val right = getGray(pixels[y * sampleSize + (x + 1)])

                // Laplacian = 4 * center - up - down - left - right
                val laplacian = (4 * center - up - down - left - right).toDouble()

                sum += laplacian
                sumSquared += laplacian * laplacian
                count++
            }
        }

        if (count == 0) return true

        // Variance = E[X²] - E[X]²
        val mean = sum / count
        val variance = (sumSquared / count) - (mean * mean)

        return variance > BLUR_THRESHOLD
    }

    /** Get grayscale value from pixel */
    private fun getGray(pixel: Int): Int {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return (r + g + b) / 3
    }

    /** Update state helper */
    private fun updateState(
            state: RegistrationState,
            phase: Int = _progress.value.currentPhase,
            phaseProgress: Float = _progress.value.phaseProgress,
            totalProgress: Float = _progress.value.totalProgress,
            embeddingsCount: Int = _progress.value.embeddingsCollected,
            instruction: String = _progress.value.currentInstruction,
            error: String? = null
    ) {
        _progress.value =
                RegistrationProgress(
                        state = state,
                        currentPhase = phase,
                        phaseProgress = phaseProgress,
                        totalProgress = totalProgress,
                        embeddingsCollected = embeddingsCount,
                        currentInstruction = instruction,
                        errorMessage = error
                )
    }

    // SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                System.arraycopy(event.values, 0, accelerometerValues, 0, 3)
                updateOrientation()
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                System.arraycopy(event.values, 0, magnetometerValues, 0, 3)
                updateOrientation()
            }
            Sensor.TYPE_GYROSCOPE -> {
                // Use sensor timestamps only; do not mix with wall-clock milliseconds.
                val prevTs = lastGyroTimestampNs
                if (prevTs != 0L) {
                    val deltaTime = (event.timestamp - prevTs) / 1_000_000_000f
                    if (deltaTime > 0f && deltaTime < 0.1f) {
                        currentRotation[0] += event.values[0] * deltaTime
                        currentRotation[1] += event.values[1] * deltaTime
                        currentRotation[2] += event.values[2] * deltaTime
                    }
                }
                lastGyroTimestampNs = event.timestamp
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed
    }

    /** Update orientation from accelerometer + magnetometer */
    private fun updateOrientation() {
        if (SensorManager.getRotationMatrix(
                        rotationMatrix,
                        null,
                        accelerometerValues,
                        magnetometerValues
                )
        ) {
            SensorManager.getOrientation(rotationMatrix, orientationAngles)
            System.arraycopy(orientationAngles, 0, currentRotation, 0, 3)
        }
    }

    /** Clean up resources */
    fun release() {
        stopSensors()
        scope.cancel()
        clipExtractor?.close()
        yoloSegmenter?.close()
        yoloSegmenter = null
        geminiLabeler?.release()
        bestFrame?.recycle()
        phaseEmbeddings.clear()
        allEmbeddings.clear()
    }
}
