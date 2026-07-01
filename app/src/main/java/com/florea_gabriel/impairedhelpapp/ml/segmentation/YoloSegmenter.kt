package com.florea_gabriel.impairedhelpapp.ml.segmentation

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * YoloSegmenter: Fast instance segmentation using YOLO11-seg (ONNX).
 *
 * Replaces MobileSAM for real-time background removal.
 *
 * Model requirements:
 * - File: yolo11n-seg.onnx (or yolov8n-seg.onnx)
 * - Input: 1x3x640x640 (normalized 0-1)
 * - Output 0: 1x116x8400 (boxes + classes + mask_coeffs)
 * - Output 1: 1x32x160x160 (proto_masks)
 */
class YoloSegmenter(private val context: Context) {

    companion object {
        private const val TAG = "YoloSegmenter"
        private const val MODEL_FILE = "yolo11n-seg.onnx"
        private const val INPUT_SIZE = 640
        private const val MASK_SIZE = 160
        private const val NUM_CLASSES = 80
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.5f

        // COCO 80 class names (official order)
        private val COCO_CLASSES =
                arrayOf(
                        "person",
                        "bicycle",
                        "car",
                        "motorcycle",
                        "airplane",
                        "bus",
                        "train",
                        "truck",
                        "boat",
                        "traffic light",
                        "fire hydrant",
                        "stop sign",
                        "parking meter",
                        "bench",
                        "bird",
                        "cat",
                        "dog",
                        "horse",
                        "sheep",
                        "cow",
                        "elephant",
                        "bear",
                        "zebra",
                        "giraffe",
                        "backpack",
                        "umbrella",
                        "handbag",
                        "tie",
                        "suitcase",
                        "frisbee",
                        "skis",
                        "snowboard",
                        "sports ball",
                        "kite",
                        "baseball bat",
                        "baseball glove",
                        "skateboard",
                        "surfboard",
                        "tennis racket",
                        "bottle",
                        "wine glass",
                        "cup",
                        "fork",
                        "knife",
                        "spoon",
                        "bowl",
                        "banana",
                        "apple",
                        "sandwich",
                        "orange",
                        "broccoli",
                        "carrot",
                        "hot dog",
                        "pizza",
                        "donut",
                        "cake",
                        "chair",
                        "couch",
                        "potted plant",
                        "bed",
                        "dining table",
                        "toilet",
                        "tv",
                        "laptop",
                        "mouse",
                        "remote",
                        "keyboard",
                        "cell phone",
                        "microwave",
                        "oven",
                        "toaster",
                        "sink",
                        "refrigerator",
                        "book",
                        "clock",
                        "vase",
                        "scissors",
                        "teddy bear",
                        "hair drier",
                        "toothbrush"
                )
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    /** Last detected COCO class label from segment(). Used for offline keyword fallback. */
    var lastDetectedLabel: String? = null
        private set

    init {
        initModel()
    }

    private fun initModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            sessionOptions.setIntraOpNumThreads(4)
            sessionOptions.setInterOpNumThreads(2)
            sessionOptions.addConfigEntry("session.load_model_format", "ONNX")

            // Check if model exists
            val assetManager = context.assets
            val modelList = assetManager.list("") ?: emptyArray()
            if (!modelList.contains(MODEL_FILE)) {
                Log.w(TAG, "⚠️ Model file $MODEL_FILE not found in assets. Segmentation will fail.")
                return
            }

            // Read model bytes
            val modelBytes = assetManager.open(MODEL_FILE).readBytes()
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)

            Log.d(TAG, "✅ YOLO Segmenter initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init YOLO Segmenter: ${e.message}")
        }
    }

    fun isReady(): Boolean = ortSession != null

    /** Get class label for detection ID */
    fun getLabel(classId: Int): String =
            if (classId in COCO_CLASSES.indices) COCO_CLASSES[classId] else "unknown"

    /**
     * Detect all objects in the bitmap. Returns list of Detection with bbox, class, score. Use this
     * for search when you need to check multiple objects.
     */
    suspend fun detectAll(bitmap: Bitmap): List<Detection> =
            withContext(Dispatchers.Default) {
                if (!isReady()) return@withContext emptyList()
                val session = ortSession ?: return@withContext emptyList()

                try {
                    val (resizedBitmap, scale, padX, padY) = preprocess(bitmap)
                    val floatBuffer = bitmapToFloatBuffer(resizedBitmap)
                    val inputName = session.inputNames.iterator().next()
                    val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                    val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
                    val results = session.run(Collections.singletonMap(inputName, inputTensor))

                    val output0 = results[0].value as Array<Array<FloatArray>>
                    val detections = output0[0]
                    val candidates = ArrayList<Detection>()
                    val numAnchors = detections[0].size

                    for (i in 0 until numAnchors) {
                        var maxScore = 0f
                        var classId = -1
                        for (c in 0 until NUM_CLASSES) {
                            val score = detections[4 + c][i]
                            if (score > maxScore) {
                                maxScore = score
                                classId = c
                            }
                        }
                        if (maxScore > CONFIDENCE_THRESHOLD) {
                            val cx = detections[0][i]
                            val cy = detections[1][i]
                            val w = detections[2][i]
                            val h = detections[3][i]
                            // Convert from 640x640 coords to original bitmap coords
                            val left = ((cx - w / 2) - padX) / scale
                            val top = ((cy - h / 2) - padY) / scale
                            val right = ((cx + w / 2) - padX) / scale
                            val bottom = ((cy + h / 2) - padY) / scale

                            val maskCoeffs = FloatArray(32)
                            for (m in 0 until 32) maskCoeffs[m] = detections[4 + NUM_CLASSES + m][i]

                            candidates.add(
                                    Detection(
                                            classId,
                                            maxScore,
                                            floatArrayOf(
                                                    left.coerceAtLeast(0f),
                                                    top.coerceAtLeast(0f),
                                                    right.coerceAtMost(bitmap.width.toFloat()),
                                                    bottom.coerceAtMost(bitmap.height.toFloat())
                                            ),
                                            maskCoeffs
                                    )
                            )
                        }
                    }
                    inputTensor.close()
                    return@withContext nms(candidates)
                } catch (e: Exception) {
                    Log.e(TAG, "detectAll failed: ${e.message}")
                    return@withContext emptyList()
                }
            }

    /**
     * Detect all objects AND generate per-detection segmentation masks in a single YOLO pass.
     * Returns list of DetectionWithMask(detection, segmentedCrop).
     * The segmented crop has transparent background — same domain as registration.
     */
    suspend fun detectAllWithMasks(bitmap: Bitmap): List<DetectionWithMask> =
            withContext(Dispatchers.Default) {
                if (!isReady()) return@withContext emptyList()
                val session = ortSession ?: return@withContext emptyList()

                try {
                    val (resizedBitmap, scale, padX, padY) = preprocess(bitmap)
                    val floatBuffer = bitmapToFloatBuffer(resizedBitmap)
                    val inputName = session.inputNames.iterator().next()
                    val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                    val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
                    val results = session.run(Collections.singletonMap(inputName, inputTensor))

                    val output0 = results[0].value as Array<Array<FloatArray>>
                    val detections = output0[0]
                    val protos = (results[1].value as Array<Array<Array<FloatArray>>>)[0] // [32][160][160]
                    val candidates = ArrayList<Detection>()
                    val numAnchors = detections[0].size

                    for (i in 0 until numAnchors) {
                        var maxScore = 0f
                        var classId = -1
                        for (c in 0 until NUM_CLASSES) {
                            val score = detections[4 + c][i]
                            if (score > maxScore) {
                                maxScore = score
                                classId = c
                            }
                        }
                        if (maxScore > CONFIDENCE_THRESHOLD) {
                            val cx = detections[0][i]
                            val cy = detections[1][i]
                            val w = detections[2][i]
                            val h = detections[3][i]
                            val left = ((cx - w / 2) - padX) / scale
                            val top = ((cy - h / 2) - padY) / scale
                            val right = ((cx + w / 2) - padX) / scale
                            val bottom = ((cy + h / 2) - padY) / scale

                            val maskCoeffs = FloatArray(32)
                            for (m in 0 until 32) maskCoeffs[m] = detections[4 + NUM_CLASSES + m][i]

                            candidates.add(
                                    Detection(
                                            classId,
                                            maxScore,
                                            floatArrayOf(
                                                    left.coerceAtLeast(0f),
                                                    top.coerceAtLeast(0f),
                                                    right.coerceAtMost(bitmap.width.toFloat()),
                                                    bottom.coerceAtMost(bitmap.height.toFloat())
                                            ),
                                            maskCoeffs
                                    )
                            )
                        }
                    }
                    inputTensor.close()
                    val nmsDetections = nms(candidates)

                    // Generate segmented crop for each detection
                    // Memory-efficient: compute mask ONLY on bbox region, not full frame
                    nmsDetections.map { det ->
                        val box = det.box
                        val bw = box[2] - box[0]
                        val bh = box[3] - box[1]
                        val padXCrop = (bw * 0.10f).toInt()
                        val padYCrop = (bh * 0.10f).toInt()
                        val cropLeft = (box[0].toInt() - padXCrop).coerceIn(0, bitmap.width - 1)
                        val cropTop = (box[1].toInt() - padYCrop).coerceIn(0, bitmap.height - 1)
                        val cropRight = (box[2].toInt() + padXCrop).coerceIn(cropLeft + 1, bitmap.width)
                        val cropBottom = (box[3].toInt() + padYCrop).coerceIn(cropTop + 1, bitmap.height)
                        val cw = cropRight - cropLeft
                        val ch = cropBottom - cropTop

                        val croppedSegmented = if (cw > 10 && ch > 10) {
                            // Crop original pixels from frame (small bitmap, e.g. 350x150)
                            val crop = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cw, ch)
                            val cropPixels = IntArray(cw * ch)
                            crop.getPixels(cropPixels, 0, cw, 0, 0, cw, ch)

                            // Compute mask ONLY for crop pixels (not full 1920x1080!)
                            for (y in 0 until ch) {
                                for (x in 0 until cw) {
                                    // Map crop pixel (x,y) back to original frame coords
                                    val origX = cropLeft + x
                                    val origY = cropTop + y

                                    // Map original coords to YOLO input space (640x640)
                                    val inputX = origX * scale + padX
                                    val inputY = origY * scale + padY

                                    // Map to mask space (160x160)
                                    val maskX = (inputX * 0.25f).toInt().coerceIn(0, MASK_SIZE - 1)
                                    val maskY = (inputY * 0.25f).toInt().coerceIn(0, MASK_SIZE - 1)

                                    // Compute mask value: maskCoeffs dot protos at this position
                                    var sum = 0f
                                    for (c in 0 until 32) {
                                        sum += det.maskCoeffs[c] * protos[c][maskY][maskX]
                                    }
                                    val maskVal = 1.0f / (1.0f + kotlin.math.exp(-sum))

                                    // Apply: keep pixel if mask > 0.5, transparent otherwise
                                    if (maskVal <= 0.5f) {
                                        cropPixels[y * cw + x] = 0 // transparent
                                    }
                                }
                            }

                            // Write masked pixels back
                            val result = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
                            result.setPixels(cropPixels, 0, cw, 0, 0, cw, ch)
                            crop.recycle()
                            result
                        } else {
                            null
                        }

                        DetectionWithMask(det, croppedSegmented)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "detectAllWithMasks failed: ${e.message}")
                    return@withContext emptyList()
                }
            }

    /**
     * Segment the main object in the bitmap. Returns a valid segmented bitmap (transparent
     * background) or original if failed.
     */
    suspend fun segment(bitmap: Bitmap): Bitmap =
            withContext(Dispatchers.Default) {
                if (!isReady()) return@withContext bitmap
                val session = ortSession ?: return@withContext bitmap

                try {
                    // 1. Preprocess
                    val (resizedBitmap, scale, padX, padY) = preprocess(bitmap)
                    val floatBuffer = bitmapToFloatBuffer(resizedBitmap)

                    val inputName = session.inputNames.iterator().next()
                    val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                    val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)

                    // 2. Inference
                    val results = session.run(Collections.singletonMap(inputName, inputTensor))

                    // 3. Process outputs
                    // Output 0: [1, 116, 8400] (boxes + scores + masks)
                    // Output 1: [1, 32, 160, 160] (protos)

                    // Note: Output names/order might vary based on export. Assuming standard.
                    // Usually output0 is main, output1 is protos.
                    val output0 =
                            results[0].value as
                                    Array<Array<FloatArray>> // [1][116][8400] -> Transposed? No.
                    // ONNX Runtime gives java array structures.
                    // Shape check needed.

                    // DEBUG: Print actual shapes
                    Log.d(
                            TAG,
                            "🔍 Output0 shape: [${output0.size}][${output0[0].size}][${output0[0][0].size}]"
                    )

                    // Assuming output0 is the detection head [1, 116, 8400]
                    val detections = output0[0] // [116][8400]

                    // Output 1 is protos [1, 32, 160, 160]
                    val protos =
                            (results[1].value as Array<Array<Array<FloatArray>>>)[
                                    0] // [32][160][160]

                    Log.d(
                            TAG,
                            "🔍 Protos shape: [${protos.size}][${protos[0].size}][${protos[0][0].size}]"
                    )

                    // 4. Decode detections
                    // We need to transpose to [8400][116] for easier loop
                    val numAnchors = detections[0].size // 8400
                    val numChannels = detections.size // 116 (4box + 80cls + 32mask)

                    Log.d(TAG, "🔍 Detection matrix: $numChannels channels x $numAnchors anchors")

                    // DEBUG: Print sample values from first few anchors to verify data format
                    for (sampleIdx in listOf(0, 1000, 4000, 8000)) {
                        val cx = detections[0][sampleIdx]
                        val cy = detections[1][sampleIdx]
                        val w = detections[2][sampleIdx]
                        val h = detections[3][sampleIdx]
                        // Find max class score
                        var maxClassScore = 0f
                        var maxClassId = -1
                        for (c in 0 until NUM_CLASSES) {
                            val score = detections[4 + c][sampleIdx]
                            if (score > maxClassScore) {
                                maxClassScore = score
                                maxClassId = c
                            }
                        }
                        Log.d(
                                TAG,
                                "📊 Anchor[$sampleIdx]: bbox=($cx,$cy,$w,$h) maxClass=$maxClassId maxScore=$maxClassScore"
                        )
                    }

                    val candidates = ArrayList<Detection>()

                    // Helper function: sigmoid activation to convert logits to probabilities
                    fun sigmoid(x: Float): Float = 1.0f / (1.0f + kotlin.math.exp(-x))

                    for (i in 0 until numAnchors) {
                        // Class scores are at indices 4..83 - need sigmoid activation!
                        var maxScore = 0f
                        var classId = -1

                        for (c in 0 until NUM_CLASSES) {
                            val logit = detections[4 + c][i]
                            val score = sigmoid(logit) // Convert logit to probability
                            if (score > maxScore) {
                                maxScore = score
                                classId = c
                            }
                        }

                        if (maxScore > CONFIDENCE_THRESHOLD) {
                            // Extract box
                            val cx = detections[0][i]
                            val cy = detections[1][i]
                            val w = detections[2][i]
                            val h = detections[3][i]

                            val left = (cx - w / 2)
                            val top = (cy - h / 2)
                            val right = (cx + w / 2)
                            val bottom = (cy + h / 2)

                            // Extract mask coefficients (last 32)
                            val maskCoeffs = FloatArray(32)
                            for (m in 0 until 32) {
                                maskCoeffs[m] = detections[4 + NUM_CLASSES + m][i]
                            }

                            candidates.add(
                                    Detection(
                                            classId = classId,
                                            score = maxScore,
                                            box = floatArrayOf(left, top, right, bottom),
                                            maskCoeffs = maskCoeffs
                                    )
                            )
                        }
                    }

                    Log.d(
                            TAG,
                            "YOLO found ${candidates.size} raw candidates (thresh=$CONFIDENCE_THRESHOLD)"
                    )

                    // 5. NMS
                    val bestDetections = nms(candidates)

                    if (bestDetections.isEmpty()) {
                        Log.w(TAG, "⚠️ No detections after NMS! Returning original.")
                        inputTensor.close()
                        return@withContext bitmap
                    }

                    val bestClass = bestDetections[0].classId
                    val bestClassName =
                            if (bestClass in COCO_CLASSES.indices) COCO_CLASSES[bestClass]
                            else "unknown"
                    lastDetectedLabel = bestClassName
                    Log.d(
                            TAG,
                            "Best detection: $bestClassName (id=$bestClass) score=${bestDetections[0].score}"
                    )

                    // Find most central/prominent object
                    // For registration, we assume the object is roughly centered
                    val centerX = INPUT_SIZE / 2f
                    val centerY = INPUT_SIZE / 2f

                    val bestDetection =
                            bestDetections.minByOrNull { det ->
                                val boxCx = (det.box[0] + det.box[2]) / 2
                                val boxCy = (det.box[1] + det.box[3]) / 2
                                // Distance to center
                                kotlin.math.sqrt(
                                        (boxCx - centerX) * (boxCx - centerX) +
                                                (boxCy - centerY) * (boxCy - centerY)
                                )
                            }
                                    ?: bestDetections.first()

                    // 6. Generate Mask
                    val maskBitmap =
                            processMask(
                                    bestDetection,
                                    protos,
                                    scale,
                                    padX,
                                    padY,
                                    bitmap.width,
                                    bitmap.height
                            )

                    // 7. Apply mask to original
                    val result = applyMask(bitmap, maskBitmap)

                    inputTensor.close()
                    return@withContext result
                } catch (e: Exception) {
                    Log.e(TAG, "Inference failed: ${e.message}")
                    e.printStackTrace()
                    return@withContext bitmap
                }
            }

    /**
     * Segment with debug overlay - returns Pair<segmented, overlay> Overlay is a semi-transparent
     * red mask showing what YOLO-seg detected. Use this for debugging visualization.
     */
    suspend fun segmentWithOverlay(bitmap: Bitmap): Pair<Bitmap, Bitmap?> =
            withContext(Dispatchers.Default) {
                if (!isReady()) return@withContext Pair(bitmap, null)
                val session = ortSession ?: return@withContext Pair(bitmap, null)

                try {
                    val (resizedBitmap, scale, padX, padY) = preprocess(bitmap)
                    val floatBuffer = bitmapToFloatBuffer(resizedBitmap)

                    val inputName = session.inputNames.iterator().next()
                    val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                    val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)

                    val results = session.run(Collections.singletonMap(inputName, inputTensor))

                    val output0 = results[0].value as Array<Array<FloatArray>>
                    val detections = output0[0]
                    val protos = (results[1].value as Array<Array<Array<FloatArray>>>)[0]

                    val numAnchors = detections[0].size
                    val candidates = ArrayList<Detection>()

                    for (i in 0 until numAnchors) {
                        var maxScore = 0f
                        var classId = -1
                        for (c in 0 until NUM_CLASSES) {
                            val score = detections[4 + c][i]
                            if (score > maxScore) {
                                maxScore = score
                                classId = c
                            }
                        }
                        if (maxScore > CONFIDENCE_THRESHOLD) {
                            val cx = detections[0][i]
                            val cy = detections[1][i]
                            val w = detections[2][i]
                            val h = detections[3][i]
                            val maskCoeffs = FloatArray(32)
                            for (m in 0 until 32) maskCoeffs[m] = detections[4 + NUM_CLASSES + m][i]
                            candidates.add(
                                    Detection(
                                            classId,
                                            maxScore,
                                            floatArrayOf(
                                                    cx - w / 2,
                                                    cy - h / 2,
                                                    cx + w / 2,
                                                    cy + h / 2
                                            ),
                                            maskCoeffs
                                    )
                            )
                        }
                    }

                    if (candidates.isEmpty()) {
                        inputTensor.close()
                        return@withContext Pair(bitmap, null)
                    }

                    val bestDetections = nms(candidates)
                    val centerX = INPUT_SIZE / 2f
                    val centerY = INPUT_SIZE / 2f

                    val bestDetection =
                            bestDetections.minByOrNull { det ->
                                val boxCx = (det.box[0] + det.box[2]) / 2
                                val boxCy = (det.box[1] + det.box[3]) / 2
                                kotlin.math.sqrt(
                                        (boxCx - centerX) * (boxCx - centerX) +
                                                (boxCy - centerY) * (boxCy - centerY)
                                )
                            }
                                    ?: bestDetections.first()

                    val bestClassName = getLabel(bestDetection.classId)
                    Log.d(TAG, "Debug overlay: $bestClassName score=${bestDetection.score}")

                    val maskBitmap =
                            processMask(
                                    bestDetection,
                                    protos,
                                    scale,
                                    padX,
                                    padY,
                                    bitmap.width,
                                    bitmap.height
                            )
                    val segmentedResult = applyMask(bitmap, maskBitmap)

                    // Create RED overlay for debugging
                    val overlay =
                            createRedOverlay(bitmap, maskBitmap, bestClassName, bestDetection.score)

                    inputTensor.close()
                    return@withContext Pair(segmentedResult, overlay)
                } catch (e: Exception) {
                    Log.e(TAG, "segmentWithOverlay failed: ${e.message}")
                    return@withContext Pair(bitmap, null)
                }
            }

    /** Create a red semi-transparent overlay showing the mask with class label */
    private fun createRedOverlay(
            original: Bitmap,
            mask: Bitmap,
            label: String,
            score: Float
    ): Bitmap {
        val result = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)

        val origPixels = IntArray(original.width * original.height)
        original.getPixels(origPixels, 0, original.width, 0, 0, original.width, original.height)

        val maskPixels = IntArray(original.width * original.height)
        mask.getPixels(maskPixels, 0, original.width, 0, 0, original.width, original.height)

        for (i in origPixels.indices) {
            val origPixel = origPixels[i]
            if (Color.alpha(maskPixels[i]) > 128) {
                // Blend with green (to differentiate from FastSAM red)
                val r = (Color.red(origPixel) / 2).coerceIn(0, 255)
                val g = ((Color.green(origPixel) + 200) / 2).coerceIn(0, 255)
                val b = (Color.blue(origPixel) / 2).coerceIn(0, 255)
                origPixels[i] = Color.argb(255, r, g, b)
            }
        }
        result.setPixels(origPixels, 0, original.width, 0, 0, original.width, original.height)

        // Draw label on top
        val canvas = android.graphics.Canvas(result)
        val paint =
                android.graphics.Paint().apply {
                    color = Color.WHITE
                    textSize = 48f
                    isAntiAlias = true
                    setShadowLayer(4f, 2f, 2f, Color.BLACK)
                }
        canvas.drawText("$label (${String.format("%.2f", score)})", 20f, 60f, paint)

        return result
    }

    private fun processMask(
            det: Detection,
            protos: Array<Array<FloatArray>>, // [32][160][160]
            scale: Float,
            padX: Float,
            padY: Float,
            origW: Int,
            origH: Int
    ): Bitmap {
        // MatMul: mask_coeffs (1x32) * protos (32x160x160) -> 160x160
        val maskData = FloatArray(MASK_SIZE * MASK_SIZE)

        for (h in 0 until MASK_SIZE) {
            for (w in 0 until MASK_SIZE) {
                var sum = 0f
                for (c in 0 until 32) {
                    sum += det.maskCoeffs[c] * protos[c][h][w]
                }
                // Sigmoid
                maskData[h * MASK_SIZE + w] = (1.0 / (1.0 + exp(-sum))).toFloat()
            }
        }

        // Resize mask to input size (640x640) -> crop to original image coords
        // We'll construct the bitmap directly at output size for efficiency
        val outputMask = Bitmap.createBitmap(origW, origH, Bitmap.Config.ALPHA_8)

        // Very simplified scaling (nearest neighbor) for speed
        // Real implementation should use better interpolation or standard resize
        val pixels = IntArray(origW * origH)

        for (y in 0 until origH) {
            for (x in 0 until origW) {
                // Map original (x,y) to resized input (640x640)
                val inputX = x * scale + padX
                val inputY = y * scale + padY

                // Map to mask size (160x160)
                // Mask is 160, Input is 640 -> factor 0.25
                val maskX = (inputX * 0.25f).toInt().coerceIn(0, MASK_SIZE - 1)
                val maskY = (inputY * 0.25f).toInt().coerceIn(0, MASK_SIZE - 1)

                val maskVal = maskData[maskY * MASK_SIZE + maskX]

                // Check if inside box (optional, but cleaner)
                // Note: Box is in 640 coords
                // Since this is per-pixel, segmentation should be fine even outside box
                // typically we crop to box, but for speed we might skip

                val alpha = if (maskVal > 0.5f) 255 else 0
                pixels[y * origW + x] = Color.argb(alpha, 255, 255, 255)
            }
        }

        // This creates a Color_8888 bitmap but we put alpha into ARGB
        return Bitmap.createBitmap(pixels, origW, origH, Bitmap.Config.ARGB_8888)
    }

    private fun applyMask(original: Bitmap, mask: Bitmap): Bitmap {
        val result = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(result)

        // Draw mask (it determines alpha)
        val paint = android.graphics.Paint()
        // Simple way: Iterate pixels. Canvas blending is complex for masking.

        val origPixels = IntArray(original.width * original.height)
        original.getPixels(origPixels, 0, original.width, 0, 0, original.width, original.height)

        val maskPixels = IntArray(original.width * original.height)
        mask.getPixels(maskPixels, 0, original.width, 0, 0, original.width, original.height)

        for (i in origPixels.indices) {
            val alpha = android.graphics.Color.alpha(maskPixels[i])
            if (alpha > 128) {
                // Keep pixel
                // Ensure full opacity if it was transparent in original? Original is camera frame
                // (opaque)
            } else {
                // Transparent
                origPixels[i] = 0 // Clear pixel
            }
        }

        result.setPixels(origPixels, 0, original.width, 0, 0, original.width, original.height)
        return result
    }

    private fun preprocess(bitmap: Bitmap): PreprocessResult {
        // Resize while maintaining aspect ratio, pad with gray
        val w = bitmap.width
        val h = bitmap.height
        val scale = minOf(INPUT_SIZE.toFloat() / w, INPUT_SIZE.toFloat() / h)

        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()

        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val padded = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(padded)
        canvas.drawColor(Color.GRAY)

        val padX = (INPUT_SIZE - newW) / 2f
        val padY = (INPUT_SIZE - newH) / 2f

        canvas.drawBitmap(resized, padX, padY, null)

        return PreprocessResult(padded, scale, padX, padY)
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // YOLO expects CHW format (Channel, Height, Width)
        // First all R values, then all G values, then all B values
        val pixelCount = INPUT_SIZE * INPUT_SIZE

        // Channel R
        for (i in 0 until pixelCount) {
            val pixel = intValues[i]
            buffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        }
        // Channel G
        for (i in 0 until pixelCount) {
            val pixel = intValues[i]
            buffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        }
        // Channel B
        for (i in 0 until pixelCount) {
            val pixel = intValues[i]
            buffer.put((pixel and 0xFF) / 255.0f)
        }

        buffer.rewind()
        return buffer
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val selected = ArrayList<Detection>()

        for (candidate in sorted) {
            var keep = true
            for (saved in selected) {
                if (iou(candidate.box, saved.box) > IOU_THRESHOLD) {
                    keep = false
                    break
                }
            }
            if (keep) {
                selected.add(candidate)
            }
        }
        return selected
    }

    private fun iou(boxA: FloatArray, boxB: FloatArray): Float {
        val xA = maxOf(boxA[0], boxB[0])
        val yA = maxOf(boxA[1], boxB[1])
        val xB = minOf(boxA[2], boxB[2])
        val yB = minOf(boxA[3], boxB[3])

        val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
        val boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        val boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / (boxAArea + boxBArea - interArea)
    }

    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            // Do NOT close ortEnv — it's a process-wide singleton shared by all ONNX models
        } catch (e: Exception) {
            // ignore
        }
    }

    data class Detection(
            val classId: Int,
            val score: Float,
            val box: FloatArray, // [left, top, right, bottom]
            val maskCoeffs: FloatArray
    )

    data class DetectionWithMask(
            val detection: Detection,
            val segmentedCrop: Bitmap?  // Cropped + background removed, null if crop too small
    )

    data class PreprocessResult(
            val bitmap: Bitmap,
            val scale: Float,
            val padX: Float,
            val padY: Float
    )
}
