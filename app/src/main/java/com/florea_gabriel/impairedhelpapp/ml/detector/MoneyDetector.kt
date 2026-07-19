package com.florea_gabriel.impairedhelpapp.ml.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.florea_gabriel.impairedhelpapp.data.model.Detection
import com.florea_gabriel.impairedhelpapp.utils.Constants
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * MoneyDetector: ONNX-based YOLO26n detector for RON banknotes.
 *
 * Model: yolo26n_money.onnx (YOLO26n, 8 classes — custom-trained on personal dataset)
 * Input: [1, 3, 640, 640] NCHW float32, normalized /255
 * Output: [1, 300, 6] end-to-end (NMS baked into the model) — up to 300 rows of
 *         [x1, y1, x2, y2, score, classId]. (The model was exported with nms=True.)
 * Classes: 100_ron, 10_ron, 1_ron, 200_ron, 20_ron, 500_ron, 50_ron, 5_ron
 */
class MoneyDetector(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val inputSize = Constants.MONEY_INPUT_SIZE
    private var labels: List<String> = emptyList()

    companion object {
        private const val TAG = "MoneyDetector"
        private const val CONFIDENCE_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.45f
        private const val NUM_CLASSES = 8

        // Label → numeric value mapping
        private val LABEL_TO_VALUE = mapOf(
            "100_ron" to 100,
            "10_ron" to 10,
            "1_ron" to 1,
            "200_ron" to 200,
            "20_ron" to 20,
            "500_ron" to 500,
            "50_ron" to 50,
            "5_ron" to 5
        )

        fun labelToValue(label: String): Int = LABEL_TO_VALUE[label] ?: 0
    }

    init {
        loadLabels()
        loadModel()
    }

    private fun loadLabels() {
        try {
            labels = context.assets.open(Constants.MONEY_LABELS_FILE)
                .bufferedReader()
                .readLines()
                .filter { it.isNotBlank() }
            Log.d(TAG, "Loaded ${labels.size} labels: $labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels: ${e.message}", e)
        }
    }

    private fun copyAssetToInternal(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (!outFile.exists()) {
            Log.d(TAG, "Copying $assetName to internal storage...")
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied $assetName (${outFile.length()} bytes)")
        }
        return outFile
    }

    private fun loadModel() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelFile = copyAssetToInternal(Constants.MONEY_MODEL_FILE)

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(Constants.NUM_THREADS)
                setInterOpNumThreads(2)
            }

            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
            Log.d(TAG, "Money model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
        }
    }

    private data class Letterbox(val scale: Float, val dx: Float, val dy: Float)

    /**
     * Ultralytics-style letterbox: uniform scale + centered gray (114) padding.
     * Matches the preprocessing the model was trained/exported with — no
     * aspect-ratio distortion.
     */
    private fun letterbox(bitmap: Bitmap): Pair<Bitmap, Letterbox> {
        val scale = minOf(
            inputSize.toFloat() / bitmap.width,
            inputSize.toFloat() / bitmap.height
        )
        val scaledW = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val scaledH = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val dx = (inputSize - scaledW) / 2f
        val dy = (inputSize - scaledH) / 2f

        val out = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(
            bitmap, null,
            RectF(dx, dy, dx + scaledW, dy + scaledH),
            Paint(Paint.FILTER_BITMAP_FLAG)
        )
        return out to Letterbox(scale, dx, dy)
    }

    /**
     * Convert a 640x640 letterboxed bitmap to NCHW float buffer, normalized /255.
     */
    private fun bitmapToBuffer(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // NCHW: channels first (R, G, B)
        for (c in 0 until 3) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val value = when (c) {
                    0 -> ((pixel shr 16) and 0xFF) / 255.0f // R
                    1 -> ((pixel shr 8) and 0xFF) / 255.0f  // G
                    2 -> (pixel and 0xFF) / 255.0f           // B
                    else -> 0f
                }
                buffer.put(value)
            }
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Run detection on a bitmap.
     * @param bitmap Input camera frame
     * @return List of detections with bounding boxes in original frame coordinates
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val session = ortSession ?: return emptyList()
        val env = ortEnvironment ?: return emptyList()

        try {
            val (letterboxed, lb) = letterbox(bitmap)
            val inputBuffer = bitmapToBuffer(letterboxed)
            letterboxed.recycle()

            val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)
            val inputName = session.inputNames.first()

            val results = session.run(mapOf(inputName to inputTensor))
            val outputTensor = results.get(0) as? OnnxTensor
            inputTensor.close()

            if (outputTensor == null) {
                results.close()
                return emptyList()
            }

            val output = outputTensor.floatBuffer
            val detections = parseOutput(output, bitmap.width, bitmap.height, lb)
            results.close()

            val finalDetections = nms(detections, IOU_THRESHOLD)
            Log.d(TAG, "Detections: ${detections.size} raw → ${finalDetections.size} after NMS" +
                    if (finalDetections.isNotEmpty()) " | top: ${finalDetections[0].label} ${"%.2f".format(finalDetections[0].confidence)}" else "")
            return finalDetections
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}", e)
            return emptyList()
        }
    }

    /**
     * Parse YOLO end-to-end (NMS-included) output, shape [1, 300, 6].
     * Each of the up-to-300 rows is already NMS-filtered: [x1, y1, x2, y2, score, classId].
     * Box coordinates are in the model input space (0..inputSize); undo the
     * letterbox (padding offset + uniform scale) to get original frame coordinates.
     */
    private fun parseOutput(
        buffer: FloatBuffer,
        srcWidth: Int,
        srcHeight: Int,
        lb: Letterbox
    ): List<Detection> {
        buffer.rewind()

        val maxDets = 300
        val stride = 6   // x1, y1, x2, y2, score, classId

        val detections = mutableListOf<Detection>()
        var globalMaxScore = 0f  // diagnostic: highest score seen this frame

        for (i in 0 until maxDets) {
            val base = i * stride
            if (base + stride > buffer.limit()) break

            val x1 = buffer.get(base)
            val y1 = buffer.get(base + 1)
            val x2 = buffer.get(base + 2)
            val y2 = buffer.get(base + 3)
            val score = buffer.get(base + 4)
            val classId = buffer.get(base + 5).toInt()

            if (score > globalMaxScore) globalMaxScore = score
            if (score < CONFIDENCE_THRESHOLD) continue

            val label = if (classId in labels.indices) labels[classId] else "class_$classId"

            detections.add(
                Detection(
                    label = label,
                    confidence = score,
                    boundingBox = RectF(
                        ((x1 - lb.dx) / lb.scale).coerceIn(0f, srcWidth.toFloat()),
                        ((y1 - lb.dy) / lb.scale).coerceIn(0f, srcHeight.toFloat()),
                        ((x2 - lb.dx) / lb.scale).coerceIn(0f, srcWidth.toFloat()),
                        ((y2 - lb.dy) / lb.scale).coerceIn(0f, srcHeight.toFloat())
                    )
                )
            )
        }

        Log.d(TAG, "Max score this frame: ${"%.3f".format(globalMaxScore)} (threshold $CONFIDENCE_THRESHOLD)")
        return detections
    }

    /**
     * Non-Maximum Suppression: keep only the best non-overlapping detections.
     */
    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { other ->
                iou(best.boundingBox, other.boundingBox) > iouThreshold
            }
        }

        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        val unionArea = aArea + bArea - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            Log.d(TAG, "MoneyDetector closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing session: ${e.message}")
        }
    }
}
