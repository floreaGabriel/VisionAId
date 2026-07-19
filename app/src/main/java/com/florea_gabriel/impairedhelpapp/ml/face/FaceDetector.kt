package com.florea_gabriel.impairedhelpapp.ml.face

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FaceDetector(private val context: Context) {

    companion object {
        private const val TAG = "FaceDetector"
        private const val MODEL_FILE = "face_detection_yunet_2023mar.onnx"
        private const val INPUT_SIZE = 640
        // Score = sqrt(cls * obj), same scale as OpenCV FaceDetectorYN
        private const val CONFIDENCE_THRESHOLD = 0.6f
        private const val IOU_THRESHOLD = 0.3f
        // Minimum face size in original bitmap pixels
        private const val MIN_FACE_SIZE = 12f

        private val STRIDES = intArrayOf(8, 16, 32)
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val anchorCache: List<Anchor> by lazy { generateAnchors() }

    init {
        initModel()
    }

    private fun initModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                setInterOpNumThreads(2)
            }

            val modelBytes = context.assets.open(MODEL_FILE).readBytes()
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
            Log.d(TAG, "YuNet initialized (${modelBytes.size / 1024}KB)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init YuNet: ${e.message}")
        }
    }

    fun isReady(): Boolean = ortSession != null

    suspend fun detectFaces(bitmap: Bitmap): List<FaceDetection> =
        withContext(Dispatchers.Default) {
            if (!isReady()) return@withContext emptyList()
            val session = ortSession ?: return@withContext emptyList()

            try {
                // Letterbox: uniform scale + black padding right/bottom,
                // same preprocessing as OpenCV FaceDetectorYN (no aspect distortion)
                val scale = minOf(
                    INPUT_SIZE.toFloat() / bitmap.width,
                    INPUT_SIZE.toFloat() / bitmap.height
                )
                val scaledW = (bitmap.width * scale).toInt().coerceAtLeast(1)
                val scaledH = (bitmap.height * scale).toInt().coerceAtLeast(1)

                val letterboxed = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(letterboxed)
                canvas.drawColor(android.graphics.Color.BLACK)
                canvas.drawBitmap(bitmap, null, Rect(0, 0, scaledW, scaledH), Paint(Paint.FILTER_BITMAP_FLAG))

                val floatBuffer = bitmapToFloatBuffer(letterboxed)
                val inputName = session.inputNames.iterator().next()
                val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
                val results = session.run(Collections.singletonMap(inputName, inputTensor))

                val candidates = decodeOutputs(results, bitmap.width.toFloat(), bitmap.height.toFloat(), scale)

                inputTensor.close()
                results.close()
                letterboxed.recycle()

                val after = nms(candidates)
                Log.d(TAG, "RESULT: ${candidates.size} candidates -> ${after.size} after NMS")
                return@withContext after
            } catch (e: Exception) {
                Log.e(TAG, "detectFaces failed: ${e.message}", e)
                return@withContext emptyList()
            }
        }

    private fun decodeOutputs(
        results: OrtSession.Result,
        bmpW: Float,
        bmpH: Float,
        scale: Float
    ): List<FaceDetection> {

        val cls = Array(3) { i ->
            (results.get("cls_${STRIDES[i]}").get().value as Array<Array<FloatArray>>)[0]
        }
        val obj = Array(3) { i ->
            (results.get("obj_${STRIDES[i]}").get().value as Array<Array<FloatArray>>)[0]
        }
        val bbox = Array(3) { i ->
            (results.get("bbox_${STRIDES[i]}").get().value as Array<Array<FloatArray>>)[0]
        }
        val kps = Array(3) { i ->
            (results.get("kps_${STRIDES[i]}").get().value as Array<Array<FloatArray>>)[0]
        }

        val candidates = ArrayList<FaceDetection>()
        var anchorIdx = 0

        for (strideIdx in STRIDES.indices) {
            val numAnchors = cls[strideIdx].size

            for (i in 0 until numAnchors) {
                // OpenCV FaceDetectorYN score: sqrt(cls * obj), both clamped to [0,1]
                val clsScore = cls[strideIdx][i][0].coerceIn(0f, 1f)
                val objScore = obj[strideIdx][i][0].coerceIn(0f, 1f)
                val conf = sqrt(clsScore * objScore)

                if (conf < CONFIDENCE_THRESHOLD) {
                    anchorIdx++
                    continue
                }

                val anchor = anchorCache[anchorIdx]

                // OpenCV decode: cx = (col + dx) * stride — anchor sits at col*stride, no +0.5
                val cx = anchor.cx + bbox[strideIdx][i][0] * anchor.size
                val cy = anchor.cy + bbox[strideIdx][i][1] * anchor.size
                val w = exp(bbox[strideIdx][i][2]) * anchor.size
                val h = exp(bbox[strideIdx][i][3]) * anchor.size

                // Map back to original bitmap coordinates (single letterbox scale)
                val left = ((cx - w / 2f) / scale).coerceIn(0f, bmpW)
                val top = ((cy - h / 2f) / scale).coerceIn(0f, bmpH)
                val right = ((cx + w / 2f) / scale).coerceIn(0f, bmpW)
                val bottom = ((cy + h / 2f) / scale).coerceIn(0f, bmpH)

                val boxW = right - left
                val boxH = bottom - top
                if (boxW < MIN_FACE_SIZE || boxH < MIN_FACE_SIZE) {
                    anchorIdx++
                    continue
                }

                val landmarks = FloatArray(10)
                for (j in 0 until 5) {
                    val lx = anchor.cx + kps[strideIdx][i][2 * j] * anchor.size
                    val ly = anchor.cy + kps[strideIdx][i][2 * j + 1] * anchor.size
                    landmarks[2 * j] = (lx / scale).coerceIn(0f, bmpW)
                    landmarks[2 * j + 1] = (ly / scale).coerceIn(0f, bmpH)
                }

                candidates.add(
                    FaceDetection(
                        box = floatArrayOf(left, top, right, bottom),
                        confidence = conf,
                        landmarks = landmarks
                    )
                )

                anchorIdx++
            }
        }

        return candidates
    }

    private fun generateAnchors(): List<Anchor> {
        val anchors = ArrayList<Anchor>()
        for (stride in STRIDES) {
            val gridH = INPUT_SIZE / stride
            val gridW = INPUT_SIZE / stride
            for (row in 0 until gridH) {
                for (col in 0 until gridW) {
                    // Anchor at cell origin: OpenCV decodes cx = (col + dx) * stride
                    anchors.add(
                        Anchor(
                            cx = col * stride.toFloat(),
                            cy = row * stride.toFloat(),
                            size = stride.toFloat()
                        )
                    )
                }
            }
        }
        return anchors
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val pixelCount = INPUT_SIZE * INPUT_SIZE

        // YuNet expects BGR, raw 0-255 (NO normalization)
        for (i in 0 until pixelCount) {
            buffer.put((intValues[i] and 0xFF).toFloat())           // B
        }
        for (i in 0 until pixelCount) {
            buffer.put(((intValues[i] shr 8) and 0xFF).toFloat())   // G
        }
        for (i in 0 until pixelCount) {
            buffer.put(((intValues[i] shr 16) and 0xFF).toFloat())  // R
        }

        buffer.rewind()
        return buffer
    }

    private fun nms(detections: List<FaceDetection>): List<FaceDetection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = ArrayList<FaceDetection>()

        for (candidate in sorted) {
            var keep = true
            for (saved in selected) {
                if (iou(candidate.box, saved.box) > IOU_THRESHOLD) {
                    keep = false
                    break
                }
            }
            if (keep) selected.add(candidate)
        }
        Log.d(TAG, "Detected ${selected.size} faces")
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

        return interArea / (boxAArea + boxBArea - interArea + 1e-6f)
    }

    fun close() {
        try {
            ortSession?.close()
            ortSession = null
        } catch (_: Exception) {}
    }

    data class FaceDetection(
        val box: FloatArray,
        val confidence: Float,
        val landmarks: FloatArray? = null
    )

    private data class Anchor(val cx: Float, val cy: Float, val size: Float)
}
