package com.florea_gabriel.impairedhelpapp.ml.face

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FaceDetector(private val context: Context) {

    companion object {
        private const val TAG = "FaceDetector"
        private const val MODEL_FILE = "face_detection_yunet_2023mar.onnx"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.3f

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
                // Direct resize to 640x640 (no letterbox padding)
                val scaleX = INPUT_SIZE.toFloat() / bitmap.width
                val scaleY = INPUT_SIZE.toFloat() / bitmap.height
                val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

                Log.d(TAG, "INPUT: bitmap=${bitmap.width}x${bitmap.height}, scaleX=$scaleX, scaleY=$scaleY")

                val floatBuffer = bitmapToFloatBuffer(resized)
                val inputName = session.inputNames.iterator().next()
                val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
                val results = session.run(Collections.singletonMap(inputName, inputTensor))

                val candidates = decodeOutputs(results, bitmap.width.toFloat(), bitmap.height.toFloat(), scaleX, scaleY)

                inputTensor.close()
                results.close()
                if (resized != bitmap) resized.recycle()

                val after = nms(candidates)
                Log.d(TAG, "RESULT: ${candidates.size} candidates -> ${after.size} after NMS")
                for ((idx, f) in after.withIndex()) {
                    Log.d(TAG, "  face[$idx]: box=[${f.box[0].toInt()},${f.box[1].toInt()},${f.box[2].toInt()},${f.box[3].toInt()}] conf=${f.confidence}")
                }
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
        scaleX: Float,
        scaleY: Float
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
                // cls and obj are already sigmoid — multiply directly
                val conf = cls[strideIdx][i][0] * obj[strideIdx][i][0]

                if (conf < CONFIDENCE_THRESHOLD) {
                    anchorIdx++
                    continue
                }

                val anchor = anchorCache[anchorIdx]

                val cx = anchor.cx + bbox[strideIdx][i][0] * anchor.size
                val cy = anchor.cy + bbox[strideIdx][i][1] * anchor.size
                val w = exp(bbox[strideIdx][i][2]) * anchor.size
                val h = exp(bbox[strideIdx][i][3]) * anchor.size

                // Map back to original bitmap coordinates
                val left = ((cx - w / 2f) / scaleX).coerceIn(0f, bmpW)
                val top = ((cy - h / 2f) / scaleY).coerceIn(0f, bmpH)
                val right = ((cx + w / 2f) / scaleX).coerceIn(0f, bmpW)
                val bottom = ((cy + h / 2f) / scaleY).coerceIn(0f, bmpH)

                val boxW = right - left
                val boxH = bottom - top
                if (boxW < 30f || boxH < 30f) {
                    anchorIdx++
                    continue
                }

                val landmarks = FloatArray(10)
                for (j in 0 until 5) {
                    val lx = anchor.cx + kps[strideIdx][i][2 * j] * anchor.size
                    val ly = anchor.cy + kps[strideIdx][i][2 * j + 1] * anchor.size
                    landmarks[2 * j] = (lx / scaleX).coerceIn(0f, bmpW)
                    landmarks[2 * j + 1] = (ly / scaleY).coerceIn(0f, bmpH)
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
                    // 1 anchor per position, stride as size
                    anchors.add(
                        Anchor(
                            cx = (col + 0.5f) * stride,
                            cy = (row + 0.5f) * stride,
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
