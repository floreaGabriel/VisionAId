package com.florea_gabriel.impairedhelpapp.ml.face

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * FaceEmbedder: ONNX wrapper for MobileFaceNet face recognition model.
 *
 * Model: mobilefacenet.onnx
 * - Input: (1, 3, 112, 112) normalized (px - 127.5) / 127.5
 * - Output: (1, 512) L2-normalized embedding
 */
class FaceEmbedder(private val context: Context) {

    companion object {
        private const val TAG = "FaceEmbedder"
        private const val MODEL_FILE = "mobilefacenet.onnx"
        private const val INPUT_SIZE = 112
        const val EMBEDDING_SIZE = 512
        const val RECOGNITION_THRESHOLD = 0.45f

        // Fallback crop padding (fraction of box size) — shared by all callers
        const val CROP_PADDING = 0.2f

        // ArcFace 112x112 reference landmarks: left eye, right eye, nose,
        // left mouth corner, right mouth corner (image-left = smaller x)
        private val ARCFACE_TEMPLATE = floatArrayOf(
            38.2946f, 51.6963f,
            73.5318f, 51.5014f,
            56.0252f, 71.7366f,
            41.5493f, 92.3655f,
            70.7299f, 92.2041f
        )
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false

    init {
        loadModel()
    }

    fun isReady(): Boolean = isInitialized && ortSession != null

    private fun copyModelToInternal(): File? {
        val outFile = File(context.filesDir, MODEL_FILE)
        if (!outFile.exists()) {
            try {
                context.assets.open(MODEL_FILE).use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
                Log.d(TAG, "Copied MobileFaceNet model (${outFile.length() / 1024 / 1024}MB)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy model: ${e.message}")
                return null
            }
        }
        return outFile
    }

    private fun loadModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            val modelFile = copyModelToInternal()
            if (modelFile == null || !modelFile.exists()) {
                Log.e(TAG, "MobileFaceNet model not found. Add $MODEL_FILE to assets/")
                return
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                setInterOpNumThreads(2)
            }

            ortSession = ortEnv?.createSession(modelFile.absolutePath, sessionOptions)

            if (ortSession != null) {
                isInitialized = true
                Log.d(TAG, "MobileFaceNet loaded successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading MobileFaceNet: ${e.message}", e)
            isInitialized = false
        }
    }

    /**
     * Extract 512D face embedding from a cropped face bitmap.
     * @param faceBitmap Cropped face image (any size, will be resized to 112x112)
     * @return L2-normalized 512D embedding, or null on failure
     */
    suspend fun extractEmbedding(faceBitmap: Bitmap): FloatArray? =
        withContext(Dispatchers.Default) {
            if (!isReady()) return@withContext null
            val session = ortSession ?: return@withContext null
            val env = ortEnv ?: return@withContext null

            try {
                val resized = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)
                val floatBuffer = preprocessFace(resized)
                if (resized != faceBitmap) resized.recycle()

                val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
                val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)
                val inputName = session.inputNames.first()

                val results = session.run(mapOf(inputName to inputTensor))
                val outputTensor = results.get(0) as? OnnxTensor
                inputTensor.close()

                if (outputTensor == null) {
                    results.close()
                    return@withContext null
                }

                val buffer = outputTensor.floatBuffer
                val embedding = FloatArray(EMBEDDING_SIZE)
                buffer.rewind()
                buffer.get(embedding)
                results.close()

                // L2 normalize
                normalizeL2(embedding)

                return@withContext embedding
            } catch (e: Exception) {
                Log.e(TAG, "extractEmbedding failed: ${e.message}")
                return@withContext null
            }
        }

    /**
     * Extract an embedding from the full frame using the detector's 5 landmarks
     * for ArcFace-style alignment. Falls back to a padded box crop when
     * landmarks are missing or degenerate.
     *
     * @param bitmap Full camera frame
     * @param box Face box [left, top, right, bottom] in bitmap coordinates
     * @param landmarks 5 points (10 floats) in bitmap coordinates, or null
     */
    suspend fun extractAlignedEmbedding(
        bitmap: Bitmap,
        box: FloatArray,
        landmarks: FloatArray?
    ): FloatArray? {
        val face = alignFace(bitmap, landmarks) ?: cropFace(bitmap, box) ?: return null
        val embedding = extractEmbedding(face)
        face.recycle()
        return embedding
    }

    /**
     * Warp the face to the 112x112 ArcFace template via a similarity transform
     * (rotation + uniform scale + translation) estimated from the 5 landmarks
     * by least squares. Returns null if landmarks are unusable.
     */
    private fun alignFace(bitmap: Bitmap, landmarks: FloatArray?): Bitmap? {
        if (landmarks == null || landmarks.size < 10) return null

        // Normalize ordering by x so eye/mouth left-right naming can't flip the face
        val src = FloatArray(10)
        val (eyeL, eyeR) = if (landmarks[0] <= landmarks[2]) 0 to 1 else 1 to 0
        src[0] = landmarks[eyeL * 2]; src[1] = landmarks[eyeL * 2 + 1]
        src[2] = landmarks[eyeR * 2]; src[3] = landmarks[eyeR * 2 + 1]
        src[4] = landmarks[4]; src[5] = landmarks[5]
        val (mouthL, mouthR) = if (landmarks[6] <= landmarks[8]) 3 to 4 else 4 to 3
        src[6] = landmarks[mouthL * 2]; src[7] = landmarks[mouthL * 2 + 1]
        src[8] = landmarks[mouthR * 2]; src[9] = landmarks[mouthR * 2 + 1]

        // Degenerate landmarks (face too small / collapsed points) → fallback crop
        val eyeDx = src[2] - src[0]
        val eyeDy = src[3] - src[1]
        if (eyeDx * eyeDx + eyeDy * eyeDy < 16f) return null

        // Least-squares similarity transform src → ARCFACE_TEMPLATE:
        // u = a*x - b*y + tx, v = b*x + a*y + ty
        val n = 5
        var sx = 0f; var sy = 0f; var su = 0f; var sv = 0f
        var sxx = 0f; var sxu = 0f; var sxv = 0f
        for (i in 0 until n) {
            val x = src[2 * i]; val y = src[2 * i + 1]
            val u = ARCFACE_TEMPLATE[2 * i]; val v = ARCFACE_TEMPLATE[2 * i + 1]
            sx += x; sy += y; su += u; sv += v
            sxx += x * x + y * y
            sxu += x * u + y * v
            sxv += x * v - y * u
        }
        val denom = sxx - (sx * sx + sy * sy) / n
        if (denom < 1e-3f) return null

        val a = (sxu - (sx * su + sy * sv) / n) / denom
        val b = (sxv - (sx * sv - sy * su) / n) / denom
        val tx = (su - a * sx + b * sy) / n
        val ty = (sv - b * sx - a * sy) / n

        val matrix = Matrix()
        matrix.setValues(floatArrayOf(a, -b, tx, b, a, ty, 0f, 0f, 1f))

        val aligned = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(aligned)
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(bitmap, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        return aligned
    }

    /**
     * Fallback: axis-aligned crop of the face box with CROP_PADDING on each side.
     */
    private fun cropFace(bitmap: Bitmap, box: FloatArray): Bitmap? {
        val fw = box[2] - box[0]
        val fh = box[3] - box[1]
        val padX = (fw * CROP_PADDING).toInt()
        val padY = (fh * CROP_PADDING).toInt()
        val cropLeft = (box[0].toInt() - padX).coerceAtLeast(0)
        val cropTop = (box[1].toInt() - padY).coerceAtLeast(0)
        val cropRight = (box[2].toInt() + padX).coerceAtMost(bitmap.width)
        val cropBottom = (box[3].toInt() + padY).coerceAtMost(bitmap.height)
        val cw = cropRight - cropLeft
        val ch = cropBottom - cropTop
        if (cw < 20 || ch < 20) return null
        return Bitmap.createBitmap(bitmap, cropLeft, cropTop, cw, ch)
    }

    /**
     * Preprocess face bitmap for MobileFaceNet.
     * Normalize: (pixel - 127.5) / 127.5 → range [-1, 1]
     * Format: CHW
     */
    private fun preprocessFace(bitmap: Bitmap): FloatBuffer {
        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val pixelCount = INPUT_SIZE * INPUT_SIZE

        // Channel R
        for (i in 0 until pixelCount) {
            val r = ((pixels[i] shr 16) and 0xFF).toFloat()
            buffer.put((r - 127.5f) / 127.5f)
        }
        // Channel G
        for (i in 0 until pixelCount) {
            val g = ((pixels[i] shr 8) and 0xFF).toFloat()
            buffer.put((g - 127.5f) / 127.5f)
        }
        // Channel B
        for (i in 0 until pixelCount) {
            val b = (pixels[i] and 0xFF).toFloat()
            buffer.put((b - 127.5f) / 127.5f)
        }

        buffer.rewind()
        return buffer
    }

    private fun normalizeL2(vector: FloatArray) {
        var norm = 0f
        for (v in vector) norm += v * v
        norm = sqrt(norm)
        if (norm > 0f) {
            for (i in vector.indices) vector[i] /= norm
        }
    }

    /** Cosine similarity between two embeddings. */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float =
        com.florea_gabriel.impairedhelpapp.utils.cosineSimilarity(a, b)

    /** Serialize a 512D embedding to ByteArray for Room BLOB storage. */
    fun embeddingToBlob(embedding: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(embedding.size * 4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        for (v in embedding) buffer.putFloat(v)
        return buffer.array()
    }

    /** Deserialize ByteArray back to 512D embedding. */
    fun blobToEmbedding(blob: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(blob)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(blob.size / 4) { buffer.getFloat() }
    }

    /**
     * Serialize multiple 512D embeddings to ByteArray for Room BLOB storage.
     * Format: [count:Int32] + N × 512 × Float32
     */
    fun embeddingsToBlob(embeddings: List<FloatArray>): ByteArray {
        val buffer = ByteBuffer.allocate(4 + embeddings.size * EMBEDDING_SIZE * 4)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(embeddings.size)
        for (emb in embeddings) {
            for (v in emb) buffer.putFloat(v)
        }
        return buffer.array()
    }

    /**
     * Deserialize ByteArray back to list of 512D embeddings.
     * Legacy support: if blob.size == 2048 (single embedding, no header), returns single-element list.
     */
    fun embeddingsFromBlob(blob: ByteArray): List<FloatArray> {
        // Legacy single-embedding format: exactly 512 floats × 4 bytes = 2048 bytes, no header
        if (blob.size == EMBEDDING_SIZE * 4) {
            return listOf(blobToEmbedding(blob))
        }
        val buffer = ByteBuffer.wrap(blob)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val count = buffer.getInt()
        return List(count) { FloatArray(EMBEDDING_SIZE) { buffer.getFloat() } }
    }

    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            isInitialized = false
        } catch (_: Exception) {}
    }
}
