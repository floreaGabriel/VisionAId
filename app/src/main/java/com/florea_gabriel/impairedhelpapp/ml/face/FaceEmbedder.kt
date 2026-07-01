package com.florea_gabriel.impairedhelpapp.ml.face

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
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
