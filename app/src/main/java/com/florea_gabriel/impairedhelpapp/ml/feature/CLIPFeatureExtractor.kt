package com.florea_gabriel.impairedhelpapp.ml.feature

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.sqrt
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * CLIPFeatureExtractor: Extracts CLIP embeddings from images using ONNX Runtime.
 *
 * Uses MobileCLIP2-S2 model for fast semantic image understanding. Produces 512-dimensional
 * embeddings that capture semantic meaning of images.
 *
 * MobileCLIP2-S2 benefits:
 * - 2.4x smaller than CLIP ViT-B/32 (143MB vs 345MB)
 * - 5-10x faster inference on mobile
 * - Similar accuracy to full CLIP
 *
 * For thesis: This demonstrates modern vision-language model deployment on mobile.
 *
 * @param context Application context for loading model from assets
 */
class CLIPFeatureExtractor(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false

    companion object {
        private const val TAG = "CLIPFeatureExtractor"

        // Model configuration - MobileCLIP2-S2
        private const val MODEL_FILE = "mobileclip2_s2_visual.onnx"
        const val INPUT_SIZE = 256 // MobileCLIP2-S2 uses 256x256
        const val EMBEDDING_SIZE = 512

        // MobileCLIP2-S2 uses NO normalization (mean=0, std=1)
        private val MEAN = floatArrayOf(0f, 0f, 0f)
        private val STD = floatArrayOf(1f, 1f, 1f)

        // ============================================
        // STATIC SERIALIZATION (no model needed)
        // ============================================

        /** Convert multiple embeddings to JSON for storage */
        fun multipleEmbeddingsToJson(embeddings: List<FloatArray>): String {
            val listOfLists = embeddings.map { it.toList() }
            return Json.encodeToString(listOfLists)
        }

        /** Parse multiple embeddings from JSON */
        fun multipleEmbeddingsFromJson(json: String): List<FloatArray> {
            return try {
                val listOfLists = Json.decodeFromString<List<List<Float>>>(json)
                listOfLists.map { it.toFloatArray() }
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing multiple embeddings: ${e.message}")
                try {
                    val single = Json.decodeFromString<List<Float>>(json).toFloatArray()
                    listOf(single)
                } catch (e2: Exception) {
                    emptyList()
                }
            }
        }

        /**
         * Serialize embeddings to binary blob for efficient storage. Format:
         * [count:Int32][embedding1:512xFloat32][embedding2:512xFloat32]...
         */
        fun embeddingsToBlob(embeddings: List<FloatArray>): ByteArray {
            val buffer = java.nio.ByteBuffer.allocate(4 + embeddings.size * EMBEDDING_SIZE * 4)
            buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(embeddings.size)
            for (embedding in embeddings) {
                for (value in embedding) {
                    buffer.putFloat(value)
                }
            }
            return buffer.array()
        }

        /** Parse embeddings from binary blob. */
        fun embeddingsFromBlob(blob: ByteArray): List<FloatArray> {
            return try {
                val buffer = java.nio.ByteBuffer.wrap(blob)
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val count = buffer.getInt()

                if (count <= 0 || count > 1000) {
                    Log.e(TAG, "Invalid embedding count in blob: $count")
                    return emptyList()
                }

                (0 until count).map { FloatArray(EMBEDDING_SIZE) { buffer.getFloat() } }
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing blob: ${e.message}")
                emptyList()
            }
        }

        /** Smart load: Prefer blob if available, fallback to JSON. */
        fun loadEmbeddings(blob: ByteArray?, json: String): List<FloatArray> {
            return if (blob != null && blob.isNotEmpty()) {
                val fromBlob = embeddingsFromBlob(blob)
                if (fromBlob.isNotEmpty()) fromBlob else multipleEmbeddingsFromJson(json)
            } else {
                multipleEmbeddingsFromJson(json)
            }
        }
    }

    init {
        loadModel()
    }

    /** Check if the CLIP model is loaded and ready */
    fun isReady(): Boolean = isInitialized && ortSession != null

    /** Copy model file from assets to internal storage */
    private fun copyModelToInternal(): File? {
        val outFile = File(context.filesDir, MODEL_FILE)

        if (!outFile.exists()) {
            try {
                Log.d(TAG, "Copying CLIP model to internal storage...")
                context.assets.open(MODEL_FILE).use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
                Log.d(TAG, "Copied CLIP model (${outFile.length() / 1024 / 1024}MB)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy model: ${e.message}")
                return null
            }
        }

        return outFile
    }

    /** Load CLIP ONNX model */
    private fun loadModel() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            val modelFile = copyModelToInternal()
            if (modelFile == null || !modelFile.exists()) {
                Log.e(TAG, "CLIP model file not found. Please add $MODEL_FILE to assets/")
                return
            }

            val sessionOptions =
                    OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                        setIntraOpNumThreads(4)
                        setInterOpNumThreads(2)
                    }

            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)

            if (ortSession != null) {
                isInitialized = true
                Log.d(TAG, "CLIP model loaded successfully")
                Log.d(TAG, "Input: ${ortSession!!.inputNames}")
                Log.d(TAG, "Output: ${ortSession!!.outputNames}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading CLIP model: ${e.message}", e)
            isInitialized = false
        }
    }

    /**
     * Preprocess bitmap for CLIP model.
     *
     * CLIP uses specific preprocessing:
     * 1. Resize to 224x224 (center crop if needed)
     * 2. Convert to RGB float [0, 1]
     * 3. Normalize with CLIP-specific mean/std
     * 4. Format as NCHW tensor
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        // Center crop and resize to 224x224
        val size = minOf(bitmap.width, bitmap.height)
        val xOffset = (bitmap.width - size) / 2
        val yOffset = (bitmap.height - size) / 2

        val cropped = Bitmap.createBitmap(bitmap, xOffset, yOffset, size, size)
        val resized = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)

        if (cropped != bitmap) cropped.recycle()

        // Allocate buffer for NCHW format: [1, 3, 224, 224]
        val floatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Convert to NCHW with CLIP normalization
        for (c in 0 until 3) {
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val pixel = pixels[y * INPUT_SIZE + x]
                    val value =
                            when (c) {
                                0 -> ((pixel shr 16) and 0xFF) / 255.0f // R
                                1 -> ((pixel shr 8) and 0xFF) / 255.0f // G
                                2 -> (pixel and 0xFF) / 255.0f // B
                                else -> 0f
                            }
                    val normalized = (value - MEAN[c]) / STD[c]
                    floatBuffer.put(normalized)
                }
            }
        }

        floatBuffer.rewind()

        if (resized != bitmap) resized.recycle()

        return floatBuffer
    }

    /**
     * Extract CLIP embedding from a bitmap.
     *
     * @param bitmap Input image (any size, will be center-cropped and resized)
     * @return 512-dimensional L2-normalized embedding, or null if failed
     */
    fun extractFeatures(bitmap: Bitmap): FloatArray? {
        if (!isInitialized || ortSession == null || ortEnvironment == null) {
            Log.e(TAG, "CLIP model not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Preprocess
            val inputBuffer = preprocessBitmap(bitmap)

            // Create input tensor
            val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, inputShape)

            // Get input name
            val inputName = ortSession!!.inputNames.first()

            // Run inference
            val results = ortSession!!.run(mapOf(inputName to inputTensor))

            val inferenceTime = System.currentTimeMillis() - startTime

            // Get output embedding
            val outputTensor = results.get(0) as? OnnxTensor
            inputTensor.close()

            if (outputTensor == null) {
                Log.e(TAG, "Output tensor is null")
                results.close()
                return null
            }

            // Extract embedding
            val embedding = outputTensor.floatBuffer
            val embeddingArray = FloatArray(EMBEDDING_SIZE)
            embedding.rewind()
            embedding.get(embeddingArray)

            results.close()

            // L2 normalize
            normalizeVector(embeddingArray)

            Log.d(TAG, "CLIP extraction: ${inferenceTime}ms")

            return embeddingArray
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting CLIP features: ${e.message}", e)
            return null
        }
    }

    /**
     * Batch feature extraction for multiple bitmaps. Uses dynamic batch for efficient parallel
     * inference (6x speedup for grid search).
     *
     * @param bitmaps List of input images (max 9 for memory safety)
     * @return List of 512D embeddings (same order as input), or null if failed
     */
    fun extractFeaturesBatch(bitmaps: List<Bitmap>): List<FloatArray>? {
        if (bitmaps.isEmpty() || bitmaps.size > 9) {
            Log.w(TAG, "Batch size must be 1-9, got ${bitmaps.size}")
            return null
        }
        if (!isInitialized || ortSession == null || ortEnvironment == null) {
            Log.e(TAG, "CLIP model not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()
            val batchSize = bitmaps.size

            // Allocate buffer for batch: [batchSize, 3, INPUT_SIZE, INPUT_SIZE]
            val floatBuffer = FloatBuffer.allocate(batchSize * 3 * INPUT_SIZE * INPUT_SIZE)

            // Preprocess all bitmaps into single buffer
            for (bitmap in bitmaps) {
                val singlePreprocessed = preprocessBitmap(bitmap)
                floatBuffer.put(singlePreprocessed)
            }
            floatBuffer.rewind()

            // Create batch tensor
            val inputShape =
                    longArrayOf(batchSize.toLong(), 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, inputShape)

            // Get input name
            val inputName = ortSession!!.inputNames.first()

            // Run inference
            val results = ortSession!!.run(mapOf(inputName to inputTensor))
            val inferenceTime = System.currentTimeMillis() - startTime

            val outputTensor = results.get(0) as? OnnxTensor
            inputTensor.close()

            if (outputTensor == null) {
                Log.e(TAG, "Batch output tensor is null")
                results.close()
                return null
            }

            // Extract all embeddings
            val buffer = outputTensor.floatBuffer
            buffer.rewind()

            val embeddings = mutableListOf<FloatArray>()
            for (i in 0 until batchSize) {
                val embedding = FloatArray(EMBEDDING_SIZE)
                buffer.get(embedding)
                normalizeVector(embedding)
                embeddings.add(embedding)
            }

            results.close()

            Log.d(
                    TAG,
                    "CLIP batch extraction: ${batchSize} images in ${inferenceTime}ms (${inferenceTime / batchSize}ms/img)"
            )

            return embeddings
        } catch (e: Exception) {
            Log.e(TAG, "Batch extraction failed: ${e.message}", e)
            // Fallback to sequential
            return bitmaps.mapNotNull { extractFeatures(it) }.takeIf { it.size == bitmaps.size }
        }
    }

    /** L2 normalize a vector in place */
    private fun normalizeVector(vector: FloatArray) {
        var norm = 0f
        for (v in vector) norm += v * v
        norm = sqrt(norm)

        if (norm > 0) {
            for (i in vector.indices) vector[i] /= norm
        }
    }

    /** Release resources */
    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            isInitialized = false
            Log.d(TAG, "CLIP extractor closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing CLIP session: ${e.message}")
        }
    }
}
