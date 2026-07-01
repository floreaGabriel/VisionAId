package com.florea_gabriel.impairedhelpapp.ml.detector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.florea_gabriel.impairedhelpapp.data.model.DepthResult
import com.florea_gabriel.impairedhelpapp.utils.Constants
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * DepthEstimator: Wrapper for Depth Anything V2 Metric ONNX model
 *
 * Estimates monocular depth from a single RGB image using ONNX Runtime.
 * Supports NNAPI hardware acceleration with automatic CPU fallback.
 *
 * Model: Depth Anything V2 Metric Small
 * Input: Defined in Constants (currently 224x224)
 * Output: Metric depth map in meters
 */
class DepthEstimator(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val inputSize = Constants.DEPTH_INPUT_SIZE
    private val outputSize = Constants.DEPTH_OUTPUT_SIZE

    companion object {
        private const val TAG = "DepthEstimator"
        private const val MODEL_FILE = "depth_anything_v2_metric_small_int8.onnx"

        // ImageNet normalization constants
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    init {
        loadModel()
    }

    /**
     * Configure optimized CPU backend with multi-threading
     */
    private fun configureCpuBackend(options: OrtSession.SessionOptions) {
        val threads = Constants.NUM_THREADS
        options.setIntraOpNumThreads(threads)
        options.setInterOpNumThreads(2)
        Log.d(TAG, "CPU backend configured: $threads threads")
    }

    /**
     * Configure session options with optimized CPU backend
     * NNAPI disabled due to poor performance on some devices (e.g. Exynos 8s inference vs 1.2s CPU)
     */
    private fun configureSessionOptions(options: OrtSession.SessionOptions) {
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        configureCpuBackend(options)
    }

    /**
     * Copy asset file to internal storage if it doesn't exist
     */
    private fun copyAssetToInternal(assetName: String): File {
        val outFile = File(context.filesDir, assetName)

        if (!outFile.exists()) {
            Log.d(TAG, "Copying $assetName to internal storage...")
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "✅ Copied $assetName (${outFile.length()} bytes)")
        } else {
            Log.d(TAG, "✅ $assetName already exists in internal storage")
        }

        return outFile
    }

    /**
     * Load ONNX depth model from assets
     *
     * The model uses external data (.onnx.data file), so we copy
     * both files to internal storage and load from there.
     */
    private fun loadModel() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Copy model to internal storage (INT8: single file, no .data needed)
            val modelFile = copyAssetToInternal(MODEL_FILE)

            Log.d(TAG, "Depth model loading with ${Constants.NUM_THREADS} threads")

            // Create optimized session
            val sessionOptions = OrtSession.SessionOptions().apply {
                configureSessionOptions(this)
            }

            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)

            if (ortSession != null) {
                Log.d(TAG, "Model loaded successfully")
            } else {
                Log.e(TAG, "Failed to load model")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
        }
    }

    /**
     * Preprocess bitmap for Depth Anything V2 model
     *
     * Converts bitmap to normalized float tensor with shape [1, 3, H, W]
     * Using ImageNet normalization: (pixel / 255 - mean) / std
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Allocate buffer for NCHW format: [1, 3, 518, 518]
        val floatBuffer = FloatBuffer.allocate(1 * 3 * inputSize * inputSize)

        // Extract pixels
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to NCHW format with ImageNet normalization
        // Channel order: R, G, B
        for (c in 0 until 3) {
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val pixel = pixels[y * inputSize + x]
                    val value = when (c) {
                        0 -> ((pixel shr 16) and 0xFF) / 255.0f  // R
                        1 -> ((pixel shr 8) and 0xFF) / 255.0f   // G
                        2 -> (pixel and 0xFF) / 255.0f           // B
                        else -> 0f
                    }
                    // Apply ImageNet normalization
                    val normalized = (value - MEAN[c]) / STD[c]
                    floatBuffer.put(normalized)
                }
            }
        }

        floatBuffer.rewind()

        // Clean up resized bitmap if it's different from original
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        return floatBuffer
    }

    /**
     * Estimate depth from a bitmap
     * @param bitmap Input RGB image
     * @return DepthResult containing depth map with metric depth values
     */
    fun estimateDepth(bitmap: Bitmap): DepthResult? {
        if (ortSession == null || ortEnvironment == null) {
            Log.e(TAG, "❌ ONNX session not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Preprocess image
            val inputBuffer = preprocessBitmap(bitmap)

            // Create input tensor with shape [1, 3, 518, 518]
            val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, inputShape)

            // Get input name from model
            val inputName = ortSession!!.inputNames.first()

            // Run inference
            val results = ortSession!!.run(mapOf(inputName to inputTensor))

            val inferenceTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "ONNX Depth inference time: ${inferenceTime}ms")

            // Parse output - get first output tensor
            val outputTensor = results.get(0) as? OnnxTensor

            // Close input tensor
            inputTensor.close()

            if (outputTensor == null) {
                Log.e(TAG, "❌ Output tensor is null")
                results.close()
                return null
            }

            // Get the tensor data as FloatBuffer
            val outputBuffer = outputTensor.floatBuffer
            val parseResult = parseDepthBuffer(outputBuffer)

            results.close()

            if (parseResult == null) {
                return null
            }

            return DepthResult(
                depthMap = parseResult.normalizedMap,
                metricDepthMap = parseResult.metricMap,
                width = outputSize,
                height = outputSize,
                inferenceTime = inferenceTime
            )

        } catch (e: Exception) {
            Log.e(TAG, "❌ Depth estimation error: ${e.message}", e)
            return null
        }
    }

    /**
     * Result from parsing depth buffer
     */
    private data class DepthParseResult(
        val metricMap: Array<FloatArray>,      // Raw metric values in meters
        val normalizedMap: Array<FloatArray>   // Normalized [0-1] for visualization
    )

    /**
     * Parse FloatBuffer output to 2D depth map
     *
     * The metric model outputs depth values directly in meters
     * Output shape is [1, 378, 378]
     */
    private fun parseDepthBuffer(buffer: java.nio.FloatBuffer): DepthParseResult? {
        try {
            val metricMap = Array(outputSize) { FloatArray(outputSize) }

            buffer.rewind()

            // Read depth values from buffer (raw metric values in meters)
            var minDepth = Float.MAX_VALUE
            var maxDepth = Float.MIN_VALUE

            for (y in 0 until outputSize) {
                for (x in 0 until outputSize) {
                    val value = if (buffer.hasRemaining()) buffer.get() else 0f
                    metricMap[y][x] = value
                    if (value < minDepth) minDepth = value
                    if (value > maxDepth) maxDepth = value
                }
            }

            Log.d(TAG, "Metric depth range: [${String.format("%.2f", minDepth)}m, ${String.format("%.2f", maxDepth)}m]")

            // Normalize depth values to [0, 1] range for visualization
            val normalizedMap = normalizeDepthMap(metricMap, minDepth, maxDepth)

            return DepthParseResult(
                metricMap = metricMap,
                normalizedMap = normalizedMap
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing depth buffer: ${e.message}", e)
            return null
        }
    }

    /**
     * Normalize depth map to [0, 1] range
     *
     * For metric depth models:
     * - Lower values = closer objects -> normalized to higher values (closer to 1)
     * - Higher values = farther objects -> normalized to lower values (closer to 0)
     *
     * This inverts the depth so that:
     * - 0.0 = far
     * - 1.0 = close
     */
    private fun normalizeDepthMap(depthMap: Array<FloatArray>, minDepth: Float, maxDepth: Float): Array<FloatArray> {
        val normalizedMap = Array(outputSize) { FloatArray(outputSize) }
        val range = maxDepth - minDepth

        if (range <= 0.001f) {
            // Uniform depth, return 0.5
            for (y in 0 until outputSize) {
                for (x in 0 until outputSize) {
                    normalizedMap[y][x] = 0.5f
                }
            }
            return normalizedMap
        }

        for (y in 0 until outputSize) {
            for (x in 0 until outputSize) {
                val metricDepth = depthMap[y][x]
                // Normalize to [0, 1] and invert so that close = high value
                val normalized = 1.0f - ((metricDepth - minDepth) / range).coerceIn(0f, 1f)
                normalizedMap[y][x] = normalized
            }
        }

        return normalizedMap
    }

    /**
     * Release resources
     */
    fun close() {
        try {
            ortSession?.close()
            ortSession = null
            // Note: OrtEnvironment is a singleton and shouldn't be closed
            Log.d(TAG, "ONNX Depth estimator closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX session: ${e.message}")
        }
    }
}
