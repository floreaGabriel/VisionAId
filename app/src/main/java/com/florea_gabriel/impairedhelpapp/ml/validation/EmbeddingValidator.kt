package com.florea_gabriel.impairedhelpapp.ml.validation

import android.util.Log
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * EmbeddingValidator: Validates quality of embeddings after registration.
 * 
 * This helps detect problems BEFORE the user tries to search:
 * - Too few embeddings
 * - Embeddings too similar (not enough variety)
 * - Embeddings too diverse (object not consistent)
 * - Poor retrieval potential
 * 
 * Based on research insights:
 * - ProbaCLIP: Using distribution analysis for better matching
 * - CLIP retrieval papers: Optimal similarity thresholds
 */
class EmbeddingValidator {

    companion object {
        private const val TAG = "EmbeddingValidator"
        
        // Minimum requirements
        private const val MIN_EMBEDDINGS = 50
        private const val RECOMMENDED_EMBEDDINGS = 100
        
        // Quality thresholds
        private const val MIN_INTRA_SIMILARITY = 0.60f   // Embeddings should be at least 60% similar
        private const val MAX_INTRA_SIMILARITY = 0.95f   // But not too similar (need variety)
        private const val IDEAL_INTRA_SIMILARITY = 0.75f // Sweet spot
        
        // Variance thresholds
        private const val LOW_VARIANCE = 0.01f    // Very uniform object
        private const val HIGH_VARIANCE = 0.05f   // Very diverse object
        
        // Retrieval simulation
        private const val MIN_RETRIEVAL_SUCCESS = 0.70f  // At least 70% self-retrieval
    }

    /**
     * Result of embedding validation
     */
    data class ValidationResult(
        val isValid: Boolean,
        val quality: Float,              // 0-1 overall quality score
        val intraClassSimilarity: Float, // How consistent embeddings are (0-1)
        val variance: Float,             // Variance in similarities
        val retrievalScore: Float,       // Self-retrieval success rate (0-1)
        val optimalThreshold: Float,     // Recommended threshold for search
        val recommendations: List<String>,
        val details: ValidationDetails
    )
    
    /**
     * Detailed statistics for debugging
     */
    data class ValidationDetails(
        val embeddingCount: Int,
        val meanSimilarity: Float,
        val minSimilarity: Float,
        val maxSimilarity: Float,
        val stdDeviation: Float,
        val percentile10: Float,
        val percentile90: Float
    )

    /**
     * Validate embeddings quality after registration.
     * 
     * @param embeddings List of CLIP embeddings (512D each)
     * @return ValidationResult with quality metrics and recommendations
     */
    fun validate(embeddings: List<FloatArray>): ValidationResult {
        Log.d(TAG, "Validating ${embeddings.size} embeddings...")
        
        // Check minimum count
        if (embeddings.size < MIN_EMBEDDINGS) {
            return createFailedResult(
                embeddings.size,
                "Prea puține embeddings (${embeddings.size}). Minim necesar: $MIN_EMBEDDINGS. " +
                "Încearcă din nou și mișcă obiectul mai încet."
            )
        }

        // Calculate pairwise similarities (sample for efficiency)
        val similarities = calculatePairwiseSimilarities(embeddings)
        
        if (similarities.isEmpty()) {
            return createFailedResult(embeddings.size, "Nu s-au putut calcula similaritățile.")
        }

        // Statistics
        val meanSim = similarities.average().toFloat()
        val variance = similarities.map { (it - meanSim).pow(2) }.average().toFloat()
        val stdDev = sqrt(variance.toDouble()).toFloat()
        val sortedSims = similarities.sorted()
        val minSim = sortedSims.first()
        val maxSim = sortedSims.last()
        val p10 = sortedSims[(sortedSims.size * 0.1).toInt()]
        val p90 = sortedSims[(sortedSims.size * 0.9).toInt()]

        Log.d(TAG, "Stats: mean=${"%.3f".format(meanSim)}, std=${"%.3f".format(stdDev)}, " +
                "range=[${"%.3f".format(minSim)}, ${"%.3f".format(maxSim)}]")

        // Simulate retrieval
        val retrievalScore = simulateRetrieval(embeddings, meanSim)
        
        // Generate recommendations
        val recommendations = mutableListOf<String>()
        var qualityScore = 1.0f

        // Check intra-class similarity
        when {
            meanSim < MIN_INTRA_SIMILARITY -> {
                recommendations.add("⚠️ Embeddings prea diverse (media: ${"%.0f".format(meanSim * 100)}%). " +
                        "Obiectul poate arăta foarte diferit din unghiuri diferite sau fundalul a contaminat înregistrarea.")
                qualityScore *= 0.5f
            }
            meanSim > MAX_INTRA_SIMILARITY -> {
                recommendations.add("⚠️ Embeddings prea similare (media: ${"%.0f".format(meanSim * 100)}%). " +
                        "Încearcă să rotești obiectul mai mult pentru varietate.")
                qualityScore *= 0.7f
            }
            meanSim >= 0.70f && meanSim <= 0.85f -> {
                recommendations.add("✅ Consistență bună (${"%.0f".format(meanSim * 100)}%)")
            }
        }

        // Check variance
        when {
            variance < LOW_VARIANCE -> {
                recommendations.add("ℹ️ Variabilitate scăzută - obiect uniform. OK pentru obiecte simple.")
            }
            variance > HIGH_VARIANCE -> {
                recommendations.add("⚠️ Variabilitate mare (${"%.3f".format(variance)}). " +
                        "Obiectul arată foarte diferit din unghiuri diferite. Poate fi greu de găsit.")
                qualityScore *= 0.8f
            }
        }

        // Check retrieval potential
        if (retrievalScore < MIN_RETRIEVAL_SUCCESS) {
            recommendations.add("⚠️ Rată retrieval scăzută (${"%.0f".format(retrievalScore * 100)}%). " +
                    "Obiectul poate fi greu de regăsit. Încearcă să-l înregistrezi pe un fundal diferit.")
            qualityScore *= 0.6f
        } else {
            recommendations.add("✅ Retrieval bun (${"%.0f".format(retrievalScore * 100)}%)")
        }

        // Check embedding count
        if (embeddings.size < RECOMMENDED_EMBEDDINGS) {
            recommendations.add("ℹ️ ${embeddings.size} embeddings. Recomandat: $RECOMMENDED_EMBEDDINGS+ pentru precizie maximă.")
        } else {
            recommendations.add("✅ ${embeddings.size} embeddings colectate")
        }

        // Calculate optimal threshold based on distribution
        // Use percentile 10 as threshold - 90% of embeddings should match above this
        val optimalThreshold = calculateOptimalThreshold(meanSim, stdDev, p10)

        val details = ValidationDetails(
            embeddingCount = embeddings.size,
            meanSimilarity = meanSim,
            minSimilarity = minSim,
            maxSimilarity = maxSim,
            stdDeviation = stdDev,
            percentile10 = p10,
            percentile90 = p90
        )

        val isValid = qualityScore >= 0.5f && retrievalScore >= 0.5f

        Log.d(TAG, "Validation complete: valid=$isValid, quality=${"%.2f".format(qualityScore)}, " +
                "threshold=${"%.3f".format(optimalThreshold)}")

        return ValidationResult(
            isValid = isValid,
            quality = qualityScore,
            intraClassSimilarity = meanSim,
            variance = variance,
            retrievalScore = retrievalScore,
            optimalThreshold = optimalThreshold,
            recommendations = recommendations,
            details = details
        )
    }

    /**
     * Calculate pairwise similarities between embeddings.
     * Samples for efficiency if too many embeddings.
     */
    private fun calculatePairwiseSimilarities(embeddings: List<FloatArray>): List<Float> {
        val similarities = mutableListOf<Float>()
        
        // Sample strategy: if too many, sample randomly
        val sampleSize = minOf(embeddings.size, 50)
        val sampledIndices = if (embeddings.size <= 50) {
            embeddings.indices.toList()
        } else {
            embeddings.indices.shuffled().take(sampleSize)
        }

        for (i in sampledIndices.indices) {
            for (j in i + 1 until sampledIndices.size) {
                val sim = cosineSimilarity(
                    embeddings[sampledIndices[i]], 
                    embeddings[sampledIndices[j]]
                )
                similarities.add(sim)
            }
        }

        return similarities
    }

    /**
     * Simulate retrieval: take random embeddings as queries and check if they find matches.
     */
    private fun simulateRetrieval(embeddings: List<FloatArray>, meanSim: Float): Float {
        if (embeddings.size < 10) return 0f

        var successCount = 0
        val testCount = minOf(20, embeddings.size / 5)
        val threshold = meanSim - 0.15f  // Use mean - 0.15 as threshold

        repeat(testCount) {
            val testIdx = (embeddings.indices).random()
            val query = embeddings[testIdx]
            
            // Find best match excluding self
            var bestSim = 0f
            for (i in embeddings.indices) {
                if (i == testIdx) continue
                val sim = cosineSimilarity(query, embeddings[i])
                if (sim > bestSim) bestSim = sim
            }

            // Success if best match is above threshold
            if (bestSim >= threshold) successCount++
        }

        return successCount / testCount.toFloat()
    }

    /**
     * Calculate optimal threshold for this object based on its distribution.
     * 
     * Strategy: Use percentile10 as base, but not lower than 0.55 or higher than 0.75
     */
    private fun calculateOptimalThreshold(mean: Float, stdDev: Float, p10: Float): Float {
        // Method 1: percentile-based (conservative)
        val percentileThreshold = p10 - 0.05f
        
        // Method 2: mean - 2*stdDev (statistical)
        val statisticalThreshold = mean - 2 * stdDev
        
        // Use the higher of the two, clamped to reasonable range
        val optimal = maxOf(percentileThreshold, statisticalThreshold)
        
        return optimal.coerceIn(0.55f, 0.75f)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float =
        com.florea_gabriel.impairedhelpapp.utils.cosineSimilarity(a, b)

    /**
     * Create a failed validation result
     */
    private fun createFailedResult(count: Int, message: String): ValidationResult {
        return ValidationResult(
            isValid = false,
            quality = 0f,
            intraClassSimilarity = 0f,
            variance = 0f,
            retrievalScore = 0f,
            optimalThreshold = 0.65f,
            recommendations = listOf("❌ $message"),
            details = ValidationDetails(
                embeddingCount = count,
                meanSimilarity = 0f,
                minSimilarity = 0f,
                maxSimilarity = 0f,
                stdDeviation = 0f,
                percentile10 = 0f,
                percentile90 = 0f
            )
        )
    }

    /**
     * Quick validation - just check if embeddings meet minimum requirements.
     * Use this for fast feedback during registration.
     */
    fun quickCheck(embeddings: List<FloatArray>): QuickCheckResult {
        if (embeddings.size < 20) {
            return QuickCheckResult(
                isOk = false,
                message = "Colectare în progres... (${embeddings.size}/$MIN_EMBEDDINGS)"
            )
        }

        // Quick sample check
        val sampleSize = minOf(10, embeddings.size)
        val sample = embeddings.shuffled().take(sampleSize)
        
        var totalSim = 0f
        var count = 0
        for (i in sample.indices) {
            for (j in i + 1 until sample.size) {
                totalSim += cosineSimilarity(sample[i], sample[j])
                count++
            }
        }
        
        val avgSim = if (count > 0) totalSim / count else 0f
        
        return when {
            avgSim < 0.50f -> QuickCheckResult(
                isOk = false,
                message = "⚠️ Calitate scăzută - ține obiectul mai stabil"
            )
            avgSim > 0.95f -> QuickCheckResult(
                isOk = false,
                message = "⚠️ Mișcă obiectul pentru varietate"
            )
            else -> QuickCheckResult(
                isOk = true,
                message = "✅ Calitate bună (${embeddings.size} frames)"
            )
        }
    }

    data class QuickCheckResult(
        val isOk: Boolean,
        val message: String
    )
}
