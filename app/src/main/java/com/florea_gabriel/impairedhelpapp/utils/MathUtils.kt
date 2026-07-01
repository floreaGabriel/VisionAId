package com.florea_gabriel.impairedhelpapp.utils

import kotlin.math.sqrt

/**
 * Cosine similarity between two vectors.
 * Works with both normalized and un-normalized vectors.
 */
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f

    var dot = 0f
    var normA = 0f
    var normB = 0f

    for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }

    val denom = sqrt(normA) * sqrt(normB)
    return if (denom > 0f) dot / denom else 0f
}
