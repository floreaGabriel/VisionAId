package com.florea_gabriel.impairedhelpapp.data.model

/**
 * Data class representing depth estimation result
 *
 * @property depthMap 2D array of normalized depth values (0.0-1.0) for visualization
 * @property metricDepthMap 2D array of metric depth values in METERS (raw from model)
 * @property width Width of the depth map
 * @property height Height of the depth map
 * @property inferenceTime Time taken for inference in milliseconds
 */
data class DepthResult(
    val depthMap: Array<FloatArray>,           // Normalized [0-1] for visualization
    val metricDepthMap: Array<FloatArray>,     // Raw metric values in METERS
    val width: Int,
    val height: Int,
    val inferenceTime: Long
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DepthResult

        if (!depthMap.contentDeepEquals(other.depthMap)) return false
        if (!metricDepthMap.contentDeepEquals(other.metricDepthMap)) return false
        if (width != other.width) return false
        if (height != other.height) return false

        return true
    }

    override fun hashCode(): Int {
        var result = depthMap.contentDeepHashCode()
        result = 31 * result + metricDepthMap.contentDeepHashCode()
        result = 31 * result + width
        result = 31 * result + height
        return result
    }
}
