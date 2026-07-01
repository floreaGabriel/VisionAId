package com.florea_gabriel.impairedhelpapp.data.model

import android.graphics.RectF

/**
 * Data class representing a detected object with depth information
 *
 * @property label The class name of the detected object
 * @property confidence Confidence score (0.0-1.0) indicating model certainty
 * @property boundingBox The rectangular region containing the detected object
 * @property distance Distance to the object in meters (null if depth not available)
 */
data class Detection(
    val label: String,
    val confidence: Float,
    val boundingBox: RectF,
    val distance: Float? = null
)
