package com.florea_gabriel.impairedhelpapp.ml.processor

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import androidx.camera.core.ImageProxy

/**
 * ImageProcessor: Helper functions for image conversion and preprocessing
 */
object ImageProcessor {

    /**
     * Convert ImageProxy (YUV_420_888) to an upright Bitmap (ARGB).
     * Direct YUV→RGB conversion — no YuvImage/JPEG round-trip.
     * Honors imageProxy.cropRect: when the camera is bound with a shared
     * ViewPort, the returned bitmap contains exactly the region shown
     * by the preview (WYSIWYG).
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = yuv420ToBitmap(imageProxy, imageProxy.cropRect)
        return rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
    }

    /**
     * Direct YUV_420_888 → ARGB conversion honoring row/pixel strides.
     * BT.601 full-range coefficients (matches the previous JPEG path).
     */
    private fun yuv420ToBitmap(imageProxy: ImageProxy, crop: Rect): Bitmap {
        val width = crop.width()
        val height = crop.height()

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yBytes = ByteArray(yPlane.buffer.remaining()).also { yPlane.buffer.get(it) }
        val uBytes = ByteArray(uPlane.buffer.remaining()).also { uPlane.buffer.get(it) }
        val vBytes = ByteArray(vPlane.buffer.remaining()).also { vPlane.buffer.get(it) }

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val argb = IntArray(width * height)
        var outIdx = 0

        for (row in crop.top until crop.bottom) {
            val yRowOffset = row * yRowStride
            val uvRowOffset = (row shr 1) * uvRowStride

            for (col in crop.left until crop.right) {
                val y = yBytes[yRowOffset + col * yPixelStride].toInt() and 0xFF
                val uvOffset = uvRowOffset + (col shr 1) * uvPixelStride
                val u = (uBytes[uvOffset].toInt() and 0xFF) - 128
                val v = (vBytes[uvOffset].toInt() and 0xFF) - 128

                // Fixed-point BT.601: R = Y + 1.402V, G = Y - 0.344U - 0.714V, B = Y + 1.772U
                var r = y + ((1436 * v) shr 10)
                var g = y - ((352 * u + 731 * v) shr 10)
                var b = y + ((1815 * u) shr 10)

                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255

                argb[outIdx++] = -0x1000000 or (r shl 16) or (g shl 8) or b
            }
        }

        return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Rotate bitmap by degrees
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap

        val matrix = Matrix().apply {
            postRotate(degrees)
        }

        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }
}
