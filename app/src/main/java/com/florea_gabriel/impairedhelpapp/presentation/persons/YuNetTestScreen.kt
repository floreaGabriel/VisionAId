package com.florea_gabriel.impairedhelpapp.presentation.persons

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.florea_gabriel.impairedhelpapp.ml.face.FaceDetector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "YuNetTest"

@Composable
fun YuNetTestScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val isActive = remember { AtomicBoolean(true) }
    val isProcessing = remember { AtomicBoolean(false) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val mlScope = remember { CoroutineScope(Dispatchers.Default + SupervisorJob()) }

    var faceDetector by remember { mutableStateOf<FaceDetector?>(null) }
    var detections by remember { mutableStateOf<List<FaceDetector.FaceDetection>>(emptyList()) }
    var previewWidth by remember { mutableIntStateOf(1) }
    var previewHeight by remember { mutableIntStateOf(1) }
    var debugInfo by remember { mutableStateOf("Loading...") }
    var frameCount by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            faceDetector = FaceDetector(context)
        }
        debugInfo = if (faceDetector?.isReady() == true) "YuNet ready" else "YuNet FAILED"
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                }

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor) analyzer@{ imageProxy ->
                        if (!isActive.get() || !isProcessing.compareAndSet(false, true)) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val detector = faceDetector
                        if (detector == null || !detector.isReady()) {
                            isProcessing.set(false)
                            imageProxy.close()
                            return@analyzer
                        }

                        // Log raw ImageProxy info
                        val rotation = imageProxy.imageInfo.rotationDegrees
                        val proxyW = imageProxy.width
                        val proxyH = imageProxy.height

                        val bitmap: Bitmap
                        try {
                            bitmap = imageProxyToBitmapDirect(imageProxy)
                        } catch (e: Exception) {
                            Log.e(TAG, "Bitmap conversion failed: ${e.message}")
                            isProcessing.set(false)
                            imageProxy.close()
                            return@analyzer
                        }
                        imageProxy.close()

                        mlScope.launch {
                            try {
                                val bmpW = bitmap.width
                                val bmpH = bitmap.height

                                // Log first pixel to check channel order
                                val firstPixel = IntArray(1)
                                bitmap.getPixels(firstPixel, 0, 1, bmpW / 2, bmpH / 2, 1, 1)
                                val px = firstPixel[0]
                                val a = (px shr 24) and 0xFF
                                val r = (px shr 16) and 0xFF
                                val g = (px shr 8) and 0xFF
                                val b = px and 0xFF

                                Log.d(TAG, "FRAME: proxy=${proxyW}x${proxyH} rot=$rotation -> bitmap=${bmpW}x${bmpH}, center_pixel: A=$a R=$r G=$g B=$b")

                                val dets = detector.detectFaces(bitmap)

                                withContext(Dispatchers.Main) {
                                    previewWidth = bmpW
                                    previewHeight = bmpH
                                    detections = dets
                                    frameCount++
                                    debugInfo = "proxy=${proxyW}x${proxyH} rot=$rotation\n" +
                                            "bitmap=${bmpW}x${bmpH}\n" +
                                            "faces=${dets.size}, frame=$frameCount\n" +
                                            dets.joinToString("\n") { d ->
                                                "  [${d.box[0].toInt()},${d.box[1].toInt()},${d.box[2].toInt()},${d.box[3].toInt()}] " +
                                                "w=${(d.box[2]-d.box[0]).toInt()} h=${(d.box[3]-d.box[1]).toInt()} " +
                                                "c=${"%.2f".format(d.confidence)}"
                                            }
                                }
                                bitmap.recycle()
                            } catch (e: Exception) {
                                Log.e(TAG, "Detection error: ${e.message}")
                                bitmap.recycle()
                            } finally {
                                isProcessing.set(false)
                            }
                        }
                    }

                    if (!isActive.get()) return@addListener

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera bind error: ${e.message}")
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Draw bounding boxes + landmarks
        if (detections.isNotEmpty()) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                // FILL_CENTER mapping: uniform scale to fill both dimensions, then crop
                val fillScale = maxOf(size.width / previewWidth, size.height / previewHeight)
                val offsetX = (previewWidth * fillScale - size.width) / 2f
                val offsetY = (previewHeight * fillScale - size.height) / 2f

                fun mapX(imgX: Float) = imgX * fillScale - offsetX
                fun mapY(imgY: Float) = imgY * fillScale - offsetY

                for (det in detections) {
                    val box = det.box
                    val left = mapX(box[0])
                    val top = mapY(box[1])
                    val right = mapX(box[2])
                    val bottom = mapY(box[3])

                    // Green box
                    drawRect(
                        color = Color(0xFF4CAF50),
                        topLeft = Offset(left, top),
                        size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                        style = Stroke(width = 3f)
                    )

                    // Confidence label
                    drawContext.canvas.nativeCanvas.drawText(
                        "${"%.2f".format(det.confidence)}",
                        left,
                        top - 10f,
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.GREEN
                            textSize = 40f
                            isAntiAlias = true
                        }
                    )

                    // Landmarks (5 colored dots)
                    val lmColors = intArrayOf(
                        android.graphics.Color.RED,
                        android.graphics.Color.BLUE,
                        android.graphics.Color.YELLOW,
                        android.graphics.Color.MAGENTA,
                        android.graphics.Color.CYAN
                    )
                    val lm = det.landmarks
                    if (lm != null && lm.size >= 10) {
                        for (j in 0 until 5) {
                            val lx = mapX(lm[2 * j])
                            val ly = mapY(lm[2 * j + 1])
                            drawCircle(
                                color = Color(lmColors[j]),
                                radius = 6f,
                                center = Offset(lx, ly)
                            )
                        }
                    }
                }
            }
        }

        // Debug info overlay
        Text(
            text = debugInfo,
            color = Color.White,
            fontSize = 14.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(top = 60.dp, start = 16.dp)
                .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(8.dp))
                .padding(12.dp)
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            isActive.set(false)
            cameraExecutor.shutdown()
            cameraExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)
            mlScope.cancel()
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Exception) {}
            val det = faceDetector
            Thread {
                Thread.sleep(500)
                det?.close()
            }.start()
        }
    }
}

// Direct bitmap conversion WITHOUT going through ImageProcessor
// This lets us log every step
private fun imageProxyToBitmapDirect(imageProxy: ImageProxy): Bitmap {
    val yBuffer = imageProxy.planes[0].buffer
    val uBuffer = imageProxy.planes[1].buffer
    val vBuffer = imageProxy.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
    val imageBytes = out.toByteArray()
    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

    val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
    Log.d(TAG, "imageProxyToBitmap: proxy=${imageProxy.width}x${imageProxy.height}, rotation=$rotation, bitmap=${bitmap.width}x${bitmap.height}")

    if (rotation == 0f) return bitmap

    val matrix = Matrix().apply { postRotate(rotation) }
    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    if (rotated != bitmap) bitmap.recycle()

    Log.d(TAG, "After rotation: ${rotated.width}x${rotated.height}")
    return rotated
}
