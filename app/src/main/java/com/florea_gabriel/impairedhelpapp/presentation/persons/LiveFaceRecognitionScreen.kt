package com.florea_gabriel.impairedhelpapp.presentation.persons

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.florea_gabriel.impairedhelpapp.data.database.KnownPerson
import com.florea_gabriel.impairedhelpapp.ml.face.FaceDetector
import com.florea_gabriel.impairedhelpapp.ml.face.FaceEmbedder
import com.florea_gabriel.impairedhelpapp.ml.processor.ImageProcessor
import com.florea_gabriel.impairedhelpapp.presentation.viewmodel.PersonsViewModel
import com.florea_gabriel.impairedhelpapp.ui.theme.Emerald
import com.florea_gabriel.impairedhelpapp.ui.theme.Indigo
import kotlinx.coroutines.*
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "LiveFaceRecognition"

private data class TrackedFace(
    val id: Int,
    val box: FloatArray,
    val name: String?,
    val similarity: Float,
    val framesSinceEmbed: Int
)

private data class CachedPerson(
    val person: KnownPerson,
    val embeddings: List<FloatArray>
)

@Composable
fun LiveFaceRecognitionScreen(
    isScreenVisible: Boolean,
    personsViewModel: PersonsViewModel,
    bottomPadding: Dp,
    onSpeakRequest: (String) -> Unit,
    onShowPersonsList: () -> Unit,
    onStartRegistration: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val isActive = remember { AtomicBoolean(true) }
    val isProcessing = remember { AtomicBoolean(false) }
    val currentIsVisible by rememberUpdatedState(isScreenVisible)
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Dedicated ML scope on Default dispatcher (background threads)
    val mlScope = remember { CoroutineScope(Dispatchers.Default + SupervisorJob()) }

    // ML models
    var faceDetector by remember { mutableStateOf<FaceDetector?>(null) }
    var faceEmbedder by remember { mutableStateOf<FaceEmbedder?>(null) }

    // Detection state
    var trackedFaces by remember { mutableStateOf<List<TrackedFace>>(emptyList()) }
    var previewWidth by remember { mutableIntStateOf(1) }
    var previewHeight by remember { mutableIntStateOf(1) }
    var nextFaceId by remember { mutableIntStateOf(0) }

    // Known persons cache
    val allPersons by personsViewModel.persons.collectAsState()
    var cachedPersons by remember { mutableStateOf<List<CachedPerson>>(emptyList()) }

    // Cache known persons embeddings when DB changes
    LaunchedEffect(allPersons) {
        val embedder = faceEmbedder ?: return@LaunchedEffect
        cachedPersons = allPersons.map { person ->
            CachedPerson(
                person = person,
                embeddings = embedder.embeddingsFromBlob(person.embeddingBlob)
            )
        }
        Log.d(TAG, "Cached ${cachedPersons.size} known persons")
    }

    // Initialize models
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            faceDetector = FaceDetector(context)
            faceEmbedder = FaceEmbedder(context)
        }
        val embedder = faceEmbedder ?: return@LaunchedEffect
        cachedPersons = allPersons.map { person ->
            CachedPerson(
                person = person,
                embeddings = embedder.embeddingsFromBlob(person.embeddingBlob)
            )
        }
        Log.d(TAG, "Models loaded, cached ${cachedPersons.size} persons")
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview
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
                        // Gate: skip if not active, not visible, or already processing
                        if (!isActive.get() || !currentIsVisible || !isProcessing.compareAndSet(false, true)) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val detector = faceDetector
                        val embedder = faceEmbedder
                        if (detector == null || !detector.isReady() || embedder == null || !embedder.isReady()) {
                            isProcessing.set(false)
                            imageProxy.close()
                            return@analyzer
                        }

                        // Convert to bitmap immediately, then release imageProxy
                        val bitmap: Bitmap
                        try {
                            bitmap = ImageProcessor.imageProxyToBitmap(imageProxy)
                        } catch (e: Exception) {
                            isProcessing.set(false)
                            imageProxy.close()
                            return@analyzer
                        }
                        imageProxy.close() // Release camera frame ASAP

                        // Run ML on background scope
                        mlScope.launch {
                            try {
                                val pw = bitmap.width
                                val ph = bitmap.height

                                // Face detection (~50-80ms)
                                val detections = detector.detectFaces(bitmap)
                                Log.d(TAG, "Detected ${detections.size} faces after NMS")
                                val prevTracked = trackedFaces
                                val newTracked = mutableListOf<TrackedFace>()
                                val matched = BooleanArray(prevTracked.size)

                                for (det in detections) {
                                    var bestIdx = -1
                                    var bestIou = 0.4f
                                    for (i in prevTracked.indices) {
                                        if (matched[i]) continue
                                        val iouVal = iou(det.box, prevTracked[i].box)
                                        if (iouVal > bestIou) {
                                            bestIou = iouVal
                                            bestIdx = i
                                        }
                                    }

                                    if (bestIdx >= 0) {
                                        matched[bestIdx] = true
                                        val prev = prevTracked[bestIdx]
                                        val frames = prev.framesSinceEmbed + 1

                                        if (frames > 5) {
                                            // Stale → re-embed (async, don't block detection)
                                            val result = embedAndMatch(bitmap, det.box, embedder, cachedPersons)
                                            newTracked.add(
                                                TrackedFace(prev.id, det.box, result?.first, result?.second ?: 0f, 0)
                                            )
                                        } else {
                                            // Carry identity, just update box
                                            newTracked.add(prev.copy(box = det.box, framesSinceEmbed = frames))
                                        }
                                    } else {
                                        // New face → embed immediately
                                        val result = embedAndMatch(bitmap, det.box, embedder, cachedPersons)
                                        newTracked.add(
                                            TrackedFace(nextFaceId++, det.box, result?.first, result?.second ?: 0f, 0)
                                        )
                                    }
                                }

                                // Update UI state on Main
                                withContext(Dispatchers.Main) {
                                    previewWidth = pw
                                    previewHeight = ph
                                    trackedFaces = newTracked
                                }
                                bitmap.recycle()
                            } catch (e: Exception) {
                                Log.e(TAG, "Analysis error: ${e.message}")
                            } finally {
                                isProcessing.set(false)
                            }
                        }
                    }

                    // Guard: don't bind if screen was already disposed during async provider fetch
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

        // Face overlay
        if (trackedFaces.isNotEmpty()) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                // FILL_CENTER mapping: uniform scale to fill both dimensions, then crop
                val fillScale = maxOf(size.width / previewWidth, size.height / previewHeight)
                val offsetX = (previewWidth * fillScale - size.width) / 2f
                val offsetY = (previewHeight * fillScale - size.height) / 2f

                fun mapX(imgX: Float) = imgX * fillScale - offsetX
                fun mapY(imgY: Float) = imgY * fillScale - offsetY

                for (face in trackedFaces) {
                    val box = face.box
                    val left = mapX(box[0])
                    val top = mapY(box[1])
                    val right = mapX(box[2])
                    val bottom = mapY(box[3])
                    val w = right - left
                    val h = bottom - top

                    val isKnown = face.name != null
                    val boxColor = if (isKnown) Color(0xFF4CAF50) else Color(0xFFF44336)
                    val label = face.name ?: "Necunoscut"

                    drawRect(
                        color = boxColor,
                        topLeft = Offset(left, top),
                        size = androidx.compose.ui.geometry.Size(w, h),
                        style = Stroke(width = 3f)
                    )

                    val labelHeight = 28.dp.toPx()
                    drawRect(
                        color = boxColor.copy(alpha = 0.7f),
                        topLeft = Offset(left, top - labelHeight),
                        size = androidx.compose.ui.geometry.Size(w, labelHeight)
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        label,
                        left + 6.dp.toPx(),
                        top - 8.dp.toPx(),
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.WHITE
                            textSize = 16.dp.toPx()
                            isAntiAlias = true
                            typeface = android.graphics.Typeface.DEFAULT_BOLD
                        }
                    )
                }
            }
        }

        // Bottom controls
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp + bottomPadding, start = 16.dp, end = 16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            FilledTonalButton(
                onClick = onShowPersonsList,
                modifier = Modifier.semantics { contentDescription = "Lista persoanelor" },
                colors = ButtonDefaults.filledTonalButtonColors(
                    containerColor = Color.Black.copy(alpha = 0.6f),
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(16.dp)
            ) {
                Icon(Icons.Default.People, null, Modifier.size(20.dp))
                Spacer(Modifier.width(6.dp))
                Text("Listă", fontWeight = FontWeight.SemiBold)
            }

            Button(
                onClick = {
                    val faces = trackedFaces
                    if (faces.isEmpty()) {
                        onSpeakRequest("Nu văd nicio persoană")
                    } else {
                        val known = faces.filter { it.name != null }
                        val unknown = faces.size - known.size
                        val parts = mutableListOf<String>()
                        if (faces.size == 1) parts.add("Văd o persoană:")
                        else parts.add("Văd ${faces.size} persoane:")
                        for (k in known) parts.add(k.name!!)
                        if (unknown == 1) parts.add("o persoană necunoscută")
                        else if (unknown > 1) parts.add("$unknown persoane necunoscute")
                        onSpeakRequest(parts.joinToString(", "))
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = Indigo, contentColor = Color.White),
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.semantics { contentDescription = "Cine este în fața camerei" }
            ) {
                Icon(Icons.Default.RecordVoiceOver, null, Modifier.size(20.dp))
                Spacer(Modifier.width(6.dp))
                Text("Cine e?", fontWeight = FontWeight.Bold)
            }

            FilledTonalButton(
                onClick = onStartRegistration,
                modifier = Modifier.semantics { contentDescription = "Înregistrează persoană nouă" },
                colors = ButtonDefaults.filledTonalButtonColors(containerColor = Emerald, contentColor = Color.White),
                shape = RoundedCornerShape(16.dp)
            ) {
                Icon(Icons.Default.PersonAdd, null, Modifier.size(20.dp))
                Spacer(Modifier.width(6.dp))
                Text("Nou", fontWeight = FontWeight.SemiBold)
            }
        }
    }

    // Cleanup
    DisposableEffect(Unit) {
        onDispose {
            isActive.set(false)
            cameraExecutor.shutdown()
            cameraExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)
            mlScope.cancel()
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Exception) {}
            // Close models on background thread with delay to avoid
            // destroying ONNX sessions while native calls are in-flight
            val det = faceDetector
            val emb = faceEmbedder
            Thread {
                Thread.sleep(500)
                det?.close()
                emb?.close()
            }.start()
        }
    }
}

private suspend fun embedAndMatch(
    bitmap: Bitmap,
    box: FloatArray,
    embedder: FaceEmbedder,
    cachedPersons: List<CachedPerson>
): Pair<String, Float>? {
    val fw = box[2] - box[0]
    val fh = box[3] - box[1]
    val padX = (fw * 0.2f).toInt()
    val padY = (fh * 0.2f).toInt()
    val cropLeft = (box[0].toInt() - padX).coerceAtLeast(0)
    val cropTop = (box[1].toInt() - padY).coerceAtLeast(0)
    val cropRight = (box[2].toInt() + padX).coerceAtMost(bitmap.width)
    val cropBottom = (box[3].toInt() + padY).coerceAtMost(bitmap.height)
    val cw = cropRight - cropLeft
    val ch = cropBottom - cropTop

    if (cw < 20 || ch < 20) return null

    val faceCrop = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cw, ch)

    // DEBUG: save face crop to disk to inspect quality
    try {
        val debugDir = java.io.File(bitmap.let { android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_PICTURES) }, "face_debug")
        debugDir.mkdirs()
        val file = java.io.File(debugDir, "crop_${System.currentTimeMillis()}.jpg")
        java.io.FileOutputStream(file).use { out ->
            faceCrop.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        Log.d(TAG, "DEBUG: saved face crop to ${file.absolutePath}")
    } catch (e: Exception) {
        Log.w(TAG, "DEBUG: couldn't save crop: ${e.message}")
    }

    val embedding = embedder.extractEmbedding(faceCrop)
    faceCrop.recycle()

    if (embedding == null) return null

    // Diagnostic: log first 5 values of live embedding
    Log.d(TAG, "LIVE emb first 5: [${embedding[0]}, ${embedding[1]}, ${embedding[2]}, ${embedding[3]}, ${embedding[4]}]")

    var bestName: String? = null
    var bestSim = FaceEmbedder.RECOGNITION_THRESHOLD

    for (cached in cachedPersons) {
        var maxSimForPerson = 0f
        // Log first 5 values of first stored embedding (once per person)
        if (cached.embeddings.isNotEmpty()) {
            val stored = cached.embeddings[0]
            Log.d(TAG, "STORED '${cached.person.name}' emb[0] first 5: [${stored[0]}, ${stored[1]}, ${stored[2]}, ${stored[3]}, ${stored[4]}]")
        }
        for (emb in cached.embeddings) {
            val sim = embedder.cosineSimilarity(embedding, emb)
            if (sim > maxSimForPerson) maxSimForPerson = sim
            if (sim > bestSim) {
                bestSim = sim
                bestName = cached.person.name
            }
        }
        Log.d(TAG, "Match vs '${cached.person.name}': maxSim=$maxSimForPerson (threshold=${FaceEmbedder.RECOGNITION_THRESHOLD})")
    }

    Log.d(TAG, "Best match: name=$bestName, sim=$bestSim, cropSize=${cw}x${ch}")
    return if (bestName != null) Pair(bestName, bestSim) else null
}

private fun iou(boxA: FloatArray, boxB: FloatArray): Float {
    val xA = maxOf(boxA[0], boxB[0])
    val yA = maxOf(boxA[1], boxB[1])
    val xB = minOf(boxA[2], boxB[2])
    val yB = minOf(boxA[3], boxB[3])
    val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
    val boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    val boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6f)
}
