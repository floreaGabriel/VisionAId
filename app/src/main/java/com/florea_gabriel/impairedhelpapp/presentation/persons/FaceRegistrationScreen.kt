package com.florea_gabriel.impairedhelpapp.presentation.persons

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.florea_gabriel.impairedhelpapp.data.database.AppDatabase
import com.florea_gabriel.impairedhelpapp.data.database.KnownPerson
import com.florea_gabriel.impairedhelpapp.ml.face.FaceDetector
import com.florea_gabriel.impairedhelpapp.ml.face.FaceEmbedder
import com.florea_gabriel.impairedhelpapp.ml.processor.ImageProcessor
import com.florea_gabriel.impairedhelpapp.ui.theme.Emerald
import com.florea_gabriel.impairedhelpapp.ui.theme.Indigo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

private const val TAG = "FaceRegistration"

private enum class RegState {
    WAITING_FOR_FACE,
    NAME_INPUT,
    SCANNING,
    SAVING,
    DONE
}

private const val NUM_BUCKETS = 8
private const val TARGET_EMBEDDINGS = 15
private const val SCAN_TIMEOUT_MS = 20_000L

@Composable
fun FaceRegistrationScreen(
    onRegistrationComplete: (String) -> Unit,
    onBackClick: () -> Unit,
    onSpeakRequest: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val isActive = remember { AtomicBoolean(true) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val mlScope = remember { CoroutineScope(Dispatchers.Default + SupervisorJob()) }

    // ML models
    var faceDetector by remember { mutableStateOf<FaceDetector?>(null) }
    var faceEmbedder by remember { mutableStateOf<FaceEmbedder?>(null) }

    // State machine
    var regState by remember { mutableStateOf(RegState.WAITING_FOR_FACE) }
    var personName by remember { mutableStateOf("") }
    var statusMessage by remember { mutableStateOf("Se încarcă...") }

    // Detection state
    var faceDetected by remember { mutableStateOf(false) }
    var faceInCircle by remember { mutableStateOf(false) }
    var faceStableCount by remember { mutableIntStateOf(0) }
    var lastFaceBox by remember { mutableStateOf<FloatArray?>(null) }
    var lastLandmarks by remember { mutableStateOf<FloatArray?>(null) }
    var lastBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var previewWidth by remember { mutableIntStateOf(1) }
    var previewHeight by remember { mutableIntStateOf(1) }

    // Scanning state
    val collectedEmbeddings = remember { mutableListOf<FloatArray>() }
    val coveredBuckets = remember { mutableStateListOf<Int>() }
    var scanProgress by remember { mutableFloatStateOf(0f) }
    var bestThumbnail by remember { mutableStateOf<Bitmap?>(null) }

    var faceReady by remember { mutableStateOf(false) }

    // Push-to-talk
    var activeSpeechRecognizer by remember { mutableStateOf<SpeechRecognizer?>(null) }
    var isRecording by remember { mutableStateOf(false) }
    var showNameDialog by remember { mutableStateOf(false) }
    var nameDialogText by remember { mutableStateOf("") }

    var hasAudioPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val audioPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted -> hasAudioPermission = granted }

    // Initialize models
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            faceDetector = FaceDetector(context)
            faceEmbedder = FaceEmbedder(context)
        }
        if (faceDetector?.isReady() == true && faceEmbedder?.isReady() == true) {
            statusMessage = "Pune fața în cerc"
            onSpeakRequest("Pune fața în cercul de pe ecran")
        } else {
            statusMessage = "Eroare la încărcarea modelelor"
        }
        if (!hasAudioPermission) {
            audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    // WAITING → NAME_INPUT
    LaunchedEffect(faceReady) {
        if (!faceReady || regState != RegState.WAITING_FOR_FACE) return@LaunchedEffect
        regState = RegState.NAME_INPUT
        if (hasAudioPermission) {
            statusMessage = "Ține apăsat butonul și spune numele"
            onSpeakRequest("Față detectată. Ține apăsat butonul și spune numele")
        } else {
            statusMessage = "Introduceți numele"
            onSpeakRequest("Față detectată. Introduceți numele.")
            showNameDialog = true
        }
    }

    fun startPushToTalk() {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            showNameDialog = true
            return
        }
        isRecording = true
        statusMessage = "Ascult... spune numele"

        val recognizer = SpeechRecognizer.createSpeechRecognizer(context)
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ro-RO")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }

        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                recognizer.destroy()
                activeSpeechRecognizer = null
                isRecording = false
                statusMessage = "Introduceți numele"
                showNameDialog = true
            }
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull() ?: ""
                recognizer.destroy()
                activeSpeechRecognizer = null
                isRecording = false

                val cleaned = text.trim()
                if (cleaned.isNotBlank()) {
                    personName = cleaned.replaceFirstChar { it.uppercase() }
                    regState = RegState.SCANNING
                } else {
                    statusMessage = "Introduceți numele"
                    showNameDialog = true
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        activeSpeechRecognizer = recognizer
        recognizer.startListening(intent)
    }

    fun stopPushToTalk() {
        activeSpeechRecognizer?.stopListening()
    }

    // Continuous scanning with head rotation tracking
    LaunchedEffect(regState) {
        if (regState != RegState.SCANNING) return@LaunchedEffect

        statusMessage = "Rotește încet capul în toate direcțiile"
        onSpeakRequest("Rotește încet capul în toate direcțiile")

        val embedder = faceEmbedder ?: return@LaunchedEffect
        val startTime = System.currentTimeMillis()

        while (collectedEmbeddings.size < TARGET_EMBEDDINGS &&
            System.currentTimeMillis() - startTime < SCAN_TIMEOUT_MS &&
            isActive.get()
        ) {
            val bitmap = lastBitmap
            val box = lastFaceBox
            val landmarks = lastLandmarks

            if (bitmap != null && box != null && faceDetected) {
                val yaw = if (landmarks != null) estimateYaw(landmarks) else 0f
                val bucket = angleToBucket(yaw)

                val shouldCapture = !coveredBuckets.contains(bucket) || collectedEmbeddings.size < 3

                if (shouldCapture) {
                    val fw = box[2] - box[0]
                    val fh = box[3] - box[1]
                    val padX = (fw * 0.15f).toInt()
                    val padY = (fh * 0.15f).toInt()
                    val cropLeft = (box[0].toInt() - padX).coerceAtLeast(0)
                    val cropTop = (box[1].toInt() - padY).coerceAtLeast(0)
                    val cropRight = (box[2].toInt() + padX).coerceAtMost(bitmap.width)
                    val cropBottom = (box[3].toInt() + padY).coerceAtMost(bitmap.height)
                    val cw = cropRight - cropLeft
                    val ch = cropBottom - cropTop

                    if (cw > 80 && ch > 80) {
                        val faceCrop = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cw, ch)
                        val embedding = embedder.extractEmbedding(faceCrop)
                        if (embedding != null) {
                            collectedEmbeddings.add(embedding)
                            if (!coveredBuckets.contains(bucket)) {
                                coveredBuckets.add(bucket)
                            }
                            scanProgress = coveredBuckets.size.toFloat() / NUM_BUCKETS

                            if (bestThumbnail == null && bucket == NUM_BUCKETS / 2) {
                                bestThumbnail = faceCrop
                            } else {
                                faceCrop.recycle()
                            }

                            Log.d(TAG, "Scan: emb=${collectedEmbeddings.size}, bucket=$bucket, yaw=${"%.1f".format(yaw)}, buckets=${coveredBuckets.size}/$NUM_BUCKETS")

                            // Update status with progress
                            val pct = (coveredBuckets.size * 100 / NUM_BUCKETS)
                            statusMessage = "Scanare: $pct% — rotește capul"
                        } else {
                            faceCrop.recycle()
                        }
                    }
                }
            }

            delay(250)
        }

        if (collectedEmbeddings.size >= 3) {
            scanProgress = 1f
            regState = RegState.SAVING
        } else {
            statusMessage = "Prea puține capturi. Încearcă din nou."
            onSpeakRequest("Nu am reușit suficiente capturi. Încearcă din nou.")
            regState = RegState.WAITING_FOR_FACE
            faceReady = false
            faceStableCount = 0
            collectedEmbeddings.clear()
            coveredBuckets.clear()
            scanProgress = 0f
            bestThumbnail?.recycle()
            bestThumbnail = null
        }
    }

    // Save to DB
    LaunchedEffect(regState) {
        if (regState != RegState.SAVING) return@LaunchedEffect

        val embedder = faceEmbedder ?: return@LaunchedEffect
        statusMessage = "Se salvează..."

        try {
            val embList = collectedEmbeddings.toList()
            Log.d(TAG, "Saving ${embList.size} embeddings, ${coveredBuckets.size} buckets covered")

            val blob = embedder.embeddingsToBlob(embList)

            val thumbnailFile = File(context.filesDir, "face_${System.currentTimeMillis()}.jpg")
            val thumb = bestThumbnail
            if (thumb != null) {
                withContext(Dispatchers.IO) {
                    FileOutputStream(thumbnailFile).use { out ->
                        thumb.compress(Bitmap.CompressFormat.JPEG, 85, out)
                    }
                }
            }

            val person = KnownPerson(
                name = personName,
                embeddingBlob = blob,
                thumbnailPath = thumbnailFile.absolutePath
            )
            val dao = AppDatabase.getInstance(context).knownPersonDao()
            withContext(Dispatchers.IO) { dao.insert(person) }

            statusMessage = "$personName a fost salvat"
            onSpeakRequest("$personName a fost salvat")

            delay(2000)
            onRegistrationComplete(personName)
        } catch (e: Exception) {
            if (e is kotlinx.coroutines.CancellationException) throw e
            Log.e(TAG, "Save failed: ${e.message}", e)
            statusMessage = "Eroare la salvare"
            onSpeakRequest("Eroare la salvare")
        }
    }

    // UI
    Box(modifier = Modifier.fillMaxSize().padding(top = 48.dp, bottom = 80.dp).background(Color.Black)) {
        // Camera preview (front camera)
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
                        if (!isActive.get()) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val detector = faceDetector
                        if (detector == null || !detector.isReady()) {
                            imageProxy.close()
                            return@analyzer
                        }

                        val bitmap: Bitmap
                        try {
                            bitmap = ImageProcessor.imageProxyToBitmap(imageProxy)
                        } catch (e: Exception) {
                            imageProxy.close()
                            return@analyzer
                        }
                        imageProxy.close()

                        if (!isActive.get()) {
                            bitmap.recycle()
                            return@analyzer
                        }

                        mlScope.launch {
                            try {
                                val detections = detector.detectFaces(bitmap)

                                withContext(Dispatchers.Main) {
                                    previewWidth = bitmap.width
                                    previewHeight = bitmap.height

                                    if (detections.isNotEmpty()) {
                                        val best = detections.maxByOrNull {
                                            (it.box[2] - it.box[0]) * (it.box[3] - it.box[1])
                                        } ?: detections[0]

                                        lastFaceBox = best.box
                                        lastLandmarks = best.landmarks
                                        val old = lastBitmap
                                        lastBitmap = bitmap
                                        old?.recycle()

                                        faceDetected = true

                                        // Check if face is in the center region (circle area)
                                        val faceCx = (best.box[0] + best.box[2]) / 2f
                                        val faceCy = (best.box[1] + best.box[3]) / 2f
                                        val bmpCx = bitmap.width / 2f
                                        val bmpCy = bitmap.height * 0.4f
                                        val circleR = bitmap.width * 0.35f

                                        // Mirror X for front camera
                                        val mirroredFaceCx = bitmap.width - faceCx
                                        val dx = mirroredFaceCx - bmpCx
                                        val dy = faceCy - bmpCy
                                        val dist = sqrt(dx * dx + dy * dy)

                                        val faceW = best.box[2] - best.box[0]
                                        val faceSize = faceW
                                        val inCircle = dist < circleR * 0.6f &&
                                                faceSize > circleR * 0.5f &&
                                                faceSize < circleR * 2.5f
                                        faceInCircle = inCircle

                                        if (inCircle && regState == RegState.WAITING_FOR_FACE) {
                                            faceStableCount++
                                            if (faceStableCount >= 3 && !faceReady) {
                                                faceReady = true
                                            }
                                        }
                                    } else {
                                        faceDetected = false
                                        faceInCircle = false
                                        if (regState == RegState.WAITING_FOR_FACE) {
                                            faceStableCount = 0
                                        }
                                        lastFaceBox = null
                                        lastLandmarks = null
                                        bitmap.recycle()
                                    }
                                }
                            } catch (e: Exception) {
                                if (e is kotlinx.coroutines.CancellationException) return@launch
                                Log.e(TAG, "Analysis error: ${e.message}")
                                bitmap.recycle()
                            }
                        }
                    }

                    if (!isActive.get()) return@addListener

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
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

        // Circle overlay with cutout + progress arc
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
        ) {
            val circleRadius = size.width * 0.35f
            val circleCenter = Offset(size.width / 2f, size.height * 0.38f)

            // Dark overlay
            drawRect(Color.Black.copy(alpha = 0.5f), size = size)

            // Clear circle cutout
            drawCircle(
                color = Color.Transparent,
                radius = circleRadius,
                center = circleCenter,
                blendMode = BlendMode.Clear
            )

            // Circle border
            val borderColor = when {
                regState == RegState.SCANNING -> Emerald
                faceInCircle -> Color(0xFF4CAF50)
                faceDetected -> Color(0xFFFFA726)
                else -> Color.White.copy(alpha = 0.6f)
            }
            drawCircle(
                color = borderColor,
                radius = circleRadius,
                center = circleCenter,
                style = Stroke(width = 4f)
            )

            // Progress arc during scanning
            if (regState == RegState.SCANNING && scanProgress > 0f) {
                val arcSize = androidx.compose.ui.geometry.Size(circleRadius * 2 + 16f, circleRadius * 2 + 16f)
                val arcTopLeft = Offset(circleCenter.x - circleRadius - 8f, circleCenter.y - circleRadius - 8f)

                // Background track
                drawArc(
                    color = Color.White.copy(alpha = 0.2f),
                    startAngle = -90f,
                    sweepAngle = 360f,
                    useCenter = false,
                    topLeft = arcTopLeft,
                    size = arcSize,
                    style = Stroke(width = 8f)
                )

                // Progress
                drawArc(
                    color = Emerald,
                    startAngle = -90f,
                    sweepAngle = 360f * scanProgress,
                    useCenter = false,
                    topLeft = arcTopLeft,
                    size = arcSize,
                    style = Stroke(width = 8f, cap = androidx.compose.ui.graphics.StrokeCap.Round)
                )
            }
        }

        // Bottom controls
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp, start = 24.dp, end = 24.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Name during scanning
            if (personName.isNotBlank() && (regState == RegState.SCANNING || regState == RegState.SAVING)) {
                Text(
                    text = personName,
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                )
            }

            // Status message
            Box(
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(16.dp))
                    .padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    when (regState) {
                        RegState.SAVING -> CircularProgressIndicator(
                            color = Color.White, modifier = Modifier.size(24.dp), strokeWidth = 2.dp
                        )
                        RegState.DONE -> Icon(
                            Icons.Default.CheckCircle, null, tint = Emerald, modifier = Modifier.size(24.dp)
                        )
                        RegState.NAME_INPUT -> Icon(
                            Icons.Default.Mic, null, tint = Color(0xFFFFA726), modifier = Modifier.size(24.dp)
                        )
                        RegState.SCANNING -> Icon(
                            Icons.Default.FaceRetouchingNatural, null, tint = Emerald, modifier = Modifier.size(24.dp)
                        )
                        else -> {}
                    }

                    Text(
                        text = statusMessage,
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.SemiBold,
                        textAlign = TextAlign.Center
                    )
                }
            }

            // Push-to-talk button
            if (regState == RegState.NAME_INPUT && hasAudioPermission) {
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .background(
                            if (isRecording) Color(0xFFE53935) else Indigo,
                            CircleShape
                        )
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onPress = {
                                    startPushToTalk()
                                    tryAwaitRelease()
                                    stopPushToTalk()
                                }
                            )
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = "Ține apăsat pentru a spune numele",
                        tint = Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }

                TextButton(
                    onClick = {
                        statusMessage = "Introduceți numele"
                        showNameDialog = true
                    },
                    colors = ButtonDefaults.textButtonColors(contentColor = Color.White)
                ) {
                    Icon(Icons.Default.Keyboard, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Scrie manual", fontSize = 14.sp)
                }
            }

            // Manual trigger if face detected but not in circle
            if (regState == RegState.WAITING_FOR_FACE && faceDetected && !faceInCircle) {
                TextButton(
                    onClick = { faceReady = true },
                    colors = ButtonDefaults.textButtonColors(contentColor = Color.White)
                ) {
                    Icon(Icons.Default.TouchApp, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Apasă dacă fața e vizibilă", fontSize = 14.sp)
                }
            }
        }
    }

    // Name input dialog
    if (showNameDialog) {
        AlertDialog(
            onDismissRequest = {
                showNameDialog = false
                nameDialogText = ""
                regState = RegState.WAITING_FOR_FACE
                faceReady = false
                faceStableCount = 0
            },
            icon = {
                Icon(Icons.Default.Person, null, tint = Indigo, modifier = Modifier.size(32.dp))
            },
            title = {
                Text("Numele persoanei", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            },
            text = {
                OutlinedTextField(
                    value = nameDialogText,
                    onValueChange = { nameDialogText = it },
                    label = { Text("Nume") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (nameDialogText.isNotBlank()) {
                            personName = nameDialogText.trim().replaceFirstChar { it.uppercase() }
                            showNameDialog = false
                            nameDialogText = ""
                            regState = RegState.SCANNING
                        }
                    },
                    enabled = nameDialogText.isNotBlank(),
                    colors = ButtonDefaults.buttonColors(containerColor = Indigo),
                    shape = RoundedCornerShape(8.dp)
                ) { Text("Continuă", fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(
                    onClick = {
                        showNameDialog = false
                        nameDialogText = ""
                        regState = RegState.WAITING_FOR_FACE
                        faceReady = false
                        faceStableCount = 0
                    },
                    shape = RoundedCornerShape(8.dp)
                ) { Text("Anulează") }
            },
            shape = RoundedCornerShape(16.dp)
        )
    }

    // Cleanup
    DisposableEffect(Unit) {
        onDispose {
            activeSpeechRecognizer?.destroy()
            isActive.set(false)
            cameraExecutor.shutdown()
            cameraExecutor.awaitTermination(300, TimeUnit.MILLISECONDS)
            mlScope.cancel()
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Exception) {}
            val det = faceDetector
            val emb = faceEmbedder
            val bmp = lastBitmap
            val thumb = bestThumbnail
            Thread {
                Thread.sleep(500)
                det?.close()
                emb?.close()
                bmp?.recycle()
                thumb?.recycle()
            }.start()
        }
    }
}

private fun estimateYaw(landmarks: FloatArray): Float {
    if (landmarks.size < 10) return 0f
    val rightEyeX = landmarks[0]
    val leftEyeX = landmarks[2]
    val noseX = landmarks[4]

    val eyeMidX = (rightEyeX + leftEyeX) / 2f
    val eyeSpan = leftEyeX - rightEyeX
    if (kotlin.math.abs(eyeSpan) < 1f) return 0f

    val ratio = (noseX - eyeMidX) / eyeSpan
    return ratio * 90f
}

private fun angleToBucket(yaw: Float): Int {
    return ((yaw + 90f) / (180f / NUM_BUCKETS)).toInt().coerceIn(0, NUM_BUCKETS - 1)
}
