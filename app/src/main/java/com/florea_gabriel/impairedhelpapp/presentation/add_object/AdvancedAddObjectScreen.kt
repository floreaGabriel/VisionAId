package com.florea_gabriel.impairedhelpapp.presentation.add_object

import android.graphics.Bitmap
import android.graphics.Matrix
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.florea_gabriel.impairedhelpapp.ml.registration.AdvancedObjectRegistration
import com.florea_gabriel.impairedhelpapp.ml.validation.EmbeddingValidator
import com.florea_gabriel.impairedhelpapp.ui.theme.Amber
import com.florea_gabriel.impairedhelpapp.ui.theme.DarkBackground
import com.florea_gabriel.impairedhelpapp.ui.theme.Emerald
import com.florea_gabriel.impairedhelpapp.ui.theme.Indigo
import com.florea_gabriel.impairedhelpapp.ui.theme.Rose
import java.util.Locale
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "AdvancedAddObjectScreen"

/** UI Step for the registration flow. Controls which full-screen is displayed. */
private enum class RegistrationStep {
        INTRO, // Name input + explanation (full-screen)
        INSTRUCTION, // Phase details before capture (full-screen)
        CAPTURING, // Camera active with progress
        SUCCESS, // Object saved (full-screen)
        ERROR // Something went wrong (full-screen)
}

/**
 * AdvancedAddObjectScreen: Seeing AI-style guided object registration.
 *
 * Features:
 * - Guided capture with voice feedback
 * - Visual progress indicators
 * - IMU-based keyframe selection
 * - Blur detection
 * - Target: 50 embeddings per object
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AdvancedAddObjectScreen(
        onObjectSaved: () -> Unit,
        onBackClick: () -> Unit
) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val scope = rememberCoroutineScope()

        // UI State - controls which screen is shown
        var objectName by remember { mutableStateOf("") }
        var currentStep by remember { mutableStateOf(RegistrationStep.INTRO) }
        var errorMessage by remember { mutableStateOf<String?>(null) }
        var isSaving by remember { mutableStateOf(false) }

        // Phase tracking for user-triggered phase starts
        var currentPhase by remember { mutableStateOf(AdvancedObjectRegistration.PHASE_CLOSE) }

        // Validation result state
        var validationResult by remember {
                mutableStateOf<EmbeddingValidator.ValidationResult?>(null)
        }
        var showValidationDialog by remember { mutableStateOf(false) }

        // Registration system
        var registration by remember { mutableStateOf<AdvancedObjectRegistration?>(null) }
        val progress = registration?.progress?.collectAsState()?.value

        // Debug mask overlay
        var debugMaskEnabled by remember { mutableStateOf(false) }
        val debugOverlay = registration?.debugOverlay?.collectAsState()?.value

        // Sync debug mode with registration
        LaunchedEffect(debugMaskEnabled, registration) {
                registration?.debugMaskEnabled = debugMaskEnabled
        }

        // TTS for voice guidance
        var tts by remember { mutableStateOf<TextToSpeech?>(null) }

        // Camera
        val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
        val previewView = remember { PreviewView(context) }
        var isCapturing by remember { mutableStateOf(false) }

        // Initialize TTS
        LaunchedEffect(Unit) {
                tts =
                        TextToSpeech(context) { status ->
                                if (status == TextToSpeech.SUCCESS) {
                                        tts?.language = Locale("ro", "RO")
                                        tts?.setSpeechRate(1.15f)
                                        tts?.setPitch(0.8f)
                                }
                        }
        }

        // Initialize registration system
        LaunchedEffect(Unit) {
                registration =
                        AdvancedObjectRegistration(context) { message ->
                                tts?.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                registration?.initialize()
        }

        // Setup camera
        LaunchedEffect(Unit) {
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener(
                        {
                                val cameraProvider = cameraProviderFuture.get()

                                val preview =
                                        Preview.Builder().build().also {
                                                it.setSurfaceProvider(previewView.surfaceProvider)
                                        }

                                val imageAnalysis =
                                        ImageAnalysis.Builder()
                                                .setBackpressureStrategy(
                                                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                                )
                                                .build()
                                                .also { analysis ->
                                                        analysis.setAnalyzer(cameraExecutor) {
                                                                imageProxy ->
                                                                // Only check isCapturing flag -
                                                                // don't access Compose
                                                                // State from background thread
                                                                Log.d(
                                                                        TAG,
                                                                        "Frame received - isCapturing: $isCapturing"
                                                                )

                                                                if (isCapturing) {
                                                                        scope.launch {
                                                                                processFrame(
                                                                                        imageProxy,
                                                                                        registration
                                                                                )
                                                                        }
                                                                } else {
                                                                        imageProxy.close()
                                                                }
                                                        }
                                                }

                                try {
                                        cameraProvider.unbindAll()
                                        cameraProvider.bindToLifecycle(
                                                lifecycleOwner,
                                                CameraSelector.DEFAULT_BACK_CAMERA,
                                                preview,
                                                imageAnalysis
                                        )
                                } catch (e: Exception) {
                                        Log.e(TAG, "Camera binding failed", e)
                                        errorMessage = "Eroare la pornirea camerei"
                                }
                        },
                        ContextCompat.getMainExecutor(context)
                )
        }

        // Handle state transitions
        LaunchedEffect(progress?.state) {
                Log.d(
                        TAG,
                        "State changed to: ${progress?.state}, isCapturing will be: ${progress?.state == AdvancedObjectRegistration.RegistrationState.CAPTURING}"
                )
                when (progress?.state) {
                        AdvancedObjectRegistration.RegistrationState.CAPTURING -> {
                                isCapturing = true
                                registration?.startSensors()
                                Log.d(TAG, "Set isCapturing = true, started sensors")
                        }
                        AdvancedObjectRegistration.RegistrationState.COMPLETE -> {
                                isCapturing = false
                                registration?.stopSensors()

                                // Auto-save when complete with validation
                                if (!isSaving) {
                                        isSaving = true
                                        saveObject(
                                                registration = registration!!,
                                                uiScope = scope,
                                                context = context,
                                                objectName = objectName,
                                                onSuccess = {
                                                        val qualityText =
                                                                validationResult?.let {
                                                                        "Calitate ${(it.quality * 100).toInt()}%."
                                                                }
                                                                        ?: ""
                                                        tts?.speak(
                                                                "Obiect înregistrat cu succes",
                                                                TextToSpeech.QUEUE_FLUSH,
                                                                null,
                                                                null
                                                        )
                                                        // Navigate to SuccessScreen
                                                        currentStep = RegistrationStep.SUCCESS
                                                },
                                                onError = { error ->
                                                        errorMessage = error
                                                        tts?.speak(
                                                                "Înregistrare eșuată. $error",
                                                                TextToSpeech.QUEUE_FLUSH,
                                                                null,
                                                                null
                                                        )
                                                        isSaving = false
                                                        // Navigate to ErrorScreen
                                                        currentStep = RegistrationStep.ERROR
                                                },
                                                onValidationResult = { result ->
                                                        validationResult = result
                                                        // Show validation dialog for low quality
                                                        if (result.quality < 0.7f) {
                                                                showValidationDialog = true
                                                        }
                                                }
                                        )
                                }
                        }
                        else -> {
                                isCapturing = false
                        }
                }
        }

        // Handle phase transitions - show instruction screen between phases
        LaunchedEffect(progress?.currentPhase, progress?.state) {
                val registrationPhase = progress?.currentPhase ?: return@LaunchedEffect
                val registrationState = progress?.state ?: return@LaunchedEffect

                // If phase changed and state is READY, show instruction for next phase
                if (registrationState == AdvancedObjectRegistration.RegistrationState.READY &&
                                registrationPhase > currentPhase &&
                                registrationPhase < AdvancedObjectRegistration.PHASE_COMPLETE
                ) {

                        Log.d(
                                TAG,
                                "Phase completed: $currentPhase -> $registrationPhase, showing instruction"
                        )

                        // Stop capturing, update phase, show instruction
                        isCapturing = false
                        registration?.stopSensors()

                        currentPhase = registrationPhase
                        currentStep = RegistrationStep.INSTRUCTION

                        // Voice feedback
                        val phaseName =
                                when (registrationPhase) {
                                        AdvancedObjectRegistration.PHASE_MEDIUM -> "distanța medie"
                                        AdvancedObjectRegistration.PHASE_FAR -> "alt fundal"
                                        else -> "următoarea fază"
                                }
                        tts?.speak(
                                "Faza completă! Pregătește obiectul pentru $phaseName.",
                                TextToSpeech.QUEUE_FLUSH,
                                null,
                                null
                        )
                }
        }

        // Cleanup
        DisposableEffect(Unit) {
                onDispose {
                        cameraExecutor.shutdown()
                        registration?.release()
                        tts?.shutdown()
                }
        }

        Box(modifier = Modifier.fillMaxSize().padding(top = 48.dp, bottom = 80.dp)) {
                // Camera preview
                AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

                // Semi-transparent overlay
                Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.3f)))

                // IntroScreen (Full-screen name input)
                AnimatedVisibility(
                        visible = currentStep == RegistrationStep.INTRO,
                        enter = fadeIn() + scaleIn(),
                        exit = fadeOut() + scaleOut(),
                        modifier = Modifier.align(Alignment.Center)
                ) {
                        IntroScreen(
                                objectName = objectName,
                                onNameChange = { objectName = it },
                                errorMessage = errorMessage,
                                onContinue = {
                                        if (objectName.isNotBlank()) {
                                                currentStep = RegistrationStep.INSTRUCTION
                                                errorMessage = null
                                        } else {
                                                errorMessage = "Te rog introdu un nume!"
                                        }
                                },
                                onBack = onBackClick
                        )
                }

                // PhaseInstructionScreen (Phase details before each capture phase)
                AnimatedVisibility(
                        visible = currentStep == RegistrationStep.INSTRUCTION,
                        enter = fadeIn() + scaleIn(),
                        exit = fadeOut() + scaleOut(),
                        modifier = Modifier.align(Alignment.Center)
                ) {
                        PhaseInstructionScreen(
                                phase = currentPhase,
                                onStart = {
                                        currentStep = RegistrationStep.CAPTURING
                                        if (currentPhase == AdvancedObjectRegistration.PHASE_CLOSE
                                        ) {
                                                // First phase - start registration
                                                registration?.startRegistration()
                                        } else {
                                                // Subsequent phases - just start the phase
                                                registration?.startPhase(currentPhase)
                                        }
                                },
                                onBack = {
                                        if (currentPhase == AdvancedObjectRegistration.PHASE_CLOSE
                                        ) {
                                                currentStep = RegistrationStep.INTRO
                                        } else {
                                                // Go back to previous phase instruction
                                                currentPhase = currentPhase - 1
                                        }
                                }
                        )
                }

                // Capture UI (when capturing)
                AnimatedVisibility(
                        visible = currentStep == RegistrationStep.CAPTURING && progress != null,
                        enter = fadeIn(),
                        exit = fadeOut()
                ) {
                        CaptureUI(
                                progress = progress!!,
                                objectName = objectName,
                                onStart = { registration?.startRegistration() },
                                onSkipPhase = { registration?.advancePhase() },
                                onBack = {
                                        registration?.stopSensors()
                                        currentStep = RegistrationStep.INSTRUCTION
                                }
                        )
                }

                // SuccessScreen (Full-screen success message)
                AnimatedVisibility(
                        visible = currentStep == RegistrationStep.SUCCESS,
                        enter = fadeIn() + scaleIn(),
                        exit = fadeOut() + scaleOut(),
                        modifier = Modifier.align(Alignment.Center)
                ) { SuccessScreen(objectName = objectName, onDone = onObjectSaved) }

                // ErrorScreen (Full-screen error message)
                AnimatedVisibility(
                        visible = currentStep == RegistrationStep.ERROR,
                        enter = fadeIn() + scaleIn(),
                        exit = fadeOut() + scaleOut(),
                        modifier = Modifier.align(Alignment.Center)
                ) {
                        ErrorScreen(
                                errorMessage = errorMessage ?: "Eroare necunoscută",
                                onRetry = { currentStep = RegistrationStep.INSTRUCTION },
                                onBack = onBackClick
                        )
                }

                // Embedding counter (top-right, during capture)
                if (currentStep == RegistrationStep.CAPTURING && progress != null) {
                        Row(
                                modifier = Modifier.align(Alignment.TopEnd).padding(12.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                        ) {
                                Surface(
                                        color = Color.Black.copy(alpha = 0.7f),
                                        shape = RoundedCornerShape(20.dp)
                                ) {
                                        Row(
                                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                                Icon(
                                                        imageVector = Icons.Default.Memory,
                                                        contentDescription = null,
                                                        tint = Indigo,
                                                        modifier = Modifier.size(20.dp)
                                                )
                                                Text(
                                                        text = "${progress?.embeddingsCollected ?: 0}",
                                                        color = Color.White,
                                                        fontWeight = FontWeight.Bold,
                                                        fontSize = 16.sp
                                                )
                                                validationResult?.let { result ->
                                                        Text(
                                                                text = "| ${(result.quality * 100).toInt()}%",
                                                                color = when {
                                                                        result.quality >= 0.7f -> Emerald
                                                                        result.quality >= 0.5f -> Amber
                                                                        else -> Rose
                                                                },
                                                                fontWeight = FontWeight.Bold,
                                                                fontSize = 16.sp
                                                        )
                                                }
                                        }
                                }

                                // Debug Mask Toggle Button
                                IconButton(
                                        onClick = { debugMaskEnabled = !debugMaskEnabled },
                                        modifier = Modifier.size(48.dp)
                                                .background(
                                                        if (debugMaskEnabled) Rose.copy(alpha = 0.7f)
                                                        else Color.Black.copy(alpha = 0.5f),
                                                        CircleShape
                                                )
                                ) {
                                        Icon(
                                                imageVector = Icons.Default.Visibility,
                                                contentDescription = "Toggle Debug Mask",
                                                tint = Color.White
                                        )
                                }
                        }
                }

                // Debug Overlay Image - shows red mask over detected object
                if (debugMaskEnabled && debugOverlay != null) {
                        Image(
                                bitmap = debugOverlay.asImageBitmap(),
                                contentDescription = "Debug Mask Overlay",
                                modifier = Modifier.fillMaxSize().alpha(0.7f),
                                contentScale = ContentScale.FillBounds
                        )
                }

                // Validation result dialog
                if (showValidationDialog && validationResult != null) {
                        ValidationResultDialog(
                                result = validationResult!!,
                                onDismiss = { showValidationDialog = false }
                        )
                }
        }
}

/** Dialog showing validation result details */
@Composable
private fun ValidationResultDialog(
        result: EmbeddingValidator.ValidationResult,
        onDismiss: () -> Unit
) {
        AlertDialog(
                onDismissRequest = onDismiss,
                title = {
                        Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                                Icon(
                                        imageVector =
                                                if (result.isValid) Icons.Default.CheckCircle
                                                else Icons.Default.Warning,
                                        contentDescription = null,
                                        tint = if (result.isValid) Emerald else Amber
                                )
                                Text(
                                        text = "Calitate: ${(result.quality * 100).toInt()}%",
                                        fontWeight = FontWeight.Bold
                                )
                        }
                },
                text = {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                // Stats
                                Text(text = "📊 Statistici:", fontWeight = FontWeight.SemiBold)
                                Text("• Embeddings: ${result.details.embeddingCount}")
                                Text(
                                        "• Consistență: ${(result.intraClassSimilarity * 100).toInt()}%"
                                )
                                Text(
                                        "• Threshold recomandat: ${String.format("%.2f", result.optimalThreshold)}"
                                )

                                Spacer(modifier = Modifier.height(8.dp))

                                // Recommendations
                                Text(text = "💡 Recomandări:", fontWeight = FontWeight.SemiBold)
                                result.recommendations.forEach { rec ->
                                        Text(
                                                text = rec,
                                                fontSize = 14.sp,
                                                color =
                                                        when {
                                                                rec.startsWith("✅") -> Emerald
                                                                rec.startsWith("⚠️") -> Amber
                                                                rec.startsWith("❌") -> Rose
                                                                else -> Color.Gray
                                                        }
                                        )
                                }
                        }
                },
                confirmButton = { Button(onClick = onDismiss) { Text("OK") } }
        )
}

@Composable
private fun IntroScreen(
        objectName: String,
        onNameChange: (String) -> Unit,
        errorMessage: String?,
        onContinue: () -> Unit,
        onBack: () -> Unit
) {
        Box(
                modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
                contentAlignment = Alignment.Center
        ) {
                Card(
                        modifier = Modifier.fillMaxWidth(0.9f).padding(16.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
                ) {
                        Column(
                                modifier = Modifier.padding(28.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(20.dp)
                        ) {
                                // Header
                                Text(
                                        text = "Înregistrare Obiect",
                                        style = MaterialTheme.typography.headlineSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onSurface
                                )

                                // Description - simplified for 1-phase flow
                                Text(
                                        text =
                                                "Vei fi ghidat să rotești obiectul 360° în fața camerei pentru a captura " +
                                                        "toate unghiurile.\n\n" +
                                                        "Procesul durează aproximativ 2-3 minute.",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Start
                                )

                                HorizontalDivider()

                                // Name input
                                OutlinedTextField(
                                        value = objectName,
                                        onValueChange = onNameChange,
                                        modifier = Modifier.fillMaxWidth(),
                                        label = { Text("Numele obiectului") },
                                        placeholder = { Text("Ex: Portofelul meu") },
                                        singleLine = true,
                                        isError = errorMessage != null,
                                        colors =
                                                OutlinedTextFieldDefaults.colors(
                                                        focusedBorderColor =
                                                                MaterialTheme.colorScheme.primary,
                                                        focusedTextColor = MaterialTheme.colorScheme.onSurface,
                                                        unfocusedTextColor = MaterialTheme.colorScheme.onSurface
                                                )
                                )

                                if (errorMessage != null) {
                                        Text(
                                                text = errorMessage,
                                                color = MaterialTheme.colorScheme.error,
                                                style = MaterialTheme.typography.bodySmall
                                        )
                                }

                                // Buttons
                                Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                        OutlinedButton(
                                                onClick = onBack,
                                                modifier = Modifier.weight(1f)
                                        ) { Text("Anulează") }

                                        Button(
                                                onClick = onContinue,
                                                modifier = Modifier.weight(1f),
                                                enabled = objectName.isNotBlank(),
                                                colors =
                                                        ButtonDefaults.buttonColors(
                                                                containerColor = MaterialTheme.colorScheme.primary
                                                        )
                                        ) {
                                                Icon(
                                                        Icons.Default.ArrowForward,
                                                        contentDescription = null
                                                )
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text("Continuă")
                                        }
                                }
                        }
                }
        }
}

@Composable
private fun CaptureUI(
        progress: AdvancedObjectRegistration.RegistrationProgress,
        objectName: String,
        onStart: () -> Unit,
        onSkipPhase: () -> Unit,
        onBack: () -> Unit
) {
        val state = progress.state

        Box(modifier = Modifier.fillMaxSize()) {
                // Center target guide
                CenterTargetGuide(
                        isCapturing =
                                state == AdvancedObjectRegistration.RegistrationState.CAPTURING,
                        modifier = Modifier.align(Alignment.Center)
                )

                // Phase indicator and instructions
                Column(
                        modifier =
                                Modifier.fillMaxWidth()
                                        .align(Alignment.BottomCenter)
                                        .background(
                                                Brush.verticalGradient(
                                                        colors =
                                                                listOf(
                                                                        Color.Transparent,
                                                                        Color.Black.copy(
                                                                                alpha = 0.9f
                                                                        )
                                                                )
                                                )
                                        )
                                        .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                        // Phase progress dots - 1 phase only
                        if (state == AdvancedObjectRegistration.RegistrationState.CAPTURING) {
                                PhaseProgressDots(
                                        currentPhase = progress.currentPhase,
                                        totalPhases = 1
                                )
                        }

                        // Instruction text
                        Text(
                                text = progress.currentInstruction,
                                color = Color.White,
                                fontSize = 20.sp,
                                fontWeight = FontWeight.Medium,
                                textAlign = TextAlign.Center
                        )

                        // Progress bars
                        if (state == AdvancedObjectRegistration.RegistrationState.CAPTURING) {
                                Column(
                                        modifier = Modifier.fillMaxWidth(),
                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                        // Phase progress
                                        Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                                Text(
                                                        "Faza:",
                                                        color = Color.White.copy(alpha = 0.7f),
                                                        fontSize = 12.sp
                                                )
                                                LinearProgressIndicator(
                                                        progress = { progress.phaseProgress },
                                                        modifier =
                                                                Modifier.weight(1f)
                                                                        .height(6.dp)
                                                                        .clip(
                                                                                RoundedCornerShape(
                                                                                        3.dp
                                                                                )
                                                                        ),
                                                        color = Indigo
                                                )
                                                Text(
                                                        "${(progress.phaseProgress * 100).toInt()}%",
                                                        color = Color.White,
                                                        fontSize = 12.sp
                                                )
                                        }

                                        // Total progress
                                        Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                                Text(
                                                        "Total:",
                                                        color = Color.White.copy(alpha = 0.7f),
                                                        fontSize = 12.sp
                                                )
                                                LinearProgressIndicator(
                                                        progress = { progress.totalProgress },
                                                        modifier =
                                                                Modifier.weight(1f)
                                                                        .height(6.dp)
                                                                        .clip(
                                                                                RoundedCornerShape(
                                                                                        3.dp
                                                                                )
                                                                        ),
                                                        color = Emerald
                                                )
                                                Text(
                                                        "${progress.embeddingsCollected}",
                                                        color = Color.White,
                                                        fontSize = 12.sp
                                                )
                                        }
                                }

                                // Skip phase button
                                TextButton(onClick = onSkipPhase) {
                                        Text(
                                                "Salt la următoarea fază →",
                                                color = Color.White.copy(alpha = 0.7f)
                                        )
                                }
                        }

                        // Start button (when ready)
                        if (state == AdvancedObjectRegistration.RegistrationState.READY) {
                                Button(
                                        onClick = onStart,
                                        modifier = Modifier.fillMaxWidth().height(56.dp),
                                        shape = RoundedCornerShape(28.dp),
                                        colors =
                                                ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                                ) {
                                        Icon(Icons.Default.PlayArrow, contentDescription = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                                "Începe Înregistrarea",
                                                fontSize = 18.sp,
                                                fontWeight = FontWeight.Bold
                                        )
                                }

                                Text(
                                        text = "Ține obiectul \"$objectName\" în fața camerei",
                                        color = Color.White.copy(alpha = 0.7f),
                                        fontSize = 14.sp
                                )
                        }

                        // Complete state
                        if (state == AdvancedObjectRegistration.RegistrationState.COMPLETE) {
                                Card(
                                        colors = CardDefaults.cardColors(containerColor = Emerald),
                                        shape = RoundedCornerShape(16.dp)
                                ) {
                                        Row(
                                                modifier = Modifier.padding(16.dp),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                                        ) {
                                                Icon(
                                                        Icons.Default.CheckCircle,
                                                        contentDescription = null,
                                                        tint = Color.White,
                                                        modifier = Modifier.size(32.dp)
                                                )
                                                Text(
                                                        "Se salvează...",
                                                        color = Color.White,
                                                        fontWeight = FontWeight.Bold,
                                                        fontSize = 18.sp
                                                )
                                                CircularProgressIndicator(
                                                        modifier = Modifier.size(24.dp),
                                                        color = Color.White,
                                                        strokeWidth = 2.dp
                                                )
                                        }
                                }
                        }

                        // Error state
                        if (state == AdvancedObjectRegistration.RegistrationState.ERROR) {
                                Card(
                                        colors = CardDefaults.cardColors(containerColor = Rose),
                                        shape = RoundedCornerShape(16.dp)
                                ) {
                                        Column(
                                                modifier = Modifier.padding(16.dp),
                                                horizontalAlignment = Alignment.CenterHorizontally
                                        ) {
                                                Text(
                                                        progress.errorMessage
                                                                ?: "Eroare necunoscută",
                                                        color = Color.White,
                                                        textAlign = TextAlign.Center
                                                )
                                                Spacer(modifier = Modifier.height(8.dp))
                                                Button(
                                                        onClick = onBack,
                                                        colors =
                                                                ButtonDefaults.buttonColors(
                                                                        containerColor = Color.White
                                                                )
                                                ) { Text("Înapoi", color = Rose) }
                                        }
                                }
                        }
                }
        }
}

@Composable
private fun CenterTargetGuide(isCapturing: Boolean, modifier: Modifier = Modifier) {
        val rotation by
                rememberInfiniteTransition(label = "rotation")
                        .animateFloat(
                                initialValue = 0f,
                                targetValue = 360f,
                                animationSpec =
                                        infiniteRepeatable(
                                                animation = tween(3000, easing = LinearEasing),
                                                repeatMode = RepeatMode.Restart
                                        ),
                                label = "rotation"
                        )

        val scale by
                rememberInfiniteTransition(label = "scale")
                        .animateFloat(
                                initialValue = 1f,
                                targetValue = 1.1f,
                                animationSpec =
                                        infiniteRepeatable(
                                                animation =
                                                        tween(1000, easing = FastOutSlowInEasing),
                                                repeatMode = RepeatMode.Reverse
                                        ),
                                label = "scale"
                        )

        Box(modifier = modifier.size(200.dp), contentAlignment = Alignment.Center) {
                // Outer rotating ring
                Canvas(modifier = Modifier.size(200.dp).rotate(rotation)) {
                        val strokeWidth = 3.dp.toPx()

                        // Dashed circle
                        drawArc(
                                color =
                                        if (isCapturing) Emerald
                                        else Color.White.copy(alpha = 0.5f),
                                startAngle = 0f,
                                sweepAngle = 90f,
                                useCenter = false,
                                style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
                        )
                        drawArc(
                                color =
                                        if (isCapturing) Emerald
                                        else Color.White.copy(alpha = 0.5f),
                                startAngle = 180f,
                                sweepAngle = 90f,
                                useCenter = false,
                                style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
                        )
                }

                // Inner target box
                Box(
                        modifier =
                                Modifier.size((120 * scale).dp)
                                        .border(
                                                width = 2.dp,
                                                color =
                                                        if (isCapturing) Emerald
                                                        else Color.White.copy(alpha = 0.7f),
                                                shape = RoundedCornerShape(16.dp)
                                        )
                )

                // Center crosshair
                Canvas(modifier = Modifier.size(40.dp)) {
                        val center = Offset(size.width / 2, size.height / 2)
                        val lineLength = 15.dp.toPx()
                        val color = if (isCapturing) Emerald else Color.White

                        // Horizontal line
                        drawLine(
                                color = color,
                                start = Offset(center.x - lineLength, center.y),
                                end = Offset(center.x + lineLength, center.y),
                                strokeWidth = 2.dp.toPx(),
                                cap = StrokeCap.Round
                        )
                        // Vertical line
                        drawLine(
                                color = color,
                                start = Offset(center.x, center.y - lineLength),
                                end = Offset(center.x, center.y + lineLength),
                                strokeWidth = 2.dp.toPx(),
                                cap = StrokeCap.Round
                        )
                }
        }
}

@Composable
private fun PhaseProgressDots(currentPhase: Int, totalPhases: Int) {
        Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
        ) {
                repeat(totalPhases) { phase ->
                        val isCompleted = phase < currentPhase
                        val isCurrent = phase == currentPhase

                        Box(
                                modifier =
                                        Modifier.size(if (isCurrent) 14.dp else 10.dp)
                                                .clip(CircleShape)
                                                .background(
                                                        when {
                                                                isCompleted -> Emerald
                                                                isCurrent -> Indigo
                                                                else ->
                                                                        Color.White.copy(
                                                                                alpha = 0.3f
                                                                        )
                                                        }
                                                )
                                                .then(
                                                        if (isCurrent) {
                                                                Modifier.border(
                                                                        2.dp,
                                                                        Color.White,
                                                                        CircleShape
                                                                )
                                                        } else {
                                                                Modifier
                                                        }
                                                )
                        )
                }
        }
}

/** Process camera frame */
private suspend fun processFrame(
        imageProxy: ImageProxy,
        registration: AdvancedObjectRegistration?
) {
        try {
                @Suppress("DEPRECATION") var bitmap = imageProxy.toBitmap()

                // Fix rotation
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                if (rotationDegrees != 0) {
                        val matrix = Matrix()
                        matrix.postRotate(rotationDegrees.toFloat())
                        bitmap =
                                Bitmap.createBitmap(
                                        bitmap,
                                        0,
                                        0,
                                        bitmap.width,
                                        bitmap.height,
                                        matrix,
                                        true
                                )
                }

                registration?.processFrame(bitmap, null)

                // Cleanup
                if (rotationDegrees != 0) {
                        bitmap.recycle()
                }
        } catch (e: Exception) {
                Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
                imageProxy.close()
        }
}

/** Save registered object with validation */
private fun saveObject(
        registration: AdvancedObjectRegistration,
        uiScope: CoroutineScope,
        context: android.content.Context,
        objectName: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit,
        onValidationResult: ((EmbeddingValidator.ValidationResult) -> Unit)? = null
) {
        uiScope.launch(Dispatchers.IO) {
                try {
                        val embeddings = registration.getAllEmbeddings()

                        if (embeddings.isEmpty()) {
                                withContext(Dispatchers.Main) {
                                        onError("Nu s-au colectat embeddings")
                                }
                                return@launch
                        }

                        Log.d(TAG, "Validating ${embeddings.size} embeddings...")

                        // STEP 1: Validate embeddings quality
                        val validator = EmbeddingValidator()
                        val validationResult = validator.validate(embeddings)

                        // Report validation result
                        withContext(Dispatchers.Main) {
                                onValidationResult?.invoke(validationResult)
                        }

                        Log.d(
                                TAG,
                                "Validation: valid=${validationResult.isValid}, quality=${validationResult.quality}, " +
                                        "threshold=${validationResult.optimalThreshold}"
                        )

                        // STEP 2: Decide whether to save based on validation
                        if (!validationResult.isValid) {
                                withContext(Dispatchers.Main) {
                                        val message =
                                                "Calitate insuficientă (${(validationResult.quality * 100).toInt()}%). " +
                                                        validationResult
                                                                .recommendations
                                                                .firstOrNull {
                                                                        it.startsWith("⚠️") ||
                                                                                it.startsWith("❌")
                                                                }
                                                                ?.removePrefix("⚠️ ")
                                                                ?.removePrefix("❌ ")
                                                        ?: "Încearcă din nou."
                                        onError(message)
                                }
                                return@launch
                        }

                        Log.d(
                                TAG,
                                "Saving object with ${embeddings.size} embeddings (quality: ${(validationResult.quality * 100).toInt()}%)"
                        )

                        // STEP 3: Generate Gemini Labels & Thumbnail
                        val bestFrame = registration.getBestFrame()
                        var geminiLabels:
                                com.florea_gabriel.impairedhelpapp.ml.search.GeminiObjectLabeler.LabelingResult? =
                                null

                        if (bestFrame != null) {
                                geminiLabels = registration.generateSemanticLabels(bestFrame)
                        }

                        val finalDesc =
                                geminiLabels?.description
                                        ?: "${embeddings.size} embeddings, calitate ${(validationResult.quality * 100).toInt()}%"

                        // Save thumbnail
                        val thumbnailPath =
                                if (bestFrame != null) {
                                        saveThumbnail(
                                                context,
                                                bestFrame,
                                                "thumb_${System.currentTimeMillis()}"
                                        )
                                } else ""

                        // Convert embeddings to JSON (kept for debug) and Blob (fast loading)
                        // Static calls — no model loading needed for serialization
                        val embeddingJson = com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor.multipleEmbeddingsToJson(embeddings)
                        val embeddingBlob = com.florea_gabriel.impairedhelpapp.ml.feature.CLIPFeatureExtractor.embeddingsToBlob(embeddings)

                        // Save to database with quality metadata + Gemini metadata
                        val database =
                                com.florea_gabriel.impairedhelpapp.data.database.AppDatabase
                                        .getInstance(context)

                        val personalObject =
                                com.florea_gabriel.impairedhelpapp.data.database.PersonalObject(
                                        name = objectName.trim(), // Use user name primarily, Gemini
                                        // label as
                                        // metadata
                                        description = finalDesc,
                                        thumbnailPath = thumbnailPath,
                                        embeddingJson = embeddingJson, // Kept for debug
                                        embeddingBlob =
                                                embeddingBlob, // Fast loading (2.5x smaller)
                                        // Gemini Metadata (YOLO labels as offline fallback)
                                        geminiLabel = geminiLabels?.label,
                                        geminiDescription = geminiLabels?.description,
                                        detectionKeywords = geminiLabels?.keywords
                                                ?: registration.getDetectedYoloLabels()
                                                        .takeIf { it.isNotEmpty() }
                                                        ?.joinToString(","),
                                        // Adaptive threshold for search
                                        recommendedThreshold = validationResult.optimalThreshold
                                )

                        val id = database.personalObjectDao().insert(personalObject)

                        if (bestFrame != null && !bestFrame.isRecycled) {
                                bestFrame.recycle()
                        }

                        withContext(Dispatchers.Main) {
                                if (id > 0) {
                                        Log.i(
                                                TAG,
                                                "Object saved successfully: $objectName (id=$id, quality=${(validationResult.quality * 100).toInt()}%)"
                                        )
                                        onSuccess()
                                } else {
                                        onError("Eroare la salvare în baza de date")
                                }
                        }
                } catch (e: Exception) {
                        Log.e(TAG, "Save error: ${e.message}", e)
                        withContext(Dispatchers.Main) { onError("Eroare: ${e.message}") }
                }
        }
}

private fun saveThumbnail(context: android.content.Context, bitmap: Bitmap, name: String): String {
        try {
                val dir = java.io.File(context.filesDir, "thumbnails")
                if (!dir.exists()) dir.mkdirs()
                val file = java.io.File(dir, "$name.jpg")
                val stream = java.io.FileOutputStream(file)
                val thumb = Bitmap.createScaledBitmap(bitmap, 200, 200, true)
                thumb.compress(Bitmap.CompressFormat.JPEG, 80, stream)
                stream.close()
                thumb.recycle()
                return file.absolutePath
        } catch (e: Exception) {
                Log.e(TAG, "Failed to save thumbnail: ${e.message}")
                return ""
        }
}

// ============================================
// FULL-SCREEN COMPOSABLES
// ============================================

/**
 * PhaseInstructionScreen: Shows phase-specific instructions before starting each capture phase.
 * User must tap "Start" to begin the phase capture.
 */
@Composable
private fun PhaseInstructionScreen(phase: Int, onStart: () -> Unit, onBack: () -> Unit) {
        // Phase-specific content
        val (icon, title, instructions) =
                when (phase) {
                        AdvancedObjectRegistration.PHASE_CLOSE ->
                                Triple(
                                        Icons.Default.ZoomIn,
                                        "Faza 1: Aproape (20-30cm)",
                                        "Ține obiectul APROAPE de cameră.\n\n" +
                                                "• Distanța: 20-30 cm de cameră\n" +
                                                "• Rotește obiectul încet 360°\n" +
                                                "• Înclină-l în toate direcțiile\n" +
                                                "• Asigură-te că obiectul e bine iluminat"
                                )
                        AdvancedObjectRegistration.PHASE_MEDIUM ->
                                Triple(
                                        Icons.Default.CameraAlt,
                                        "Faza 2: Distanță Medie (50-70cm)",
                                        "Depărtează obiectul la distanță MEDIE.\n\n" +
                                                "• Distanța: 50-70 cm de cameră\n" +
                                                "• Mișcă-l stânga-dreapta încet\n" +
                                                "• Rotește-l să vezi toate părțile\n" +
                                                "• Păstrează obiectul în centru"
                                )
                        AdvancedObjectRegistration.PHASE_FAR ->
                                Triple(
                                        Icons.Default.ZoomOut,
                                        "Faza 3: Alt Fundal",
                                        "Pune obiectul pe un FUNDAL DIFERIT.\n\n" +
                                                "• Schimbă locația (altă masă, alt loc)\n" +
                                                "• Lasă obiectul nemișcat\n" +
                                                "• Mișcă telefonul în jurul lui\n" +
                                                "• Filmează din mai multe unghiuri"
                                )
                        else ->
                                Triple(
                                        Icons.Default.CameraAlt,
                                        "Pregătit pentru Înregistrare",
                                        "Ține obiectul în fața camerei."
                                )
                }

        Box(
                modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
                contentAlignment = Alignment.Center
        ) {
                Card(
                        modifier = Modifier.fillMaxWidth(0.9f).padding(16.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
                ) {
                        Column(
                                modifier = Modifier.padding(32.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(20.dp)
                        ) {
                                // Phase indicator dots
                                Row(
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                ) {
                                        repeat(3) { index ->
                                                Box(
                                                        modifier =
                                                                Modifier.size(
                                                                                if (index == phase)
                                                                                        12.dp
                                                                                else 8.dp
                                                                        )
                                                                        .background(
                                                                                if (index < phase)
                                                                                        Emerald
                                                                                else if (index ==
                                                                                                phase
                                                                                )
                                                                                        Indigo
                                                                                else
                                                                                        MaterialTheme.colorScheme.outlineVariant,
                                                                                CircleShape
                                                                        )
                                                )
                                        }
                                }

                                // Icon
                                Icon(
                                        imageVector = icon,
                                        contentDescription = null,
                                        modifier = Modifier.size(80.dp),
                                        tint = MaterialTheme.colorScheme.primary
                                )

                                // Title
                                Text(
                                        text = title,
                                        style = MaterialTheme.typography.headlineSmall,
                                        fontWeight = FontWeight.Bold,
                                        textAlign = TextAlign.Center,
                                        color = MaterialTheme.colorScheme.onSurface
                                )

                                // Instructions
                                Text(
                                        text = instructions,
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Start
                                )

                                Spacer(modifier = Modifier.height(8.dp))

                                // Buttons
                                Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                        OutlinedButton(
                                                onClick = onBack,
                                                modifier = Modifier.weight(1f)
                                        ) { Text("Înapoi") }

                                        Button(
                                                onClick = onStart,
                                                modifier = Modifier.weight(1f),
                                                colors =
                                                        ButtonDefaults.buttonColors(
                                                                containerColor = MaterialTheme.colorScheme.primary
                                                        )
                                        ) {
                                                Icon(
                                                        Icons.Default.PlayArrow,
                                                        contentDescription = null
                                                )
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text("Start Faza ${phase + 1}")
                                        }
                                }
                        }
                }
        }
}

/** SuccessScreen: Shows success message after object is saved. Full-screen, themed with Emerald. */
@Composable
private fun SuccessScreen(objectName: String, onDone: () -> Unit) {
        // Auto-navigate after 2 seconds
        LaunchedEffect(Unit) {
                delay(2000)
                onDone()
        }

        Box(
                modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
                contentAlignment = Alignment.Center
        ) {
                Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(24.dp)
                ) {
                        Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Succes",
                                modifier = Modifier.size(120.dp),
                                tint = Emerald
                        )

                        Text(
                                text = "Obiect Salvat!",
                                style = MaterialTheme.typography.headlineMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onBackground
                        )

                        Text(
                                text = "\"$objectName\"",
                                style = MaterialTheme.typography.titleLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                }
        }
}

/** ErrorScreen: Shows error message with retry option. Full-screen, themed with Rose. */
@Composable
private fun ErrorScreen(errorMessage: String, onRetry: () -> Unit, onBack: () -> Unit) {
        Box(
                modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
                contentAlignment = Alignment.Center
        ) {
                Card(
                        modifier = Modifier.fillMaxWidth(0.9f).padding(16.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
                ) {
                        Column(
                                modifier = Modifier.padding(32.dp),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(20.dp)
                        ) {
                                // Error icon
                                Icon(
                                        imageVector = Icons.Default.Error,
                                        contentDescription = "Eroare",
                                        modifier = Modifier.size(80.dp),
                                        tint = Rose
                                )

                                Text(
                                        text = "Înregistrare Eșuată",
                                        style = MaterialTheme.typography.headlineSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = Rose
                                )

                                Text(
                                        text = errorMessage,
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Center
                                )

                                Spacer(modifier = Modifier.height(8.dp))

                                // Buttons
                                Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                        OutlinedButton(
                                                onClick = onBack,
                                                modifier = Modifier.weight(1f)
                                        ) { Text("Anulează") }

                                        Button(
                                                onClick = onRetry,
                                                modifier = Modifier.weight(1f),
                                                colors =
                                                        ButtonDefaults.buttonColors(
                                                                containerColor = Rose
                                                        )
                                        ) {
                                                Icon(
                                                        Icons.Default.Refresh,
                                                        contentDescription = null
                                                )
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text("Reîncearcă")
                                        }
                                }
                        }
                }
        }
}
