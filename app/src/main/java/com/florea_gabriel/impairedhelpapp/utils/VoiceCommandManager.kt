package com.florea_gabriel.impairedhelpapp.utils

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.Locale

/**
 * VoiceCommandManager: Handles push-to-talk voice command recognition.
 *
 * Provides speech recognition for mode switching, scene description, and navigation commands:
 *
 * Mode Commands:
 * - "Activează mod distanță" → Switch to Distance mode
 * - "Dezactivează mod distanță" → Switch to Detection mode
 *
 * Scene Description:
 * - "Descrie camera" → Describe current scene (Gemini Vision API)
 * - "Ce vezi" → Describe current scene
 *
 * Object Navigation (NEW):
 * - "Unde e [obiect]?" → Find and navigate to object
 * - "Caută [obiect]" → Find object
 * - "Du-mă la [obiect]" → Navigate to object
 * - "Oprește navigarea" → Stop current navigation
 *
 * Usage:
 * ```
 * val voiceManager = VoiceCommandManager(context) { command ->
 *     when (command) {
 *         VoiceCommand.ACTIVATE_DISTANCE -> selectedMode = 1
 *         VoiceCommand.DEACTIVATE_DISTANCE -> selectedMode = 0
 *     }
 * }
 *
 * // When push-to-talk button pressed:
 * voiceManager.startListening()
 * ```
 *
 * @param context Application context
 * @param onCommandRecognized Callback invoked when a valid command is recognized
 */
class VoiceCommandManager(
    private val context: Context,
    private val onCommandRecognized: (VoiceCommand) -> Unit
) {
    companion object {
        private const val TAG = "VoiceCommandManager"
    }

    /**
     * Voice commands supported by the application
     */
    enum class VoiceCommand {
        DESCRIBE_SCENE,       // Describe current scene/room
        FIND_OBJECT,          // Find/navigate to an object (with objectName parameter)
        SAVE_OBJECT,          // Save current view as a personal object (with objectName parameter)
        STOP_SEARCH,          // Stop current search
        GO_CAMERA,            // Navigate to camera tab
        GO_OBJECTS,           // Navigate to objects tab
        GO_COLORS,            // Navigate to colors tab
        GO_PERSONS,           // Navigate to persons tab
        GO_MONEY,             // Navigate to money tab
        UNKNOWN               // Unrecognized command
    }

    /**
     * Data class for command result with optional parameters
     */
    data class CommandResult(
        val command: VoiceCommand,
        val objectName: String? = null  // For FIND_OBJECT command
    )

    // SpeechRecognizer instance
    private var speechRecognizer: SpeechRecognizer? = null

    // Callbacks for listening state
    private var onListeningStarted: (() -> Unit)? = null
    private var onListeningEnded: (() -> Unit)? = null
    private var onError: ((String) -> Unit)? = null

    // Callback for commands with parameters
    private var onFindObjectCommand: ((String) -> Unit)? = null
    private var onSaveObjectCommand: ((String) -> Unit)? = null

    // Offline/online mode tracking
    private var preferOffline = true // Try offline first
    private var hasTriedOffline = false

    init {
        initializeSpeechRecognizer()
    }

    /**
     * Initialize SpeechRecognizer with RecognitionListener
     */
    private fun initializeSpeechRecognizer() {
        val isAvailable = SpeechRecognizer.isRecognitionAvailable(context)
        Log.d(TAG, "Checking speech recognition availability: $isAvailable")

        if (!isAvailable) {
            Log.e(TAG, "Speech recognition not available on this device")
            Log.e(TAG, "Please install/update Google app from Play Store")
            return
        }

        Log.d(TAG, "Creating SpeechRecognizer instance...")
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    Log.d(TAG, "Ready for speech")
                    onListeningStarted?.invoke()
                }

                override fun onBeginningOfSpeech() {
                    Log.d(TAG, "Beginning of speech")
                }

                override fun onRmsChanged(rmsdB: Float) {
                    // Audio level changed - can be used for visual feedback
                }

                override fun onBufferReceived(buffer: ByteArray?) {
                    // Audio buffer received
                }

                override fun onEndOfSpeech() {
                    Log.d(TAG, "End of speech")
                    onListeningEnded?.invoke()
                }

                override fun onError(error: Int) {
                    val errorMessage = getErrorMessage(error)
                    Log.e(TAG, "Recognition error: $errorMessage")

                    // Handle ERROR_LANGUAGE_NOT_SUPPORTED (12) - fallback to cloud
                    if (error == 12 && preferOffline && !hasTriedOffline) {
                        Log.w(TAG, "ON-DEVICE Romanian not available, falling back to CLOUD mode (requires internet)")
                        hasTriedOffline = true
                        preferOffline = false // Switch to cloud mode

                        // Retry with cloud mode automatically
                        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                            Log.d(TAG, "Auto-retrying with CLOUD mode...")
                            startListening()
                        }, 300) // Small delay before retry
                        return
                    }

                    // For other errors or if already tried cloud, notify user
                    onError?.invoke(errorMessage)
                    onListeningEnded?.invoke()
                }

                override fun onResults(results: Bundle?) {
                    Log.d(TAG, "=== onResults callback triggered ===")

                    // Reset offline retry flag on successful recognition
                    hasTriedOffline = false

                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)

                    if (matches == null || matches.isEmpty()) {
                        Log.w(TAG, "No recognition results received")
                        onError?.invoke("Nu am înțeles comanda")
                        onListeningEnded?.invoke()
                        return
                    }

                    val recognitionMode = if (preferOffline) "ON-DEVICE" else "CLOUD"
                    Log.d(TAG, "Recognition returned ${matches.size} results (Mode: $recognitionMode):")
                    matches.forEachIndexed { index, result ->
                        Log.d(TAG, "  Result #$index: '$result'")
                    }

                    // Process all results to find best match
                    val command = parseCommand(matches)

                    if (command != VoiceCommand.UNKNOWN) {
                        Log.i(TAG, "Command recognized: $command")
                        onCommandRecognized(command)
                    } else {
                        Log.w(TAG, "No valid command found in results")
                        onError?.invoke("Comandă nerecunoscută")
                    }

                    onListeningEnded?.invoke()
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    // Partial recognition results (optional feedback)
                    val partial = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (partial != null && partial.isNotEmpty()) {
                        Log.d(TAG, "Partial result: '${partial[0]}'")
                    } else {
                        Log.v(TAG, "Partial results callback, but no data")
                    }
                }

                override fun onEvent(eventType: Int, params: Bundle?) {
                    // Custom events
                }
            })
        }

        Log.d(TAG, "VoiceCommandManager initialized successfully")
    }

    /**
     * Start listening for voice commands (push-to-talk)
     */
    fun startListening() {
        Log.d(TAG, "========================================")
        Log.d(TAG, "startListening() called")

        if (speechRecognizer == null) {
            Log.e(TAG, "SpeechRecognizer not initialized")
            onError?.invoke("Recunoaștere vocală indisponibilă")
            return
        }

        val mode = if (preferOffline) "ON-DEVICE (offline)" else "CLOUD (requires internet)"
        Log.d(TAG, "Setting up RecognizerIntent with $mode recognition...")

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            // Use free form language model (best for natural commands)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)

            // Set language to Romanian
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ro-RO")
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "ro-RO")

            //  Try ON-DEVICE first, fallback to CLOUD if not available
            if (preferOffline) {
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                Log.d(TAG, "  → Trying ON-DEVICE recognition first...")
            } else {
                // Don't set EXTRA_PREFER_OFFLINE - use cloud
                Log.d(TAG, "  → Using CLOUD recognition (Romanian on-device not available)")
            }

            // Request multiple results for better matching
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)

            // Enable partial results for faster feedback
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)

            // Prompt for user (not shown in custom UI)
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Spune comanda...")
        }

        try {
            speechRecognizer?.startListening(intent)
            Log.i(TAG, "✅ Started listening for voice commands (Language: ro-RO, Mode: $mode, Max results: 5)")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error starting speech recognition: ${e.message}", e)
            onError?.invoke("Eroare la pornirea recunoașterii")
        }
    }

    /**
     * Stop listening (if currently active)
     */
    fun stopListening() {
        speechRecognizer?.stopListening()
        Log.d(TAG, "Stopped listening")
    }

    /**
     * Cancel current recognition
     */
    fun cancel() {
        speechRecognizer?.cancel()
        Log.d(TAG, "Recognition cancelled")
    }

    /**
     * Parse recognized text into a VoiceCommand
     */
    private fun parseCommand(results: List<String>): VoiceCommand {
        Log.d(TAG, "--- parseCommand() ---")
        Log.d(TAG, "Parsing ${results.size} results...")

        // Convert all results to lowercase for case-insensitive matching
        val normalizedResults = results.map { it.lowercase(Locale.getDefault()) }

        // Check each result for command patterns
        for ((index, text) in normalizedResults.withIndex()) {
            Log.d(TAG, "  Checking result #$index: '$text'")

            // Scene Description patterns - "Descrie camera", "Ce vezi", "Descrie încăperea"
            val hasDescribe = text.contains("descrie") || text.contains("spune")
            val hasRoom = text.contains("camera") || text.contains("încăpere") || text.contains("incapere") ||
                          text.contains("locație") || text.contains("locatie") || text.contains("scena") ||
                          text.contains("ce vezi") || text.contains("ce e în față") || text.contains("ce e in fata")

            if (hasDescribe && hasRoom) {
                Log.i(TAG, "    ✅ Match: DESCRIBE_SCENE (describe + room keywords found)")
                return VoiceCommand.DESCRIBE_SCENE
            }

            // Direct phrases for scene description
            if (text.contains("ce vezi") || text.contains("ce e în jur") || text.contains("ce e in jur") ||
                text.contains("ce se află") || text.contains("ce se afla")) {
                Log.i(TAG, "    ✅ Match: DESCRIBE_SCENE (direct phrase)")
                return VoiceCommand.DESCRIBE_SCENE
            }

            // Stop search patterns
            if ((text.contains("oprește") || text.contains("opreste") || text.contains("stop") ||
                text.contains("anulează") || text.contains("anuleaza")) &&
                (text.contains("navigare") || text.contains("căutare") || text.contains("cautare") ||
                 text.contains("ghidare") || text.contains("search"))) {
                Log.i(TAG, "    ✅ Match: STOP_SEARCH")
                return VoiceCommand.STOP_SEARCH
            }

            // Find object patterns - "Unde e [obiect]", "Caută [obiect]", "Găsește [obiect]", "Du-mă la [obiect]"
            val findPatterns = listOf(
                "unde e ", "unde este ", "unde îmi e ", "unde imi e ",
                "caută ", "cauta ", "găsește ", "gaseste ", "găsește-mi ", "gaseste-mi ",
                "du-mă la ", "du-ma la ", "mergi la ", "navighează la ", "navigheaza la ",
                "vreau la ", "vreau să ajung la ", "vreau sa ajung la ",
                "unde am ", "unde mi-am ", "unde mi am "
            )

            for (pattern in findPatterns) {
                if (text.contains(pattern)) {
                    // Extract object name after the pattern
                    val patternIndex = text.indexOf(pattern)
                    val afterPattern = text.substring(patternIndex + pattern.length).trim()

                    // Clean up the object name
                    val objectName = afterPattern
                        .replace("?", "")
                        .replace("!", "")
                        .replace(".", "")
                        .split(" ")
                        .take(3)  // Take first 3 words max
                        .joinToString(" ")
                        .trim()

                    if (objectName.isNotEmpty()) {
                        Log.i(TAG, "    ✅ Match: FIND_OBJECT - target='$objectName'")
                        // Call the find object callback with the object name
                        onFindObjectCommand?.invoke(objectName)
                        return VoiceCommand.FIND_OBJECT
                    }
                }
            }

            // Save object patterns - "Salvează [nume]", "Reține ca [nume]", "Memorează ca [nume]"
            val savePatterns = listOf(
                "salvează ", "salveaza ", "salvează ca ", "salveaza ca ",
                "reține ", "retine ", "reține ca ", "retine ca ",
                "memorează ", "memoreaza ", "memorează ca ", "memoreaza ca ",
                "ține minte ", "tine minte ", "reține asta ca ", "retine asta ca "
            )

            for (pattern in savePatterns) {
                if (text.contains(pattern)) {
                    val patternIndex = text.indexOf(pattern)
                    val afterPattern = text.substring(patternIndex + pattern.length).trim()

                    val objectName = afterPattern
                        .replace("?", "")
                        .replace("!", "")
                        .replace(".", "")
                        .split(" ")
                        .take(3)
                        .joinToString(" ")
                        .trim()

                    if (objectName.isNotEmpty()) {
                        Log.i(TAG, "    Match: SAVE_OBJECT - name='$objectName'")
                        onSaveObjectCommand?.invoke(objectName)
                        return VoiceCommand.SAVE_OBJECT
                    }
                }
            }

            // ── Tab navigation commands ──────────────────────────────────────────

            // Go to Camera
            if (text.contains("cameră") || text.contains("camera") ||
                text.contains("deschide camera") || text.contains("mergi la cameră") ||
                text.contains("mergi la camera") || text.contains("detecție") || text.contains("detectie")) {
                // Avoid matching "descrie camera" which is DESCRIBE_SCENE (already handled above)
                if (!text.contains("descrie") && !text.contains("spune")) {
                    Log.i(TAG, "    Match: GO_CAMERA")
                    return VoiceCommand.GO_CAMERA
                }
            }

            // Go to Objects
            if (text.contains("obiecte") || text.contains("obiectele mele") ||
                text.contains("deschide obiecte") || text.contains("mergi la obiecte") ||
                text.contains("lista obiecte") || text.contains("obiectele")) {
                Log.i(TAG, "    Match: GO_OBJECTS")
                return VoiceCommand.GO_OBJECTS
            }

            // Go to Colors
            if (text.contains("culori") || text.contains("culoare") ||
                text.contains("ce culoare") || text.contains("detecție culori") ||
                text.contains("detectie culori") || text.contains("deschide culori")) {
                Log.i(TAG, "    Match: GO_COLORS")
                return VoiceCommand.GO_COLORS
            }

            // Go to Persons
            if (text.contains("persoane") || text.contains("persoanele") ||
                text.contains("recunoaște") || text.contains("recunoaste") ||
                text.contains("fețe") || text.contains("fete") ||
                text.contains("cine e") || text.contains("cine este")) {
                Log.i(TAG, "    Match: GO_PERSONS")
                return VoiceCommand.GO_PERSONS
            }

            // Go to Money
            if (text.contains("bani") || text.contains("bancnote") ||
                text.contains("verifică bani") || text.contains("verifica bani") ||
                text.contains("detecție bani") || text.contains("detectie bani") ||
                text.contains("câți bani") || text.contains("cati bani")) {
                Log.i(TAG, "    Match: GO_MONEY")
                return VoiceCommand.GO_MONEY
            }

            Log.v(TAG, "    No match for this result")
        }

        Log.w(TAG, "--- parseCommand() result: UNKNOWN ---")
        return VoiceCommand.UNKNOWN
    }

    /**
     * Convert error code to human-readable message
     */
    private fun getErrorMessage(error: Int): String {
        return when (error) {
            1 -> "ERROR_NETWORK_TIMEOUT (1) - Timeout rețea" // SpeechRecognizer.ERROR_NETWORK_TIMEOUT
            2 -> "ERROR_NETWORK (2) - Eroare de rețea" // SpeechRecognizer.ERROR_NETWORK
            3 -> "ERROR_AUDIO (3) - Eroare audio" // SpeechRecognizer.ERROR_AUDIO
            4 -> "ERROR_SERVER (4) - Eroare server" // SpeechRecognizer.ERROR_SERVER
            5 -> "ERROR_CLIENT (5) - Eroare client (normal când oprești manual)" // SpeechRecognizer.ERROR_CLIENT
            6 -> "ERROR_SPEECH_TIMEOUT (6) - Timeout vorbire" // SpeechRecognizer.ERROR_SPEECH_TIMEOUT
            7 -> "ERROR_NO_MATCH (7) - Nicio comandă recunoscută" // SpeechRecognizer.ERROR_NO_MATCH
            8 -> "ERROR_RECOGNIZER_BUSY (8) - Recunoaștere ocupată" // SpeechRecognizer.ERROR_RECOGNIZER_BUSY
            9 -> "ERROR_INSUFFICIENT_PERMISSIONS (9) - Permisiuni insuficiente" // SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS
            12 -> "ERROR_LANGUAGE_NOT_SUPPORTED (12) - Limba nu e suportată sau Google app lipsește/outdated"
            13 -> "ERROR_LANGUAGE_UNAVAILABLE (13) - Recunoaștere vocală indisponibilă pentru limba selectată"
            else -> "Eroare necunoscută (cod: $error)"
        }
    }

    /**
     * Set callback for when listening starts
     */
    fun setOnListeningStarted(callback: () -> Unit) {
        onListeningStarted = callback
    }

    /**
     * Set callback for when listening ends
     */
    fun setOnListeningEnded(callback: () -> Unit) {
        onListeningEnded = callback
    }

    /**
     * Set callback for errors
     */
    fun setOnError(callback: (String) -> Unit) {
        onError = callback
    }

    /**
     * Set callback for FIND_OBJECT command with object name parameter
     */
    fun setOnFindObjectCommand(callback: (String) -> Unit) {
        onFindObjectCommand = callback
    }

    /**
     * Set callback for SAVE_OBJECT command with object name parameter
     */
    fun setOnSaveObjectCommand(callback: (String) -> Unit) {
        onSaveObjectCommand = callback
    }

    /**
     * Check if speech recognition is available on this device
     */
    fun isAvailable(): Boolean {
        return SpeechRecognizer.isRecognitionAvailable(context)
    }

    /**
     * Release resources. Call this when done using VoiceCommandManager.
     */
    fun release() {
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
            Log.d(TAG, "VoiceCommandManager resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing VoiceCommandManager: ${e.message}")
        }
    }
}
