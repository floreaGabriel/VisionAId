package com.florea_gabriel.impairedhelpapp.ml.captioning

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.florea_gabriel.impairedhelpapp.BuildConfig
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.android.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream

/**
 * ImageCaptioningManager: Handles scene description using Google Gemini Vision API.
 *
 * This manager captures the current camera frame and sends it to Gemini API
 * to generate a detailed description of the room/scene in Romanian.
 * Designed for visually impaired users to understand their surroundings.
 *
 * Flow:
 * 1. Capture current camera frame (Bitmap)
 * 2. Convert to Base64
 * 3. Send to Gemini API with Romanian prompt
 * 4. Return description text
 *
 * API: Google Gemini 3.5 Flash (gemini-3.5-flash)
 * - GA (stable) multimodal model with vision (image understanding)
 * - Fast inference (~1.5-2.5 seconds) with thinkingLevel = "low"
 * - Multilingual output (Romanian supported)
 *
 * Notes for the Gemini 3 series:
 * - Uses thinkingLevel ("minimal"/"low"/"medium"/"high") instead of the legacy thinkingBudget.
 * - temperature is kept at 1.0 (lower values degrade Gemini 3 output).
 * - maxOutputTokens must leave headroom for thinking tokens, otherwise the
 *   response can come back empty (finishReason = MAX_TOKENS).
 */
class ImageCaptioningManager {

    companion object {
        private const val TAG = "ImageCaptioning"
        private const val GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"

        // Romanian prompt optimized for blind users
        private const val SCENE_DESCRIPTION_PROMPT = """
Ești un asistent pentru persoane nevăzătoare. Descrie această imagine într-un mod util și concis.

Reguli importante:
1. Descrie încăperea și obiectele principale (mobilier, electronice, ferestre, uși)
2. Menționează poziția obiectelor (stânga, dreapta, centru, aproape, departe)
3. Identifică potențiale obstacole sau pericole
4. Folosește un limbaj clar și simplu în ROMÂNĂ
5. Limitează descrierea la 3-4 propoziții

Exemplu format: "Te afli într-un living. În centru este o canapea mare. În stânga, la aproximativ 2 metri, se află un televizor pe perete. În dreapta ai o fereastră."
"""

        // Romanian prompt for obstacle avoidance guidance (auto-triggered on sustained danger)
        private const val OBSTACLE_GUIDANCE_PROMPT = """
Ești un asistent pentru o persoană nevăzătoare aflată în mișcare. Utilizatorul are un obstacol foarte aproape în față și nu poate continua în direcția curentă. Imaginea reprezintă direcția în care voia să meargă.

Răspunde scurt și clar în ROMÂNĂ, în maxim 2 propoziții:
1. Ce obstacol este în față (foarte concis)
2. Pe unde să o ia ca să continue în siguranță (stânga, dreapta sau înapoi)

Format țintă: "În față ai [obstacol]. Ocolește prin [stânga/dreapta] / Întoarce-te."
Fii imediat și concret — utilizatorul nu vede și are nevoie de instrucțiunea acum.
"""
    }

    private val httpClient = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
        engine {
            connectTimeout = 30_000
            socketTimeout = 60_000
        }
    }

    /**
     * Describes the scene in the given bitmap image.
     *
     * @param bitmap The camera frame to describe
     * @return Result containing the description or error message
     */
    suspend fun describeScene(bitmap: Bitmap): Result<String> =
        callGemini(bitmap, SCENE_DESCRIPTION_PROMPT, maxOutputTokens = 1024, logLabel = "scene")

    /**
     * Auto-triggered when sustained proximity danger is detected.
     * Asks Gemini what's in front of the user and which direction to go.
     */
    suspend fun describeObstacleAndGuide(bitmap: Bitmap): Result<String> =
        callGemini(bitmap, OBSTACLE_GUIDANCE_PROMPT, maxOutputTokens = 512, logLabel = "obstacle")

    private suspend fun callGemini(
        bitmap: Bitmap,
        prompt: String,
        maxOutputTokens: Int,
        logLabel: String
    ): Result<String> {
        val apiKey = BuildConfig.GEMINI_API_KEY

        if (apiKey.isBlank()) {
            Log.e(TAG, "❌ Gemini API key not configured!")
            return Result.failure(Exception("API key not configured. Add GEMINI_API_KEY to local.properties"))
        }

        return try {
            Log.d(TAG, "🖼️ Starting Gemini call ($logLabel)...")
            val startTime = System.currentTimeMillis()

            // Convert bitmap to Base64
            val base64Image = bitmapToBase64(bitmap)
            Log.d(TAG, "📦 Image converted to Base64 (${base64Image.length} chars)")

            // Build request body
            val requestBody = GeminiRequest(
                contents = listOf(
                    Content(
                        parts = listOf(
                            Part(text = prompt),
                            Part(
                                inlineData = InlineData(
                                    mimeType = "image/jpeg",
                                    data = base64Image
                                )
                            )
                        )
                    )
                ),
                generationConfig = GenerationConfig(
                    // Gemini 3 is tuned for temperature 1.0 — going lower degrades quality.
                    temperature = 1.0f,
                    maxOutputTokens = maxOutputTokens,
                    // Gemini 3 replaces thinkingBudget with thinkingLevel. "minimal" is
                    // the "no thinking" setting — essential here because this is a
                    // real-time accessibility feature (obstacle guidance must be immediate).
                    // "low"/"medium" added 13–17s of latency for no quality gain on these
                    // short, factual prompts.
                    thinkingConfig = ThinkingConfig(thinkingLevel = "minimal")
                )
            )

            // Make API request
            Log.d(TAG, "🌐 Sending request to Gemini API...")
            val response: GeminiResponse = httpClient.post(GEMINI_API_URL) {
                parameter("key", apiKey)
                contentType(ContentType.Application.Json)
                setBody(requestBody)
            }.body()

            val duration = System.currentTimeMillis() - startTime

            // If the API returned an error payload, surface it instead of a silent fallback.
            response.error?.let { err ->
                Log.e(TAG, "❌ Gemini API error ${err.code} ${err.status}: ${err.message}")
                return Result.failure(Exception("Gemini ${err.code} ${err.status}: ${err.message}"))
            }

            // Extract text from response
            val candidate = response.candidates?.firstOrNull()
            val description = candidate
                ?.content
                ?.parts
                ?.firstOrNull()
                ?.text

            if (description == null) {
                // No text — log why (finishReason tells us: MAX_TOKENS, SAFETY, RECITATION...)
                val reason = candidate?.finishReason ?: "no candidates in response"
                Log.e(TAG, "❌ Gemini ($logLabel) returned no text. finishReason=$reason, full=$response")
                return Result.failure(Exception("Răspuns gol de la Gemini (motiv: $reason)"))
            }

            Log.i(TAG, "✅ Gemini ($logLabel) done in ${duration}ms")
            Log.d(TAG, "📝 Response: $description")

            Result.success(description)
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error in Gemini call ($logLabel): ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Converts a Bitmap to Base64 encoded string.
     * Uses JPEG compression for smaller payload.
     */
    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()

        // Scale down if too large (max 1024px on longest side)
        val scaledBitmap = if (bitmap.width > 1024 || bitmap.height > 1024) {
            val scale = 1024f / maxOf(bitmap.width, bitmap.height)
            val newWidth = (bitmap.width * scale).toInt()
            val newHeight = (bitmap.height * scale).toInt()
            Log.d(TAG, "📐 Scaling image from ${bitmap.width}x${bitmap.height} to ${newWidth}x${newHeight}")
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        } else {
            bitmap
        }

        scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 85, outputStream)
        val byteArray = outputStream.toByteArray()

        Log.d(TAG, "📊 Compressed image size: ${byteArray.size / 1024} KB")

        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
    }

    /**
     * Releases resources. Call when no longer needed.
     */
    fun release() {
        Log.d(TAG, "🧹 Releasing ImageCaptioningManager resources")
        httpClient.close()
    }
}

// ============== Gemini API Data Classes ==============

@Serializable
data class GeminiRequest(
    val contents: List<Content>,
    val generationConfig: GenerationConfig? = null
)

@Serializable
data class Content(
    val parts: List<Part>,
    val role: String? = null
)

@Serializable
data class Part(
    val text: String? = null,
    val inlineData: InlineData? = null
)

@Serializable
data class InlineData(
    val mimeType: String,
    val data: String
)

@Serializable
data class GenerationConfig(
    val temperature: Float? = null,
    val maxOutputTokens: Int? = null,
    val topP: Float? = null,
    val topK: Int? = null,
    val thinkingConfig: ThinkingConfig? = null
)

@Serializable
data class ThinkingConfig(
    val thinkingLevel: String? = null
)

@Serializable
data class GeminiResponse(
    val candidates: List<Candidate>? = null,
    val error: GeminiError? = null
)

@Serializable
data class Candidate(
    val content: Content? = null,
    val finishReason: String? = null
)

@Serializable
data class GeminiError(
    val code: Int? = null,
    val message: String? = null,
    val status: String? = null
)
