package com.florea_gabriel.impairedhelpapp.ml.search

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
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream

/**
 * GeminiObjectLabeler: Semantic object labeling using Gemini 2.0 Flash.
 * 
 * Used during object registration to generate:
 * 1. A human-readable label ("Blue Wallet")
 * 2. Visual description ("Leather wallet with zipper")
 * 3. Detection keywords used for Hybrid Search ("wallet", "bag", "accessory")
 */
class GeminiObjectLabeler {

    companion object {
        private const val TAG = "GeminiObjectLabeler"
        private const val API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    }

    private val httpClient = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
    }
    
    data class LabelingResult(
        val label: String,
        val description: String,
        val keywords: String
    )

    /**
     * Label the object in the image.
     */
    suspend fun labelObject(bitmap: Bitmap): Result<LabelingResult> {
        return try {
            val apiKey = BuildConfig.GEMINI_API_KEY
            if (apiKey.isBlank()) {
                return Result.failure(Exception("Gemini API key not configured"))
            }

            val base64Image = bitmapToBase64(bitmap)
            Log.d(TAG, "Generating semantic labels for object...")

            val request = GeminiRequest(
                contents = listOf(
                    Content(
                        parts = listOf(
                            Part(text = createPrompt()),
                            Part(inlineData = InlineData(mimeType = "image/jpeg", data = base64Image))
                        )
                    )
                ),
                generationConfig = GenerationConfig(
                    temperature = 0.2f,
                    responseMimeType = "application/json" // Force JSON response
                )
            )

            val response = httpClient.post(API_URL) {
                parameter("key", apiKey)
                contentType(ContentType.Application.Json)
                setBody(request)
            }

            if (response.status != HttpStatusCode.OK) {
                return Result.failure(Exception("API error: ${response.status}"))
            }

            val geminiResponse = response.body<GeminiResponse>()
            val jsonText = geminiResponse.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.text
            
            if (jsonText != null) {
                Log.d(TAG, "Gemini JSON: $jsonText")
                // Clean markdown code blocks if present ```json ... ```
                val cleanJson = jsonText.replace("```json", "").replace("```", "").trim()
                val parsed = Json.decodeFromString<JsonLabelResponse>(cleanJson)
                
                Result.success(LabelingResult(
                    label = parsed.label,
                    description = parsed.description,
                    keywords = parsed.keywords.joinToString(",")
                ))
            } else {
                Result.failure(Exception("Empty response from Gemini"))
            }

        } catch (e: Exception) {
            Log.e(TAG, "Labeling error: ${e.message}", e)
            Result.failure(e)
        }
    }

    private fun createPrompt(): String {
        return """
            Analyze the main object in this image.
            Provide output in strict JSON format:
            {
              "label": "Short name (2-3 words) in Romanian",
              "description": "Visual description for a blind user (in Romanian, max 15 words)",
              "keywords": ["english_keyword1", "english_keyword2", "english_keyword3"]
            }
            
            For "keywords", provide 3-5 English synonyms or categories that standard object detectors (like COCO classes) might recognize. 
            Example: for a "medicine bottle", keywords: ["bottle", "cup", "vase"].
            Example: for "slippers", keywords: ["shoe", "footwear"].
            This is critical for hybrid search.
        """.trimIndent()
    }

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val outputStream = ByteArrayOutputStream()
        val scale = 512f / maxOf(bitmap.width, bitmap.height)
        val resized = Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * scale).toInt(),
            (bitmap.height * scale).toInt(),
            true
        )
        resized.compress(Bitmap.CompressFormat.JPEG, 80, outputStream)
        return Base64.encodeToString(outputStream.toByteArray(), Base64.NO_WRAP)
    }

    fun release() {
        httpClient.close()
    }
    
    // Internal serialization classes
    @Serializable data class GeminiRequest(val contents: List<Content>, val generationConfig: GenerationConfig)
    @Serializable data class Content(val parts: List<Part>)
    @Serializable data class Part(val text: String? = null, @SerialName("inline_data") val inlineData: InlineData? = null)
    @Serializable data class InlineData(@SerialName("mime_type") val mimeType: String, val data: String)
    @Serializable data class GenerationConfig(val temperature: Float, val responseMimeType: String)
    @Serializable data class GeminiResponse(val candidates: List<Candidate>?)
    @Serializable data class Candidate(val content: Content?)
    
    @Serializable
    data class JsonLabelResponse(
        val label: String,
        val description: String,
        val keywords: List<String>
    )
}
