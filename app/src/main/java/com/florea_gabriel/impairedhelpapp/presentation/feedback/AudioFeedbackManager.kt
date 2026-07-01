package com.florea_gabriel.impairedhelpapp.presentation.feedback

import android.media.AudioManager
import android.media.ToneGenerator
import kotlin.math.abs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Manages audio feedback for object search (Geiger-counter style).
 *
 * - Plays beeps with variable frequency based on how centered the object is.
 * - Center = Fast beeps (High urgency)
 * - Edge = Slow beeps (Low urgency)
 */
class AudioFeedbackManager {

    private var toneGenerator: ToneGenerator? = null
    private var isPlaying = false
    private var feedbackJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    init {
        try {
            toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 100)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Updates the audio feedback based on the object's position.
     * @param centerX 0.0 to 1.0 (relative to screen width)
     */
    fun updateFeedback(centerX: Float, found: Boolean) {
        if (!found) {
            stopFeedback()
            return
        }

        if (feedbackJob?.isActive == true) {
            // let the loop handle updates via a shared variable or just restart if dramatic change?
            // simpler: Just update a shared state that the loop reads
            activeCenterX = centerX
        } else {
            startFeedbackLoop()
        }
    }

    private var activeCenterX: Float = 0.5f

    private fun startFeedbackLoop() {
        if (isPlaying) return
        isPlaying = true

        feedbackJob =
                scope.launch {
                    while (isActive && isPlaying) {
                        // Calculate distance from center (0.5)
                        // Range: 0.0 (center) to 0.5 (edge)
                        val distFromCenter = abs(activeCenterX - 0.5f)

                        // Map distance to beep delay
                        // 0.0 (Center) -> 100ms delay (Fast)
                        // 0.5 (Edge) -> 800ms delay (Slow)
                        val beepDelay = 100L + (distFromCenter * 2 * 700).toLong()

                        // Play beep
                        toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP, 50)

                        delay(beepDelay)
                    }
                }
    }

    fun stopFeedback() {
        isPlaying = false
        feedbackJob?.cancel()
        feedbackJob = null
    }

    fun release() {
        stopFeedback()
        toneGenerator?.release()
        toneGenerator = null
    }
}
