package com.florea_gabriel.impairedhelpapp.utils

import android.content.Context
import android.media.AudioAttributes
import android.media.SoundPool
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import com.florea_gabriel.impairedhelpapp.R

/**
 * ProximityFeedback: Handles vibration and audio alerts for proximity warnings.
 *
 * Provides haptic and audio feedback when objects are detected within a certain distance.
 * Uses different intensity levels based on distance:
 * - < 0.5m: Strong vibration + rapid beep (DANGER)
 * - 0.5m - 1.0m: Medium vibration + normal beep (WARNING)
 * - 1.0m - 1.5m: Light vibration + slow beep (CAUTION)
 *
 * Includes debounce mechanism to prevent overwhelming the user with constant feedback.
 *
 * @param context Application context for accessing system services
 */
class ProximityFeedback(private val context: Context) {

    companion object {
        private const val TAG = "ProximityFeedback"

        // Distance thresholds in meters
        const val THRESHOLD_DANGER = 0.5f      // Very close
        const val THRESHOLD_WARNING = 1.0f     // Close
        const val THRESHOLD_CAUTION = 1.5f     // Approaching

        // Debounce time in milliseconds (minimum time between alerts)
        private const val DEBOUNCE_DANGER = 300L
        private const val DEBOUNCE_WARNING = 500L
        private const val DEBOUNCE_CAUTION = 800L
    }

    // Vibrator instance (compatible with API 26+)
    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    // SoundPool for audio alerts
    private var soundPool: SoundPool? = null
    private var beepSoundId: Int = 0
    private var isSoundLoaded = false

    // Debounce tracking
    private var lastAlertTime = 0L
    private var lastAlertLevel: AlertLevel? = null

    // Alert levels
    enum class AlertLevel {
        DANGER,   // < 0.5m
        WARNING,  // 0.5m - 1.0m
        CAUTION   // 1.0m - 1.5m
    }

    init {
        initSoundPool()
    }

    /**
     * Initialize SoundPool for audio playback
     */
    private fun initSoundPool() {
        try {
            soundPool = SoundPool.Builder()
                .setMaxStreams(3)
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .build()

            soundPool?.setOnLoadCompleteListener { _, _, status ->
                if (status == 0) {
                    isSoundLoaded = true
                    Log.d(TAG, "Beep sound loaded successfully")
                } else {
                    Log.e(TAG, "Failed to load beep sound")
                }
            }

            // Load beep sound from raw resources
            beepSoundId = soundPool?.load(context, R.raw.beep_warning, 1) ?: 0
            Log.d(TAG, "ProximityFeedback initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing SoundPool: ${e.message}")
        }
    }

    /**
     * Trigger proximity warning based on distance.
     *
     * @param distance Distance to object in meters
     * @param enableVibration Whether to enable vibration feedback
     * @param enableSound Whether to enable audio feedback
     */
    fun triggerWarning(
        distance: Float,
        enableVibration: Boolean = true,
        enableSound: Boolean = true
    ) {
        val alertLevel = when {
            distance < THRESHOLD_DANGER -> AlertLevel.DANGER
            distance < THRESHOLD_WARNING -> AlertLevel.WARNING
            distance < THRESHOLD_CAUTION -> AlertLevel.CAUTION
            else -> return // No alert needed
        }

        val currentTime = System.currentTimeMillis()
        val debounceTime = getDebounceTime(alertLevel)

        // Check debounce - skip if same level alert was triggered recently
        if (lastAlertLevel == alertLevel &&
            (currentTime - lastAlertTime) < debounceTime) {
            return
        }

        // Update tracking
        lastAlertTime = currentTime
        lastAlertLevel = alertLevel

        // Trigger feedback
        if (enableVibration) {
            triggerVibration(alertLevel)
        }
        if (enableSound && isSoundLoaded) {
            triggerSound(alertLevel)
        }

        Log.d(TAG, "Alert triggered: $alertLevel at distance ${String.format("%.2f", distance)}m")
    }

    /**
     * Trigger vibration based on alert level
     */
    private fun triggerVibration(level: AlertLevel) {
        if (!vibrator.hasVibrator()) {
            Log.w(TAG, "Device does not have vibrator")
            return
        }

        val effect = when (level) {
            AlertLevel.DANGER -> {
                // Strong, long vibration pattern for danger
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    VibrationEffect.createPredefined(VibrationEffect.EFFECT_HEAVY_CLICK)
                } else {
                    VibrationEffect.createOneShot(200, 255)
                }
            }
            AlertLevel.WARNING -> {
                // Medium vibration for warning
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    VibrationEffect.createPredefined(VibrationEffect.EFFECT_CLICK)
                } else {
                    VibrationEffect.createOneShot(150, 180)
                }
            }
            AlertLevel.CAUTION -> {
                // Light vibration for caution
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK)
                } else {
                    VibrationEffect.createOneShot(100, 100)
                }
            }
        }

        try {
            vibrator.vibrate(effect)
        } catch (e: Exception) {
            Log.e(TAG, "Error triggering vibration: ${e.message}")
        }
    }

    /**
     * Trigger sound based on alert level
     */
    private fun triggerSound(level: AlertLevel) {
        val (volume, rate) = when (level) {
            AlertLevel.DANGER -> 1.0f to 1.5f   // Loud, fast
            AlertLevel.WARNING -> 0.7f to 1.0f  // Medium
            AlertLevel.CAUTION -> 0.4f to 0.8f  // Quiet, slow
        }

        try {
            soundPool?.play(
                beepSoundId,
                volume,  // left volume
                volume,  // right volume
                1,       // priority
                0,       // loop (0 = no loop)
                rate     // playback rate
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error playing sound: ${e.message}")
        }
    }

    /**
     * Get debounce time for alert level
     */
    private fun getDebounceTime(level: AlertLevel): Long {
        return when (level) {
            AlertLevel.DANGER -> DEBOUNCE_DANGER
            AlertLevel.WARNING -> DEBOUNCE_WARNING
            AlertLevel.CAUTION -> DEBOUNCE_CAUTION
        }
    }

    /**
     * Check if feedback should be triggered for given distance
     */
    fun shouldTriggerAlert(distance: Float): Boolean {
        return distance < THRESHOLD_CAUTION
    }

    /**
     * Release resources. Call this when done using ProximityFeedback.
     */
    fun release() {
        try {
            soundPool?.release()
            soundPool = null
            isSoundLoaded = false
            Log.d(TAG, "ProximityFeedback resources released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing resources: ${e.message}")
        }
    }
}
