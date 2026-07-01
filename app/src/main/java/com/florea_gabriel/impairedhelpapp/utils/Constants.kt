package com.florea_gabriel.impairedhelpapp.utils

/**
 * Application-wide constants
 */
object Constants {

    // Money Model configuration (ONNX Runtime)
    const val MONEY_MODEL_FILE = "yolo26n_money.onnx"
    const val MONEY_LABELS_FILE = "money_labels.txt"
    const val MONEY_INPUT_SIZE = 640

    // Depth Model configuration (ONNX Runtime)
    const val DEPTH_INPUT_SIZE = 384   // ONNX model fixed input size
    const val DEPTH_OUTPUT_SIZE = 378  // ONNX model fixed output size
    const val DEPTH_MODEL_FILE = "depth_anything_v2_metric_small_int8.onnx"

    // Detection thresholds
    const val CONFIDENCE_THRESHOLD = 0.25f
    const val IOU_THRESHOLD = 0.45f

    // Distance estimation configuration
    // DEPTH_CALIBRATION_FACTOR: Multiply raw model output to get accurate distance
    // If model shows 3m but real distance is 2.4m -> factor = 2.4/3 = 0.8
    // Adjust this value based on your real-world measurements!
    const val DEPTH_CALIBRATION_FACTOR = 0.55f  // Calibrated: model 3m -> real ~2.4m

    const val VERY_CLOSE_THRESHOLD = 1.0f
    const val CLOSE_THRESHOLD = 2.0f
    const val MEDIUM_THRESHOLD = 5.0f

    // Camera configuration
    const val NUM_THREADS = 4

    // UI configuration
    const val LABEL_TEXT_SIZE = 40f
    const val BOUNDING_BOX_STROKE_WIDTH = 8f
    const val LABEL_BACKGROUND_HEIGHT = 50f
    const val LABEL_PADDING = 10f

    // Performance - Base targets (adjusted by DeviceCapabilityDetector)
    const val MIN_FPS_TARGET = 10.0f
    const val TARGET_FPS_WITH_DEPTH = 7.0f  // Lower due to depth estimation overhead

    // Performance tier FPS targets
    const val FPS_TARGET_HIGH_TIER = 15.0f    // Modern flagships with NPU (S21, Pixel 6+)
    const val FPS_TARGET_MEDIUM_TIER = 10.0f  // Mid-range devices
    const val FPS_TARGET_LOW_TIER = 7.0f      // Older/budget devices

    // Depth estimation configuration
    const val DEPTH_SKIP_FRAMES_HIGH_TIER = 1    // Process every frame on modern devices
    const val DEPTH_SKIP_FRAMES_MEDIUM_TIER = 2  // Process every 2nd frame
    const val DEPTH_SKIP_FRAMES_LOW_TIER = 3     // Process every 3rd frame
}
