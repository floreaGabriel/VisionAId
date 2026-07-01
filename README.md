# VisionAId

**An offline-first, multimodal Android assistant for people with visual impairment.**

VisionAId turns a commodity Android smartphone into a real-time visual assistant.
It integrates six on-device deep-learning models (metric monocular depth, instance
segmentation, visual and facial embeddings, face detection, and a custom banknote
detector) running entirely through **ONNX Runtime**, with an optional cloud LLM
(**Google Gemini Flash**) used only for narrative scene description and automatic
object labeling. All critical functions work **fully offline**.

A distinctive feature is a **few-shot pipeline for personal objects**: the user
photographs an object from several angles, and the app later locates that specific
instance in the environment, guiding the user toward it with augmented-reality
markers, spatial audio, and distance-proportional haptics.

---

## ✨ Features

| Module | Description |
|---|---|
| **Camera (proximity)** | Real-time obstacle alerting from a **metric** depth map (calibrated to <1 cm error within 3 m), with three-level vibration/sound alerts. |
| **Scene description** | On-demand narrative description of the scene via Gemini Flash, delivered through Romanian text-to-speech. |
| **Personal objects** | Register a personal object from multiple angles (few-shot) and later find it via **AR search** with step-by-step guidance. |
| **Colors** | Deterministic HSV-based color identification (no ML, sub-millisecond latency). |
| **People** | Guided face registration and real-time recognition (YuNet + MobileFaceNet). |
| **Money** | Romanian banknote (RON) detection and counting with a custom-trained detector. |

All feedback is **multimodal**: Romanian speech synthesis, voice commands, and
graduated haptics — designed from the ground up for non-visual use.

---

## 🧱 Tech stack

- **Language / UI:** Kotlin 2.0, Jetpack Compose, Material Design 3
- **Camera:** CameraX
- **On-device inference:** ONNX Runtime
- **Augmented reality:** ARCore + SceneView
- **Persistence:** Room (embeddings stored as binary blobs)
- **Accessibility:** Android TTS (Romanian), voice-command manager, haptics
- **Cloud (optional):** Google Gemini Flash — scene description & object labeling only

**Requirements:** `minSdk 24` · `targetSdk 36` · `compileSdk 36` · ARCore-capable device recommended.

---

## 🤖 ML models

The models are **not** included in this repository (they are large binaries).
Place the following files in `app/src/main/assets/` before building:

| File | Purpose | Approx. size |
|---|---|---|
| `depth_anything_v2_metric_small_int8.onnx` | Metric monocular depth (proximity) | 35 MB |
| `mobileclip2_s2_visual.onnx` | Visual embeddings (object search) | 136 MB |
| `yolo11n-seg.onnx` | Segmentation + detection | 11 MB |
| `mobilefacenet.onnx` | Face embeddings | 13 MB |
| `face_detection_yunet_2023mar.onnx` | Face detection | 0.23 MB |
| `yolo26n_money.onnx` | RON banknote detector (custom) | 9.4 MB |
| `money_labels.txt` | Banknote class labels | included |

> The training and export scripts for these models live under [`scripts/`](scripts/).

---

## 🚀 Getting started

### Prerequisites
- Android Studio (latest stable)
- Android SDK 36
- A physical device (camera + ARCore); the emulator is not suitable for the ML/AR features

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/floreaGabriel/VisionAId.git
   cd VisionAId
   ```

2. Add the ML model files to `app/src/main/assets/` (see the table above).

3. Create a `local.properties` file in the project root:
   ```properties
   sdk.dir=/path/to/your/Android/sdk
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > `GEMINI_API_KEY` is injected at build time into `BuildConfig` and is only used
   > for the optional scene-description and object-labeling features. The app runs
   > without it; those two cloud features are simply disabled.

4. Build and run:
   ```bash
   ./gradlew installDebug
   ```

---

## 📁 Project structure

```
VisionAId/
├── app/                      # Android application
│   ├── src/main/java/...     # Kotlin sources (UI, ML managers, data)
│   ├── src/main/res/         # Resources
│   └── src/main/assets/      # ML models go here (not committed)
├── scripts/                  # Python training / export scripts for the ML models
├── gradle/                   # Gradle wrapper + version catalog
└── build.gradle.kts          # Root build configuration
```

---

## 🔐 Permissions

`CAMERA` · `RECORD_AUDIO` (voice commands) · `VIBRATE` (haptic feedback) ·
`INTERNET` (optional Gemini features only).

---

## 📄 License

Released for academic and research purposes. See the repository for details.

---

## 🎓 Citation

If you use this work, please cite the accompanying paper:

> C.-G. Florea, *VisionAId: An Offline-First Multimodal Android Assistant for
> People with Visual Impairment, Featuring Personalized Object Retrieval*, 2026.
