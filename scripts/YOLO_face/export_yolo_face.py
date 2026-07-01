"""
Export YOLO-Face .pt → ONNX + TFLite (float16) for Android deployment.

Usage:
    python export_yolo_face.py

Outputs:
    - YOLO Face.onnx          (for ONNX Runtime on Android)
    - YOLO Face_float16.tflite (for LiteRT/TFLite on Android)

Notes:
    - imgsz=640 matches standard YOLO input
    - ONNX opset=12 for broad ONNX Runtime compatibility
    - TFLite float16 = good speed/accuracy tradeoff on mobile
"""

from ultralytics import YOLO
import os

MODEL_PATH = "YOLO Face.pt"
IMG_SIZE = 640

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: '{MODEL_PATH}' not found in current directory!")
        return

    model = YOLO(MODEL_PATH)

    # Print model info
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Task: {model.task}")
    print(f"Classes: {model.names}")
    print(f"Export image size: {IMG_SIZE}")
    print("=" * 60)

    # ── Export ONNX ──────────────────────────────────────────────
    print("\n[1/2] Exporting to ONNX...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,
        simplify=True,    # onnx-simplifier for smaller/faster model
        half=False,        # ONNX doesn't support fp16 weights natively in opset12
    )
    print(f"  → ONNX saved: {onnx_path}")

    # ── Export TFLite float16 ────────────────────────────────────
    print("\n[2/2] Exporting to TFLite (float16)...")
    tflite_path = model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        half=True,         # float16 quantization
    )
    print(f"  → TFLite saved: {tflite_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE! Files ready for Android assets/:")
    if onnx_path and os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX:   {onnx_path} ({size_mb:.1f} MB)")
    if tflite_path and os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  TFLite: {tflite_path} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
