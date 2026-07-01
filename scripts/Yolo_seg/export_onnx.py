from ultralytics import YOLO
import warnings
import sys
import os

# Suppress standard outputs to keep it clean if requested, 
# though Ultralytics is verbose by nature.
# This script simply loads the model and exports it.

def export_model():
    model_name = "yolo11n-seg.pt"
    
    # Check if a custom model path was provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    # Load the YOLO11 segmentation model
    # It will automatically download if not found locally
    model = YOLO(model_name)

    # Export to ONNX
    # imgsz=640 matches our Android implementation
    # opset=12 is widely compatible with ONNX Runtime on Android
    model.export(format="onnx", imgsz=640, opset=12)

if __name__ == "__main__":
    export_model()
