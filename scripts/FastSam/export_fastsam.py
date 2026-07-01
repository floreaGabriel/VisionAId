from ultralytics import FastSAM
import sys
import os

# Script to export FastSAM to ONNX for Android
# Usage: python export_fastsam.py [optional_model_path]

def export_fastsam():
    model_name = "FastSam-s.pt"
    
    # Check if a custom model path was provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    print(f"Loading FastSAM model: {model_name}...")
    # Load the FastSAM model
    model = FastSAM(model_name)

    print("Exporting to ONNX...")
    # Export to ONNX
    # imgsz=640 is standard for mobile
    # opset=12 for compatibility
    model.export(format="onnx", imgsz=640, opset=12)
    
    print("✅ Export complete: FastSAM-s.onnx")

if __name__ == "__main__":
    export_fastsam()
