"""
Export money_model.pt → ONNX (float16) for Android/ONNX Runtime Mobile.

Usage:
    python export_onnx.py                          # defaults
    python export_onnx.py --imgsz 640 --half       # explicit
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO money model to ONNX")
    parser.add_argument("--weights", default="yolo12n_money.pt", help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", default=True, help="Export FP16 (default)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Export FP32")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = YOLO(args.weights)
    print(f"Classes: {model.names}")
    print(f"Exporting to ONNX (imgsz={args.imgsz}, half={args.half}, opset={args.opset})...")

    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify,
        opset=args.opset,
    )

    print(f"\nExported: {export_path}")
    print("Copy this .onnx file to app/src/main/assets/")


if __name__ == "__main__":
    main()
