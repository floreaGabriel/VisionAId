"""
Export YOLO11s COCO to ONNX (FP16) for standardization with the rest of
VisionAId's ML stack (all other models already use ONNX Runtime).

Output: yolo11s_fp16_coco.onnx (~19 MB, equivalent in size to the current
TFLite Float16 model but using the same inference runtime as the other models).

Usage:
    cd scripts/YOLO_COCO
    python export_yolo_to_onnx.py

Requirements:
    pip install ultralytics onnx onnxruntime onnxconverter-common
"""

from pathlib import Path
import shutil

from ultralytics import YOLO
import onnx
from onnxconverter_common import float16

SRC_WEIGHTS = "Ultralytics YOLO 11s.pt"
IMGSZ = 640
OPSET = 17
FP32_OUT = "yolo11s_fp32_coco.onnx"
FP16_OUT = "yolo11s_fp16_coco.onnx"


def export_fp32() -> Path:
    model = YOLO(SRC_WEIGHTS)
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=True,
        dynamic=False,
        half=False,
    )
    exported = Path(exported)
    target = Path(FP32_OUT)
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    return target


def convert_to_fp16(fp32_path: Path) -> Path:
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # keep input/output as FP32 so Kotlin code stays unchanged
        disable_shape_infer=False,
    )
    target = Path(FP16_OUT)
    onnx.save(model_fp16, str(target))
    return target


def main():
    print(f"[1/2] Exporting {SRC_WEIGHTS} to ONNX FP32 (opset {OPSET}, imgsz {IMGSZ})...")
    fp32 = export_fp32()
    print(f"      -> {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    print("[2/2] Converting to FP16 (keeping FP32 I/O)...")
    fp16 = convert_to_fp16(fp32)
    print(f"      -> {fp16} ({fp16.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. Copy the FP16 file into the app's assets folder:")
    print(f"  cp {fp16} ../../app/src/main/assets/{fp16.name}")
    print("\nThen update Constants.kt and the YOLO loader to use ONNX Runtime")
    print("instead of LiteRT for this model.")


if __name__ == "__main__":
    main()
