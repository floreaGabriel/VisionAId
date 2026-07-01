# Quantizare INT8 dinamica — Depth Anything V2 (~100 MB -> ~25 MB)
# Transformer-based => MatMul/Gemm sunt majoritari => reducere ~75%
# pip install onnxruntime onnx

import os
import shutil
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "app", "src", "main", "assets")
ONNX_FILE = os.path.join(ASSETS, "depth_anything_v2_metric_small.onnx")
DATA_FILE = os.path.join(ASSETS, "depth_anything_v2_metric_small.onnx.data")
WORK_DIR = os.path.dirname(__file__)

# Copiem modelul local (are nevoie de .onnx + .onnx.data in acelasi dir)
local_onnx = os.path.join(WORK_DIR, "depth_anything_v2_metric_small.onnx")
local_data = os.path.join(WORK_DIR, "depth_anything_v2_metric_small.onnx.data")
if not os.path.exists(local_onnx):
    shutil.copy2(ONNX_FILE, local_onnx)
    shutil.copy2(DATA_FILE, local_data)

OUTPUT_PATH = os.path.join(WORK_DIR, "depth_anything_v2_metric_small_int8.onnx")

# Verificare
model = onnx.load(local_onnx)
onnx.checker.check_model(model)
input_mb = (os.path.getsize(local_onnx) + os.path.getsize(local_data)) / (1024 * 1024)
print(f"Original: {input_mb:.1f} MB")

# Quantizare dinamica: greutati INT8, activari FP32
quantize_dynamic(
    model_input=local_onnx,
    model_output=OUTPUT_PATH,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)

output_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"INT8:     {output_mb:.1f} MB  ({(1 - output_mb / input_mb) * 100:.0f}% reducere)")
print(f"\nCopiaza in assets:\n  cp depth_anything_v2_metric_small_int8.onnx {os.path.abspath(ASSETS)}/")
