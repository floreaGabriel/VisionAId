# Conversie FP32 -> FP16 — MobileCLIP2-S2 (~136 MB -> ~68 MB)
# pip install onnx onnxconverter-common

import os
from onnxconverter_common import float16
import onnx

INPUT_PATH = os.path.join(os.path.dirname(__file__), "mobileclip2_s2_visual.onnx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "mobileclip2_s2_visual_fp16.onnx")

model = onnx.load(INPUT_PATH)

# Pastram input/output in FP32, convertim doar greutatile interne
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,  # input FP32, output FP32
)
onnx.save(model_fp16, OUTPUT_PATH)

input_mb = os.path.getsize(INPUT_PATH) / (1024 * 1024)
output_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"FP32: {input_mb:.1f} MB")
print(f"FP16: {output_mb:.1f} MB  ({(1 - output_mb / input_mb) * 100:.0f}% reducere)")
print(f"\ncp mobileclip2_s2_visual_fp16.onnx ../../app/src/main/assets/")
