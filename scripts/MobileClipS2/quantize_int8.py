# Quantizare INT8 dinamica — MobileCLIP2-S2 (~143 MB -> ~36 MB)
# pip install onnxruntime onnx

import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_PATH = os.path.join(os.path.dirname(__file__), "mobileclip2_s2_visual.onnx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "mobileclip2_s2_visual_int8.onnx")

# Verificare model original
onnx.checker.check_model(onnx.load(INPUT_PATH))
input_mb = os.path.getsize(INPUT_PATH) / (1024 * 1024)
print(f"Original: {input_mb:.1f} MB")

# Quantizare dinamica: greutati INT8, activari FP32
quantize_dynamic(
    model_input=INPUT_PATH,
    model_output=OUTPUT_PATH,
    weight_type=QuantType.QInt8,
    # Conv exclus — ConvInteger nu e suportat de ONNX Runtime pe Android
    op_types_to_quantize=["MatMul", "Gemm"],
)

# Rezultat
onnx.checker.check_model(onnx.load(OUTPUT_PATH))
output_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"INT8:     {output_mb:.1f} MB  ({(1 - output_mb / input_mb) * 100:.0f}% reducere)")
print(f"\nCopiaza in assets:\n  cp mobileclip2_s2_visual_int8.onnx ../../app/src/main/assets/")
