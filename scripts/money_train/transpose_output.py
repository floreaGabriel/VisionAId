# Transpune output YOLO money: [1, 12, 8400] -> [1, 8400, 12]
# Elimina necesitatea transpunerii pe telefon la fiecare frame
# pip install onnx

import os
import onnx
from onnx import helper, TensorProto

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "app", "src", "main", "assets")
INPUT_PATH = os.path.join(ASSETS_DIR, "yolo12n_money.onnx")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "yolo12n_money_transposed.onnx")

model = onnx.load(INPUT_PATH)
graph = model.graph

# Output original
original_output = graph.output[0]
original_name = original_output.name
print(f"Output original: {original_name}")

# Redenumim output-ul original (devine input pentru Transpose)
intermediate_name = original_name + "_raw"
original_output.name = intermediate_name

# Adaugam nod Transpose: perm [0, 2, 1] => [1, 12, 8400] -> [1, 8400, 12]
transpose_node = helper.make_node(
    "Transpose",
    inputs=[intermediate_name],
    outputs=[original_name + "_transposed"],
    perm=[0, 2, 1],
)
graph.node.append(transpose_node)

# Nou output cu shape [1, 8400, 12]
new_output = helper.make_tensor_value_info(
    original_name + "_transposed", TensorProto.FLOAT, [1, 8400, 12]
)

# Inlocuim output-ul vechi
del graph.output[:]
graph.output.append(new_output)

onnx.checker.check_model(model)
onnx.save(model, OUTPUT_PATH)

print(f"Output nou: [1, 8400, 12]")
print(f"Salvat: {OUTPUT_PATH}")
print(f"\ncp yolo12n_money_transposed.onnx {os.path.abspath(ASSETS_DIR)}/")
