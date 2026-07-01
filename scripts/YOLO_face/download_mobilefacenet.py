import os
import sys

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("pip install insightface onnxruntime")
    sys.exit(1)

print("Downloading buffalo_sc (16MB, contains MobileFaceNet)...")
app = FaceAnalysis(name="buffalo_sc", allowed_modules=["recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Find the downloaded model
home = os.path.expanduser("~")
model_dir = os.path.join(home, ".insightface", "models", "buffalo_sc")
print(f"\nModels downloaded to: {model_dir}")
print("Files:")
for f in os.listdir(model_dir):
    size = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
    print(f"  {f} ({size:.1f} MB)")

# Copy the recognition model (w600k_mbf.onnx) to current dir
import shutil
src = os.path.join(model_dir, "w600k_mbf.onnx")
dst = "mobilefacenet.onnx"
if os.path.exists(src):
    shutil.copy2(src, dst)
    size = os.path.getsize(dst) / (1024 * 1024)
    print(f"\nCopied: {dst} ({size:.1f} MB)")
    print("Done! This is the MobileFaceNet model for face embeddings.")
else:
    print(f"\nw600k_mbf.onnx not found in {model_dir}")
    print("Available files — copy the recognition .onnx manually:")
    for f in os.listdir(model_dir):
        print(f"  {f}")
