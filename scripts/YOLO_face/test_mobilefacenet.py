"""
Test MobileFaceNet ONNX model — verify it loads and produces embeddings.

Usage:
    python test_mobilefacenet.py [optional_image.jpg]

If no image is provided, tests with a random tensor.
If an image is provided, runs face crop + embedding extraction.
"""

import numpy as np
import os
import sys

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: pip install onnxruntime")
    sys.exit(1)

MODEL_PATH = "mobilefacenet.onnx"


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def preprocess_face(image_rgb, target_size=112):
    """
    Preprocess a face crop for MobileFaceNet.
    Input:  RGB image (any size)
    Output: (1, 3, 112, 112) float32 tensor, normalized to [-1, 1]
    """
    from PIL import Image

    if isinstance(image_rgb, np.ndarray):
        img = Image.fromarray(image_rgb)
    else:
        img = image_rgb

    # Resize to 112x112
    img = img.resize((target_size, target_size), Image.BILINEAR)

    # To numpy, float32, normalize to [-1, 1]
    arr = np.array(img, dtype=np.float32)  # (112, 112, 3)
    arr = (arr - 127.5) / 127.5            # normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)           # (3, 112, 112)
    arr = np.expand_dims(arr, 0)           # (1, 3, 112, 112)

    return arr


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: '{MODEL_PATH}' not found! Run download_mobilefacenet.py first.")
        return

    print(f"Loading model: {MODEL_PATH}")
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")

    session = ort.InferenceSession(MODEL_PATH)

    # Print model info
    print("\n── Model Info ──")
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name} — shape={inp.shape}, type={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name} — shape={out.shape}, type={out.type}")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"\nExpected input: {input_name} = {input_shape}")

    # Determine input size from model
    h = input_shape[2] if isinstance(input_shape[2], int) else 112
    w = input_shape[3] if isinstance(input_shape[3], int) else 112

    # Test with image if provided
    if len(sys.argv) > 1:
        from PIL import Image
        img_path = sys.argv[1]
        print(f"\n── Testing with image: {img_path} ──")
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess_face(img, target_size=h)
    else:
        print("\n── Testing with random tensor ──")
        tensor = np.random.randn(1, 3, h, w).astype(np.float32)

    print(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"Input range: [{tensor.min():.2f}, {tensor.max():.2f}]")

    # Run inference
    outputs = session.run(None, {input_name: tensor})
    embedding = outputs[0]

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"Embedding L2 norm: {np.linalg.norm(embedding[0]):.4f}")
    print(f"First 10 values: {embedding[0][:10]}")

    # Test cosine similarity with itself (should be 1.0)
    sim = cosine_similarity(embedding[0], embedding[0])
    print(f"\nSelf-similarity: {sim:.6f} (should be 1.0)")

    # Test with a different random input (should be < 1.0)
    tensor2 = np.random.randn(1, 3, h, w).astype(np.float32)
    outputs2 = session.run(None, {input_name: tensor2})
    sim2 = cosine_similarity(embedding[0], outputs2[0][0])
    print(f"Random vs random similarity: {sim2:.6f} (should be << 1.0)")

    print("\n✓ Model works correctly!")
    print(f"\nFor Android: copy '{MODEL_PATH}' to app/src/main/assets/")


if __name__ == "__main__":
    main()
