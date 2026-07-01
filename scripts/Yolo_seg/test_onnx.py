import onnxruntime as ort
import numpy as np
import cv2
import sys
import os

def test_model(image_path, model_path):
    print(f"Loading model: {model_path}")
    print(f"Loading image: {image_path}")

    # Load ONNX model
    session = ort.InferenceSession(model_path)
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input Name: {input_name}")
    print(f"Input Shape: {input_shape}")  # e.g. [1, 3, 640, 640]
    
    # Load and Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    
    # Normalize 0-1 and Transpose to BCHW
    input_data = img_resized.astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1) # HWC -> CHW
    input_data = np.expand_dims(input_data, axis=0) # Add Batch dim

    # Inference
    outputs = session.run(None, {input_name: input_data})
    
    # Analyze Output 0 (Detections)
    output0 = outputs[0]
    print(f"Output 0 Shape: {output0.shape}")
    
    # Check if it's [1, 116, 8400] or [1, 8400, 116]
    dims = output0.shape
    num_channels = dims[1]
    num_anchors = dims[2]
    
    print(f"Dimensions: Batch={dims[0]}, Channels={num_channels}, Anchors={num_anchors}")

    if num_channels == 116:
        print("Format appears to be [Batch, Channels, Anchors] (Standard YOLO11 export?)")
        # Logic matches Android code: detections[4 + c][i]
        
        # Let's inspect raw scores for class 0 (Person) or verify max scores
        count = 0
        for i in range(num_anchors):
            # Box: 0-3, Classes: 4-83, Masks: 84-115
            scores = output0[0, 4:84, i]
            max_score = np.max(scores)
            if max_score > 0.25:
                class_id = np.argmax(scores)
                print(f"Anchor {i}: Class {class_id}, Score {max_score:.4f}")
                count += 1
        print(f"Total detections > 0.25: {count}")
        
    elif num_channels > 8000: # [1, 8400, 116]
        print("Format appears to be [Batch, Anchors, Channels] (TRANSPOSED)")
        print("⚠️ ANDROID CODE WILL FAIL WITHOUT TRANSPOSE")
        
    else:
        print("Unknown format.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_onnx.py <image_path> <model_path>")
    else:
        test_model(sys.argv[1], sys.argv[2])
