import onnxruntime as ort
import numpy as np
import cv2
import sys

"""
FastSAM ONNX Test Script
This script verifies the correct mask post-processing for FastSAM.
It will show what the mask SHOULD look like when processed correctly.
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_fastsam(image_path, model_path):
    print(f"Loading: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Input info
    input_name = session.get_inputs()[0].name
    print(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    orig_h, orig_w = img.shape[:2]
    print(f"Original image size: {orig_w}x{orig_h}")
    
    # Preprocess - letterbox to 640x640
    input_size = 640
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Create padded image (gray background)
    img_padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    
    print(f"Scale: {scale}, pad_x: {pad_x}, pad_y: {pad_y}")
    
    # Normalize and prepare tensor
    img_norm = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Batch
    
    # Inference
    outputs = session.run(None, {input_name: input_tensor})
    
    output0 = outputs[0]  # [1, 37, 8400]
    output1 = outputs[1]  # [1, 32, 160, 160]
    
    print(f"Output 0 shape: {output0.shape}")
    print(f"Output 1 shape: {output1.shape}")
    
    detections = output0[0]  # [37, 8400]
    protos = output1[0]      # [32, 160, 160]
    
    num_anchors = detections.shape[1]
    
    # Find best candidate (highest score, closest to center)
    center_x = input_size / 2
    center_y = input_size / 2
    
    best_score = 0
    best_idx = -1
    best_dist = float('inf')
    
    candidates = []
    
    for i in range(num_anchors):
        score = detections[4, i]  # Confidence at index 4
        
        if score > 0.25:
            cx = detections[0, i]
            cy = detections[1, i]
            w = detections[2, i]
            h = detections[3, i]
            
            box_cx = cx
            box_cy = cy
            
            dist = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
            
            candidates.append({
                'idx': i,
                'score': score,
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'dist': dist,
                'mask_coeffs': detections[5:37, i]  # Last 32 coefficients
            })
    
    if not candidates:
        print("No candidates found!")
        return
    
    print(f"Found {len(candidates)} candidates")
    
    # Sort by distance to center (closest first)
    candidates.sort(key=lambda x: x['dist'])
    best = candidates[0]
    
    print(f"Best candidate: score={best['score']:.3f}, center=({best['cx']:.1f}, {best['cy']:.1f}), dist={best['dist']:.1f}")
    
    # Generate mask
    mask_coeffs = best['mask_coeffs']  # [32]
    
    # MatMul: [32] * [32, 160, 160] -> [160, 160]
    mask_160 = np.zeros((160, 160), dtype=np.float32)
    for h in range(160):
        for w in range(160):
            mask_160[h, w] = np.sum(mask_coeffs * protos[:, h, w])
    
    # Apply sigmoid
    mask_160 = sigmoid(mask_160)
    
    # Save raw 160x160 mask for debugging
    cv2.imwrite("debug_mask_160.jpg", (mask_160 * 255).astype(np.uint8))
    print("Saved: debug_mask_160.jpg (raw 160x160 mask)")
    
    # ======= CRITICAL: Correct Upscaling =======
    # The mask is at 160x160 which corresponds to the 640x640 padded input
    # We need to:
    # 1. Resize mask to 640x640 (same as padded input)
    # 2. Crop out the padded regions
    # 3. Resize to original image size
    
    # Step 1: Resize 160x160 mask to 640x640 (input space)
    mask_640 = cv2.resize(mask_160, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Crop out the actual image region (remove padding)
    mask_cropped = mask_640[pad_y:pad_y+new_h, pad_x:pad_x+new_w]
    
    # Step 3: Resize to original image dimensions
    mask_orig = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Threshold
    mask_binary = (mask_orig > 0.5).astype(np.uint8) * 255
    
    cv2.imwrite("debug_mask_binary.jpg", mask_binary)
    print("Saved: debug_mask_binary.jpg (final binary mask)")
    
    # Apply mask to original image
    mask_3ch = np.stack([mask_binary, mask_binary, mask_binary], axis=-1) / 255.0
    result = (img * mask_3ch).astype(np.uint8)
    
    cv2.imwrite("debug_result.jpg", result)
    print("Saved: debug_result.jpg (object with black background)")
    
    # Also save visualization with red overlay
    overlay = img.copy()
    overlay[mask_binary > 0] = overlay[mask_binary > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    cv2.imwrite("debug_overlay.jpg", overlay.astype(np.uint8))
    print("Saved: debug_overlay.jpg (mask overlay)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_fastsam.py <image.jpg> <FastSam-s.onnx>")
    else:
        test_fastsam(sys.argv[1], sys.argv[2])
