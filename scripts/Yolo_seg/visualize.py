import onnxruntime as ort
import numpy as np
import cv2
import sys
import math

# COCO Labels (Partial list for debugging)
def get_label(class_id):
    labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return str(class_id)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_mask(protos, mask_coeffs, detection_box, img_w, img_h, mask_size=160):
    """
    Generate mask from prototypes and coefficients
    protos: [32, 160, 160]
    mask_coeffs: [32]
    """
    # 1. Matrix Multiplication: [1, 32] * [32, 160*160] -> [1, 160*160]
    c, mh, mw = protos.shape
    protos_flat = protos.reshape(c, -1)
    mask_raw = np.matmul(mask_coeffs, protos_flat)
    mask_raw = sigmoid(mask_raw).reshape(mh, mw)
    
    # 2. Resize mask to image size (640x640)
    # Note: simple resize for visualization
    mask_resized = cv2.resize(mask_raw, (img_w, img_h))
    
    # 3. Crop to bounding box (optional, but typical for YOLO)
    # x1, y1, x2, y2 = detection_box.astype(int)
    # mask_final = np.zeros_like(mask_resized)
    # mask_final[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
    
    # 4. Binary threshold
    return (mask_resized > 0.5).astype(np.uint8) * 255

def visualize_segmentation(image_path, model_path):
    print(f"Vizualizare: {image_path}")
    
    # Load Model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Eroare la incarcarea imaginii")
        return
        
    orig_h, orig_w = img.shape[:2]
    
    # Preprocess
    # Letterbox resize to 640x640 with gray padding
    input_size = 640
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    img_padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dx = (input_size - new_w) // 2
    dy = (input_size - new_h) // 2
    img_padded[dy:dy+new_h, dx:dx+new_w] = img_resized
    
    # DEBUG: Show preprocessed input
    cv2.imwrite("debug_input.jpg", img_padded)
    
    img_norm = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1)) # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0) # Batch
    
    # Run Inference
    outputs = session.run(None, {input_name: input_tensor})
    
    output0 = outputs[0] # [1, 116, 8400]
    output1 = outputs[1] # [1, 32, 160, 160] (Protos)
    
    print(f"Output Shapes: {output0.shape} (Dets), {output1.shape} (Protos)")
    
    detections = output0[0]       # [116, 8400]
    protos = output1[0]           # [32, 160, 160]
    
    num_anchors = detections.shape[1]
    
    # Find Best Candidate
    best_score = 0
    best_idx = -1
    best_class = -1
    
    for i in range(num_anchors):
        # Classes are 4..83
        scores = detections[4:84, i]
        max_class_score = np.max(scores)
        
        if max_class_score > best_score:
            best_score = max_class_score
            best_idx = i
            best_class = np.argmax(scores)
            
    if best_score < 0.25:
        print("⚠️ Nu am gasit niciun obiect cu scor > 0.25")
        return

    print(f"✅ Cel mai bun candidat: Index={best_idx}, Class={get_label(best_class)} ({best_class}), Score={best_score:.4f}")
    
    # Process Best Candidate
    x, y, w, h = detections[0:4, best_idx]
    mask_coeffs = detections[84:116, best_idx] # Last 32
    
    print(f"Box: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
    
    # Draw Box on Padded Image
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    
    cv2.rectangle(img_padded, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_padded, f"{get_label(best_class)}: {best_score:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Generate Mask
    mask = process_mask(protos, mask_coeffs, np.array([x1, y1, x2, y2]), 640, 640)
    
    # Overlay Mask (Red)
    colored_mask = np.zeros_like(img_padded)
    colored_mask[:, :, 2] = mask # Red channel
    
    result = cv2.addWeighted(img_padded, 1.0, colored_mask, 0.5, 0)
    
    # Save results
    cv2.imwrite("result_visualization.jpg", result)
    
    # Also save the "segmented" object (transparent bg style)
    # Apply mask to original pixels
    rgba = cv2.cvtColor(img_padded, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask # Set alpha to mask
    cv2.imwrite("result_cutout.png", rgba)
    
    print("✅ Salvat: result_visualization.jpg si result_cutout.png")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <image> <model>")
    else:
        visualize_segmentation(sys.argv[1], sys.argv[2])
