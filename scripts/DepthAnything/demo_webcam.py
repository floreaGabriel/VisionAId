import cv2
import numpy as np
import onnxruntime as ort

# Incarca modelul
session = ort.InferenceSession("depth_anything_v2_metric_small.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 384, 384]
H, W = input_shape[2], input_shape[3]

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

cap = cv2.VideoCapture(0)
print(f"Model input: {H}x{W}, output: 378x378")
print("Apasa 'q' pentru a iesi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess: resize, normalize, NCHW
    img = cv2.resize(frame, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # -> NCHW

    # Inferenta
    output = session.run(None, {input_name: img})[0]
    depth = output.squeeze()  # 378x378, valori in metri

    # Heatmap: normalizeaza la 0-255, coloreaza
    d_min, d_max = depth.min(), depth.max()
    depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Text cu range-ul de distante
    cv2.putText(heatmap, f"Min: {d_min:.2f}m  Max: {d_max:.2f}m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Afiseaza side by side
    frame_resized = cv2.resize(frame, (heatmap.shape[1], heatmap.shape[0]))
    combined = np.hstack([frame_resized, heatmap])
    cv2.imshow("Camera | Depth Map (metri)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
