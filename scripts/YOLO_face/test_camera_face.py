import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("YOLO_face.onnx")
inp = session.get_inputs()[0]
print(f"Input: {inp.name} shape={inp.shape}")

CONF_THRESH = 0.5
IOU_THRESH = 0.5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

    preds = session.run(None, {inp.name: blob})[0][0]  # (5, 8400)

    mask = preds[4] > CONF_THRESH
    filtered = preds[:, mask].T  # (N, 5)

    if len(filtered) > 0:
        # Scale to frame coords
        cx = filtered[:, 0] / 640 * w
        cy = filtered[:, 1] / 640 * h
        bw = filtered[:, 2] / 640 * w
        bh = filtered[:, 3] / 640 * h
        confs = filtered[:, 4]

        x1 = (cx - bw / 2).astype(int)
        y1 = (cy - bh / 2).astype(int)
        x2 = (cx + bw / 2).astype(int)
        y2 = (cy + bh / 2).astype(int)

        # NMS wants list of [x, y, w, h]
        nms_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(filtered))]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), CONF_THRESH, IOU_THRESH).flatten()

        for idx in indices:
            cv2.rectangle(frame, (x1[idx], y1[idx]), (x2[idx], y2[idx]), (0, 255, 0), 2)
            cv2.putText(frame, f"face {confs[idx]:.2f}", (x1[idx], y1[idx] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(indices)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
