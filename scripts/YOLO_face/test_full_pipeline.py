import cv2
import numpy as np
import onnxruntime as ort

face_det = ort.InferenceSession("YOLO_face.onnx")
face_rec = ort.InferenceSession("mobilefacenet.onnx")

det_input = face_det.get_inputs()[0].name
rec_input = face_rec.get_inputs()[0].name
rec_out_shape = face_rec.get_outputs()[0].shape
print(f"Face det input: {face_det.get_inputs()[0].shape}")
print(f"Face rec input: {face_rec.get_inputs()[0].shape} -> output: {rec_out_shape}")

CONF_THRESH = 0.5
IOU_THRESH = 0.5

saved_embeddings = {}  # name -> embedding
mode = "recognize"  # "recognize" or "register"
register_name = ""

def get_embedding(frame, x1, y1, x2, y2):
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
    face_crop = frame[y1c:y2c, x1c:x2c]
    if face_crop.size == 0:
        return None
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    blob = ((face_resized.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
    emb = face_rec.run(None, {rec_input: blob})[0][0]
    emb = emb / np.linalg.norm(emb)  # L2 normalize
    return emb

def cosine_sim(a, b):
    return np.dot(a, b)

def find_match(emb):
    best_name, best_score = "Necunoscut", 0.0
    for name, saved_emb in saved_embeddings.items():
        score = cosine_sim(emb, saved_emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < 0.4:
        return "Necunoscut", best_score
    return best_name, best_score

cap = cv2.VideoCapture(0)
frame_num = 0
print("\n=== Controls ===")
print("r = register mode (then type name + Enter)")
print("q = quit")
print("================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

    preds = face_det.run(None, {det_input: blob})[0][0]  # (5, 8400)
    mask = preds[4] > CONF_THRESH
    filtered = preds[:, mask].T

    faces = []
    if len(filtered) > 0:
        if frame_num == 0:
            print(f"RAW first detection: cx={filtered[0,0]:.4f} cy={filtered[0,1]:.4f} w={filtered[0,2]:.4f} h={filtered[0,3]:.4f} conf={filtered[0,4]:.4f}")
        raw_cx = filtered[:, 0]
        if raw_cx.max() <= 1.5:
            cx = filtered[:, 0] * w
            cy = filtered[:, 1] * h
            bw = filtered[:, 2] * w
            bh = filtered[:, 3] * h
        else:
            cx = filtered[:, 0] / 640 * w
            cy = filtered[:, 1] / 640 * h
            bw = filtered[:, 2] / 640 * w
            bh = filtered[:, 3] / 640 * h
        confs = filtered[:, 4]
        x1 = (cx - bw / 2).astype(int)
        y1 = (cy - bh / 2).astype(int)
        x2 = (cx + bw / 2).astype(int)
        y2 = (cy + bh / 2).astype(int)

        nms_boxes = [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])] for i in range(len(filtered))]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), CONF_THRESH, IOU_THRESH).flatten()

        for idx in indices:
            emb = get_embedding(frame, x1[idx], y1[idx], x2[idx], y2[idx])
            if emb is not None:
                faces.append((x1[idx], y1[idx], x2[idx], y2[idx], confs[idx], emb))

    for (fx1, fy1, fx2, fy2, conf, emb) in faces:
        if mode == "register" and register_name:
            saved_embeddings[register_name] = emb
            print(f"Saved '{register_name}' (emb norm={np.linalg.norm(emb):.3f}, first5={emb[:5]})")
            mode = "recognize"
            register_name = ""

        name, score = find_match(emb)
        print(f"  Match: {name} score={score:.4f} box=({fx1},{fy1})-({fx2},{fy2}) frame={w}x{h}")
        color = (0, 255, 0) if name != "Necunoscut" else (0, 0, 255)
        # Clamp coords to frame
        dx1, dy1 = max(0, fx1), max(0, fy1)
        dx2, dy2 = min(w-1, fx2), min(h-1, fy2)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 3)
        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (dx1, max(dy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # UI info
    info = f"Saved: {list(saved_embeddings.keys())}" if saved_embeddings else "No persons saved"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if mode == "register":
        cv2.putText(frame, "REGISTER MODE - type name in terminal", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Face Recognition Pipeline", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        mode = "register"
        register_name = input("Name to register: ").strip()
        print(f"Will register '{register_name}' on next detected face...")

cap.release()
cv2.destroyAllWindows()
