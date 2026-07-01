import cv2
import numpy as np
import onnxruntime as ort

MODEL = "face_detection_yunet_2023mar.onnx"
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
CONF_THRESH = 0.3
NMS_THRESH = 0.4

session = ort.InferenceSession(MODEL)

frame = cv2.imread("poza.jpg")
h0, w0 = frame.shape[:2]
print(f"Image: {w0}x{h0}")

def generate_anchors():
    anchors = []
    for stride in STRIDES:
        grid_h = INPUT_SIZE // stride
        grid_w = INPUT_SIZE // stride
        for row in range(grid_h):
            for col in range(grid_w):
                anchors.append(((col + 0.5) * stride, (row + 0.5) * stride, float(stride)))
    return anchors

anchors = generate_anchors()

def run_decode(frame, use_letterbox):
    h0, w0 = frame.shape[:2]

    if use_letterbox:
        scale = min(INPUT_SIZE / w0, INPUT_SIZE / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        pad_x = (INPUT_SIZE - new_w) // 2
        pad_y = (INPUT_SIZE - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    else:
        # Direct resize (what OpenCV likely does)
        padded = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        scale_x = INPUT_SIZE / w0
        scale_y = INPUT_SIZE / h0
        pad_x = 0
        pad_y = 0

    blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    out_names = [o.name for o in session.get_outputs()]
    out_map = {name: outputs[i] for i, name in enumerate(out_names)}

    detections = []
    idx = 0
    for si, stride in enumerate(STRIDES):
        cls_data = out_map[f"cls_{stride}"][0].flatten()
        obj_data = out_map[f"obj_{stride}"][0].flatten()
        bbox_data = out_map[f"bbox_{stride}"][0]
        kps_data = out_map[f"kps_{stride}"][0]
        num = cls_data.shape[0]

        for i in range(num):
            score = float(cls_data[i] * obj_data[i])
            if score > CONF_THRESH:
                cx_a, cy_a, sz = anchors[idx]
                dx, dy, dw, dh = bbox_data[i]
                cx = cx_a + dx * sz
                cy = cy_a + dy * sz
                w = np.exp(dw) * sz
                h = np.exp(dh) * sz

                if use_letterbox:
                    ox1 = (cx - w/2 - pad_x) / scale
                    oy1 = (cy - h/2 - pad_y) / scale
                    ox2 = (cx + w/2 - pad_x) / scale
                    oy2 = (cy + h/2 - pad_y) / scale
                else:
                    ox1 = (cx - w/2) / scale_x
                    oy1 = (cy - h/2) / scale_y
                    ox2 = (cx + w/2) / scale_x
                    oy2 = (cy + h/2) / scale_y

                lms = []
                for j in range(5):
                    lx_raw = cx_a + kps_data[i][2*j] * sz
                    ly_raw = cy_a + kps_data[i][2*j+1] * sz
                    if use_letterbox:
                        lms.append(((lx_raw - pad_x) / scale, (ly_raw - pad_y) / scale))
                    else:
                        lms.append((lx_raw / scale_x, ly_raw / scale_y))

                detections.append((score, ox1, oy1, ox2, oy2, lms))
            idx += 1

    return detections

def nms(dets, thresh):
    if not dets:
        return []
    dets.sort(key=lambda x: -x[0])
    keep = []
    for d in dets:
        overlap = False
        for k in keep:
            x1 = max(d[1], k[1])
            y1 = max(d[2], k[2])
            x2 = min(d[3], k[3])
            y2 = min(d[4], k[4])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (d[3]-d[1]) * (d[4]-d[2])
            a2 = (k[3]-k[1]) * (k[4]-k[2])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > thresh:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep

# Test both approaches
for mode, use_lb in [("letterbox", True), ("direct_resize", False)]:
    dets = run_decode(frame, use_lb)
    dets = nms(dets, NMS_THRESH)
    print(f"\n{mode}: {len(dets)} faces after NMS")
    for d in dets:
        print(f"  score={d[0]:.3f} box=[{d[1]:.0f},{d[2]:.0f},{d[3]:.0f},{d[4]:.0f}]")

# OpenCV reference
det = cv2.FaceDetectorYN_create(MODEL, "", (w0, h0), 0.5, 0.3, 10)
_, faces = det.detect(frame)
print(f"\nOpenCV reference: {len(faces) if faces is not None else 0} faces")
if faces is not None:
    for f in faces:
        print(f"  conf={f[14]:.3f} box=[{f[0]:.0f},{f[1]:.0f},{f[0]+f[2]:.0f},{f[1]+f[3]:.0f}]")

# Draw the best (direct_resize most likely)
vis = frame.copy()
dets_draw = run_decode(frame, False)
dets_draw = nms(dets_draw, NMS_THRESH)
for d in dets_draw:
    s, x1, y1, x2, y2, lms = d
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.putText(vis, f"mine:{s:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    for lx, ly in lms:
        cv2.circle(vis, (int(lx), int(ly)), 5, (0, 255, 255), -1)

if faces is not None:
    for f in faces:
        x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (255, 0, 0), 3)
        cv2.putText(vis, f"cv:{f[14]:.2f}", (x, y+bh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

cv2.imwrite("result2.jpg", vis)
print("\nSaved result2.jpg")
