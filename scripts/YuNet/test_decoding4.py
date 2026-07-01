import cv2
import numpy as np
import onnxruntime as ort

MODEL = "face_detection_yunet_2023mar.onnx"
INPUT_SIZE = 640
STRIDES = [8, 16, 32]
CONF_THRESH = 0.3
NMS_THRESH = 0.4
VARIANCE = [0.1, 0.2]

session = ort.InferenceSession(MODEL)
frame = cv2.imread("poza.jpg")
h0, w0 = frame.shape[:2]
print(f"Image: {w0}x{h0}")

def generate_priors():
    """OpenCV-style priors: 2 min_sizes per stride, normalized coords"""
    min_sizes = [[10, 16], [32, 64], [128, 256]]
    priors = []
    for si, stride in enumerate(STRIDES):
        grid_h = INPUT_SIZE // stride
        grid_w = INPUT_SIZE // stride
        for row in range(grid_h):
            for col in range(grid_w):
                for min_size in min_sizes[si]:
                    cx = (col + 0.5) / grid_w
                    cy = (row + 0.5) / grid_h
                    sx = min_size / INPUT_SIZE
                    sy = min_size / INPUT_SIZE
                    priors.append((cx, cy, sx, sy))
    return priors

def generate_anchors_simple():
    """Simple: 1 anchor per position, stride as size"""
    anchors = []
    for stride in STRIDES:
        grid_h = INPUT_SIZE // stride
        grid_w = INPUT_SIZE // stride
        for row in range(grid_h):
            for col in range(grid_w):
                anchors.append(((col + 0.5) * stride, (row + 0.5) * stride, float(stride)))
    return anchors

priors_2anchor = generate_priors()
anchors_1anchor = generate_anchors_simple()
print(f"Priors (2 per pos): {len(priors_2anchor)}")
print(f"Anchors (1 per pos): {len(anchors_1anchor)}")

def preprocess(frame, mode):
    h0, w0 = frame.shape[:2]
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    if mode == "bgr_raw":
        blob = img.astype(np.float32)
    elif mode == "rgb_raw":
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    elif mode == "bgr_norm":
        blob = img.astype(np.float32) / 255.0
    elif mode == "rgb_norm":
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob

def decode_1anchor(out_map, scale_x, scale_y, score_mode):
    anchors = anchors_1anchor
    dets = []
    idx = 0
    for si, stride in enumerate(STRIDES):
        cls_d = out_map[f"cls_{stride}"][0].flatten()
        obj_d = out_map[f"obj_{stride}"][0].flatten()
        bbox_d = out_map[f"bbox_{stride}"][0]
        kps_d = out_map[f"kps_{stride}"][0]
        for i in range(cls_d.shape[0]):
            c, o = cls_d[i], obj_d[i]
            if score_mode == "cls*obj":
                score = c * o
            elif score_mode == "sqrt":
                score = np.sqrt(max(0, c * o))
            elif score_mode == "cls_only":
                score = c

            if score > CONF_THRESH:
                cx_a, cy_a, sz = anchors[idx]
                dx, dy, dw, dh = bbox_d[i]
                cx = cx_a + dx * sz
                cy = cy_a + dy * sz
                w = np.exp(dw) * sz
                h = np.exp(dh) * sz
                x1 = (cx - w/2) / scale_x
                y1 = (cy - h/2) / scale_y
                x2 = (cx + w/2) / scale_x
                y2 = (cy + h/2) / scale_y

                lms = []
                for j in range(5):
                    lx = (cx_a + kps_d[i][2*j] * sz) / scale_x
                    ly = (cy_a + kps_d[i][2*j+1] * sz) / scale_y
                    lms.append((lx, ly))

                dets.append((score, x1, y1, x2, y2, lms))
            idx += 1
    return dets

def decode_2anchor_variance(out_map, score_mode):
    """OpenCV-style: 2 anchors per position with variance-based decode"""
    priors = priors_2anchor
    # Flatten all stride outputs and concatenate in prior order
    all_cls = []
    all_obj = []
    all_bbox = []
    all_kps = []
    for stride in STRIDES:
        cls_d = out_map[f"cls_{stride}"][0]  # [N, 1]
        obj_d = out_map[f"obj_{stride}"][0]
        bbox_d = out_map[f"bbox_{stride}"][0]
        kps_d = out_map[f"kps_{stride}"][0]
        # Each entry in model output corresponds to 1 position
        # But priors have 2 per position. Maybe interleave?
        n = cls_d.shape[0]
        # Duplicate each entry for the 2 anchor sizes
        for i in range(n):
            all_cls.append(cls_d[i][0])
            all_cls.append(cls_d[i][0])  # same cls for both anchors
            all_obj.append(obj_d[i][0])
            all_obj.append(obj_d[i][0])
            all_bbox.append(bbox_d[i])
            all_bbox.append(bbox_d[i])
            all_kps.append(kps_d[i])
            all_kps.append(kps_d[i])

    dets = []
    for i in range(len(priors)):
        c, o = all_cls[i], all_obj[i]
        if score_mode == "cls*obj":
            score = c * o
        elif score_mode == "sqrt":
            score = np.sqrt(max(0, c * o))
        elif score_mode == "cls_only":
            score = c

        if score > CONF_THRESH:
            pcx, pcy, psw, psh = priors[i]
            dx, dy, dw, dh = all_bbox[i]

            cx = (pcx + dx * VARIANCE[0]) * INPUT_SIZE
            cy = (pcy + dy * VARIANCE[0]) * INPUT_SIZE
            w = psw * np.exp(dw * VARIANCE[1]) * INPUT_SIZE
            h = psh * np.exp(dh * VARIANCE[1]) * INPUT_SIZE

            scale_x = INPUT_SIZE / w0
            scale_y = INPUT_SIZE / h0
            x1 = (cx - w/2) / scale_x
            y1 = (cy - h/2) / scale_y
            x2 = (cx + w/2) / scale_x
            y2 = (cy + h/2) / scale_y

            dets.append((score, x1, y1, x2, y2, []))
        # no idx increment needed, we iterate all

    return dets

def nms(dets):
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
            a1 = max(1, (d[3]-d[1]) * (d[4]-d[2]))
            a2 = max(1, (k[3]-k[1]) * (k[4]-k[2]))
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > NMS_THRESH:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep

# OpenCV reference
det_cv = cv2.FaceDetectorYN_create(MODEL, "", (w0, h0), 0.5, 0.3, 10)
_, faces = det_cv.detect(frame)
print(f"\nOpenCV reference: {len(faces) if faces is not None else 0} faces")
if faces is not None:
    for f in faces:
        print(f"  conf={f[14]:.3f} box=[{f[0]:.0f},{f[1]:.0f},{f[0]+f[2]:.0f},{f[1]+f[3]:.0f}]")

# Test all combinations
preprocess_modes = ["bgr_raw", "rgb_raw", "bgr_norm", "rgb_norm"]
score_modes = ["cls*obj", "sqrt", "cls_only"]
decode_modes = ["1anchor", "2anchor_variance"]

scale_x = INPUT_SIZE / w0
scale_y = INPUT_SIZE / h0

print("\n=== Testing all combinations ===")
best_combo = None
best_score = -1

for pp in preprocess_modes:
    blob = preprocess(frame, pp)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    out_names = [o.name for o in session.get_outputs()]
    out_map = {name: outputs[i] for i, name in enumerate(out_names)}

    # Quick stats
    obj8_max = out_map["obj_8"][0].max()
    cls8_max = out_map["cls_8"][0].max()

    for sm in score_modes:
        for dm in decode_modes:
            if dm == "1anchor":
                dets = decode_1anchor(out_map, scale_x, scale_y, sm)
            else:
                dets = decode_2anchor_variance(out_map, sm)

            dets = nms(dets)
            n = len(dets)

            # Score: how close to OpenCV? (2 faces, close boxes)
            if n >= 1:
                label = f"{pp:10s} | {sm:10s} | {dm:18s} | {n} faces | obj8_max={obj8_max:.4f}"
                if n <= 5:
                    boxes = " | ".join([f"[{d[1]:.0f},{d[2]:.0f},{d[3]:.0f},{d[4]:.0f}] s={d[0]:.3f}" for d in dets[:3]])
                    label += f" | {boxes}"
                print(label)

                if n == 2:
                    if best_combo is None:
                        best_combo = (pp, sm, dm, dets)
                        best_score = n

# Draw best result
print(f"\n=== Best combo: {best_combo[0] if best_combo else 'NONE'} ===")

if best_combo:
    pp, sm, dm, dets = best_combo
    vis = frame.copy()
    for d in dets:
        s, x1, y1, x2, y2 = d[0], int(d[1]), int(d[2]), int(d[3]), int(d[4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, f"{s:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if len(d) > 5:
            for lx, ly in d[5]:
                cv2.circle(vis, (int(lx), int(ly)), 5, (0,255,255), -1)

    if faces is not None:
        for f in faces:
            x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            cv2.rectangle(vis, (x, y), (x+bw, y+bh), (255, 0, 0), 3)
            cv2.putText(vis, f"cv:{f[14]:.2f}", (x, y+bh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imwrite("result3.jpg", vis)
    print("Saved result3.jpg")
