import cv2

MODEL = "face_detection_yunet_2023mar.onnx"
LM_COLORS = [(0,255,0),(255,0,0),(0,255,255),(0,128,255),(255,0,128)]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

detector = cv2.FaceDetectorYN_create(MODEL, "", (w, h), 0.6, 0.3, 10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, faces = detector.detect(frame)

    if faces is not None:
        for f in faces:
            x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(frame, f"{f[14]:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            for j in range(5):
                lx, ly = int(f[4+2*j]), int(f[5+2*j])
                cv2.circle(frame, (lx, ly), 4, LM_COLORS[j], -1)

    cv2.imshow("YuNet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
