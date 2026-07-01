import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO money detection on webcam")
    parser.add_argument("--weights", required=True, help="Path to best.pt (trained weights)")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per frame")
    args = parser.parse_args()

    # Load YOLO model
    model = YOLO(args.weights)

    # Open webcam
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.source}")

    # Try a decent default resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_t = time.time()
    fps = 0.0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        # Inference
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
        )

        r = results[0]
        names = r.names  # dict: class_id -> class_name

        # Draw detections
        if r.boxes is not None and len(r.boxes) > 0:
            # xyxy in pixels
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = f"{names.get(cls_id, str(cls_id))} {c:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # FPS overlay
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Money Detection (YOLO)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()