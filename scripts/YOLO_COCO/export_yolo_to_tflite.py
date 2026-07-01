from ultralytics import YOLO

model = YOLO("Ultralytics YOLO 11s.pt")
model.export(format="tflite", imgsz=640)
