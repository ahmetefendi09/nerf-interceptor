from ultralytics import YOLO

# En hafif nano mimari
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    name="dart_baseline",
    device="cpu"
)
