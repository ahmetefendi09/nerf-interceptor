from ultralytics import YOLO

# Sıfırdan veya baseline ağırlıklarından başlatabiliriz, transfer learning için yolov8n temizdir
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    workers=4,
    name="dart_final",
    device="cpu",
    patience=15  # Model gelişmeyi durdurursa erken bitirip en iyi ağırlığı alır
)
