import os
from ultralytics import YOLO

yaml_content = """path: ./dataset_finetune
train: images/train
val: images/val

names:
  0: dart
"""

with open("finetune.yaml", "w") as f:
    f.write(yaml_content)

print("Mevcut model yükleniyor...")
base_model = "runs/detect/dart_final/weights/best.pt"
if not os.path.exists(base_model):
    base_model = "yolov8n.pt"

model = YOLO(base_model)

print("\n--- INCE AYAR (FINE-TUNE) BASLIYOR ---")
model.train(
    data="finetune.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    lr0=0.001,      # Mevcut agirliklari bozmamak icin dusuk ogrenme hizi
    lrf=0.01,
    name="dart_finetuned",
    device="cpu"
)

print("\nEğitim tamamlandı! Yeni ağırlık: runs/detect/dart_finetuned/weights/best.pt")
