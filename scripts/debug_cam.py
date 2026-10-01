import cv2
import time
import os
from ultralytics import YOLO

model_path = "runs/detect/dart_final/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_final/weights/best.pt"

print(f"Test Modeli: {model_path}")
model = YOLO(model_path)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- TEŞHİS BAŞLADI ---")
print("Dartı kameraya tut. Ekranda veya terminalde kaç güven skoru (Conf) bastığına bak.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Eşiği bilerek %15'e çektik ki model en ufak şüphesinde bile kutu atsın
    results = model.predict(source=frame, conf=0.15, imgsz=640, verbose=False)

    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            bw = x2 - x1
            bh = y2 - y1

            # Ekrana kırmızı/yeşil kutu bas
            renk = (0, 255, 0) if conf > 0.40 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
            cv2.putText(frame, f"ID:{cls_id} Conf:%{int(conf*100)} W:{bw} H:{bh}", 
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)
            
            print(f"Tespit -> Sınıf: {cls_id} | Güven: %{int(conf*100)} | Boyut: {bw}x{bh}")

    cv2.imshow("Dart Debug Ekrani", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
