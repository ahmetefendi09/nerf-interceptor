import cv2
import time
import os
from ultralytics import YOLO

# Alcor Kamera
CAM_ID = 2

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model_path = "runs/detect/dart_final/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_final/weights/best.pt"

print(f"Yüklenen Model: {model_path}")
model = YOLO(model_path)

prev_time = time.time()
print("Canlı test devrede. Çıkış: 'q'")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Filtreleri gevşettik: %20 güvenin üzerindeki her şeyi ekrana basıyoruz
    results = model.predict(source=frame, conf=0.20, imgsz=640, verbose=False)

    best_target = None
    max_conf = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())

        if conf > max_conf:
            max_conf = conf
            best_target = (x1, y1, x2, y2, conf, cls_id)

    # En yüksek güvenli hedefi çiz
    if best_target is not None:
        x1, y1, x2, y2, conf, cls_id = best_target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Dart %{int(conf*100)}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"Hata X: {err_x} Y: {err_y}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Ekran merkezi
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # Gerçek döngü FPS hesabı
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1.0 / dt if dt > 0 else 0

    cv2.putText(frame, f"Kamera FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Alcor Tek Kamera - Nerf Interceptor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
