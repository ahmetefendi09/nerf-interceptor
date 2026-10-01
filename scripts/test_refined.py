import cv2
import time
import numpy as np
from ultralytics import YOLO

CAM_ID = 2
model = YOLO("runs/detect/dart_final/weights/best.pt")

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Kontrast dengeleyici (soluk web kamerasını keskinleştirir)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

prev_t = time.time()
print("Canlı Test Başladı. Çıkış: 'q'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # 1. Kontrast ve Parlaklık İyileştirme (Dart kenarlarını netleştirir)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Model Çıkarımı (Gevşetilmiş %30 eşik)
    results = model.predict(source=enhanced, conf=0.30, imgsz=640, verbose=False)

    boxes = results[0].boxes
    best_target = None
    max_c = 0.0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1

        # Aşırı büyük şekilleri ele (tüm yüz / gövde)
        if (bw * bh) > (640 * 480 * 0.15):
            continue

        if conf > max_c:
            max_c = conf
            best_target = (x1, y1, x2, y2, conf)

    # Çizim
    if best_target:
        x1, y1, x2, y2, conf = best_target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Dart %{int(conf*100)}", (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(frame, (320, 240), (tx, ty), (255, 0, 0), 2)

    cv2.drawMarker(frame, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Dart Canli Kilit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
