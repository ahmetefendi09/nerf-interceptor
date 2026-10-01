import cv2
import time
import os
from ultralytics import YOLO

# Model yolu (ONNX varsa onu, yoksa pt)
model_path = "runs/detect/dart_final/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_final/weights/best.pt"

print(f"Aktif Model: {model_path}")
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# EŞİK DEĞERİ: %55 - %60 bandına çektik
CONF_THRESHOLD = 0.55

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    frame_area = w * h

    # Çıkarım
    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

    dart_var = False
    target_x, target_y = 0, 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        bw = x2 - x1
        bh = y2 - y1
        box_area = bw * bh

        # FİLTRE 1: Boyut kontrolü (Ekranın %15'inden büyükse kulak/yüz/eldir, yoksay)
        if box_area > (frame_area * 0.15):
            continue

        # FİLTRE 2: Aşırı küçük gürültüleri engelle (Örn: 10x10 pikselden ufak parazitler)
        if bw < 12 or bh < 12:
            continue

        # Dart onaylandı
        dart_var = True
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2

        # Çizimler
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Dart %{int(conf*100)}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.line(frame, (cx, cy), (target_x, target_y), (255, 0, 0), 2)
        break

    # Nişangah
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if dart_var:
        err_x = target_x - cx
        err_y = target_y - cy
        cv2.putText(frame, f"Hata X: {err_x} Y: {err_y}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Nerf Interceptor Takip", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
