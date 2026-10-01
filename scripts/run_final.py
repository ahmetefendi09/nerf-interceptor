import cv2
import time
import os
from ultralytics import YOLO

CAM_ID = 2

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Model yükleme
model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel Devrede: {model_path}")
model = YOLO(model_path)

locked_box = None
lost_frames = 0
MAX_LOST_FRAMES = 5

prev_t = time.time()
print("Sadeleştirilmiş Kilit Sistemi Başlatıldı. Çıkış: 'q'\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Ham görüntüyle tahmin (filtresiz, modelin doğrudan kendi gördüğü)
    results = model.predict(source=frame, conf=0.18, imgsz=640, verbose=False)

    best_candidate = None
    max_conf = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # 1. Boyut Sınırı: Ekranın %8'inden büyük olamaz (yüz, kafa, gövde elenir)
        if area > (w * h * 0.08) or bw < 8 or bh < 8:
            continue

        # 2. Kulak / Dil Filtresi:
        # Kulak ve dil ucu orta boyutta (alan > 400) ve karemsidir (en-boy oranı 1.0 - 1.25 arası).
        # Dart ise uzundur (aspect > 1.35) ya da çok uzaktaysa miniktir (alan < 350).
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        if area > 400 and aspect < 1.30:
            continue

        if conf > max_conf:
            max_conf = conf
            best_candidate = (x1, y1, x2, y2, conf)

    # Kilit Yönetimi (Anlık kayıpları engelle)
    if best_candidate is not None:
        locked_box = best_candidate
        lost_frames = 0
    else:
        if locked_box is not None:
            lost_frames += 1
            if lost_frames > MAX_LOST_FRAMES:
                locked_box = None

    # Çizimler
    if locked_box is not None:
        x1, y1, x2, y2, conf = locked_box
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        color = (0, 255, 0) if lost_frames == 0 else (0, 255, 255)
        status = f"DART %{int(conf*100)}" if lost_frames == 0 else "HOLDING..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"Sapma X: {err_x:+d} | Y: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TARGET SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Nerf Interceptor - Calisan Kilit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
