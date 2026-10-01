import cv2
import time
import os
from ultralytics import YOLO

CAM_ID = 2

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# En hızlı model formatı
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel: {model_path}")
model = YOLO(model_path, task="detect")

# Güvenlik eşikleri
CONF_INITIAL = 0.38  # İlk kilit için yüksek güvenlik
CONF_TRACK   = 0.28  # Kilit koruma eşiği

locked_box = None
lost_count = 0
MAX_LOST = 5  # 10 FPS'te ~0.5 saniye kilit koruması

prev_t = time.time()
fps = 10.0

print("--- 10 FPS & HIZLANDIRILMIŞ YAPAY ZEKA DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    # Bayat kare birikmesini engellemek için anlık kareyi çek
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # imgsz=256: CPU çıkarım hızını maksimuma çıkarır
    active_conf = CONF_TRACK if locked_box is not None else CONF_INITIAL
    results = model.predict(source=frame, conf=active_conf, imgsz=256, verbose=False)

    best_cand = None
    max_c = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # Alan sınırları (dev yüz/gövde elensin)
        if area > (w * h * 0.12) or bw < 8 or bh < 8:
            continue

        # Kulak filtresi (orta boy karemsi yapılar)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        if area > 350 and aspect < 1.30:
            continue

        if conf > max_c:
            max_c = conf
            best_cand = (x1, y1, x2, y2, conf)

    # Durum Yönetimi
    if best_cand is not None:
        locked_box = best_cand
        lost_count = 0
    else:
        if locked_box is not None:
            lost_count += 1
            if lost_count > MAX_LOST:
                locked_box = None

    # Görselleştirme
    if locked_box is not None:
        x1, y1, x2, y2, conf = locked_box
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = f"LOCKED %{int(conf*100)}" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TARGET SEARCHING...", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    # FPS Hesabı
    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("10 FPS Fast Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
