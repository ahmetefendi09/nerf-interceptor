import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

CAM_ID = 2

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# En optimize modeli yükle
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel: {model_path} (640x640 Kesintisiz Takip)")
model = YOLO(model_path, task="detect")

# Histerezis eşikleri
CONF_ACQUIRE = 0.40  # İlk yakalama için yüksek güvenlik
CONF_HOLD    = 0.25  # Yakalandıktan sonra kopmayı engelleyen eşik

locked_target = None
lost_count = 0
MAX_LOST = 5  # Hedef kaçsa bile 5 kare boyunca yeri koru

prev_t = time.time()
fps = 0.0

def make_letterbox(img, target_size=640):
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- KESİNTİSİZ 640x640 KİLİT MOTORU DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 640x640 Letterbox
    letterbox_frame, y_offset = make_letterbox(frame, target_size=640)

    # Duruma göre dinamik eşik: Kilitliysek %25, hedef arıyorsak %40
    active_conf = CONF_HOLD if locked_target is not None else CONF_ACQUIRE
    results = model.predict(source=letterbox_frame, conf=active_conf, imgsz=640, verbose=False)

    best_candidate = None
    max_c = 0.0

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        x1 = bx1
        x2 = bx2
        y1 = max(0, min(h, by1 - y_offset))
        y2 = max(0, min(h, by2 - y_offset))
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area > (w * h * 0.12) or bw < 8 or bh < 8:
            continue

        if conf > max_c:
            max_c = conf
            best_candidate = (x1, y1, x2, y2, conf)

    # Durum Yönetimi (State Machine)
    if best_candidate is not None:
        locked_target = best_candidate
        lost_count = 0
    else:
        if locked_target is not None:
            lost_count += 1
            if lost_count > MAX_LOST:
                locked_target = None

    # Çizim ve Görselleştirme
    if locked_target is not None:
        x1, y1, x2, y2, conf = locked_target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        # Tam kilitteyken yeşil, anlık tolerans modundayken sarı
        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = f"LOCKED %{int(conf*100)}" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TARGET SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Ironclad 640 Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
