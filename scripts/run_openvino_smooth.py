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

model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    print(f"HATA: {model_path} bulunamadı! Önce export komutunu çalıştır.")
    exit(1)

print(f"\nModel: {model_path} (OpenVINO CPU Accelerated)")
model = YOLO(model_path, task="detect")

# Takip Değişkenleri
is_locked = False
target_pos = None      # [x, y]
velocity = np.array([0.0, 0.0])  # [vx, vy]
box_dims = (40, 40)
lost_count = 0
MAX_LOST = 6           # Kaç kare kayıp toleransı tanınacak

CONF_ACQUIRE = 0.40    # İlk yakalama güven eşiği
CONF_HOLD    = 0.25    # Yakalandıktan sonra koruma eşiği

prev_t = time.time()
fps = 0.0

def make_letterbox(img, target_size=640):
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- OPENVINO SMOOTH INTERCEPTOR BAŞLATILDI ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    cap.grab()
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 640x640 Letterbox
    lb_frame, y_offset = make_letterbox(frame, target_size=640)

    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE
    results = model.predict(source=lb_frame, conf=active_conf, imgsz=640, verbose=False)

    best_cand = None
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

        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        # Kilitli değilken kulak/dil gibi karemsi şekilleri ele
        if not is_locked and area > 400 and aspect < 1.30:
            continue

        if conf > max_c:
            max_c = conf
            best_cand = (x1, y1, x2, y2, conf)

    # Akıllı Takip ve Hareket Filtresi
    if best_cand is not None:
        x1, y1, x2, y2, conf = best_cand
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        box_dims = (x2 - x1, y2 - y1)

        if not is_locked:
            target_pos = np.array([raw_x, raw_y])
            velocity = np.array([0.0, 0.0])
            is_locked = True
        else:
            # Hız vektörünü hesapla ve ani sıçramaları filtrele (Exponential Moving Average)
            new_vel = np.array([raw_x - target_pos[0], raw_y - target_pos[1]])
            # Hız tavanı koy: 1 karede 80 pikselden fazla sıçrayamaz (parazit eleme)
            vel_mag = np.linalg.norm(new_vel)
            if vel_mag > 80.0:
                new_vel = (new_vel / vel_mag) * 80.0

            velocity = 0.6 * velocity + 0.4 * new_vel
            # Yumuşatılmış hedef konumu
            target_pos = 0.7 * np.array([raw_x, raw_y]) + 0.3 * (target_pos + velocity)

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            # Model kaçırdığında hız vektörüyle sönümleyerek devam et
            velocity *= 0.85
            target_pos += velocity
            if lost_count > MAX_LOST:
                is_locked = False
                target_pos = None

    # Çizimler
    if is_locked and target_pos is not None:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        bw, bh = box_dims
        draw_x1 = max(0, tx - bw // 2)
        draw_y1 = max(0, ty - bh // 2)
        draw_x2 = min(w, tx + bw // 2)
        draw_y2 = min(h, ty + bh // 2)

        color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status = "LOCKED" if lost_count == 0 else "PREDICTING"

        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status, (draw_x1, max(22, draw_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("OpenVINO Smooth Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
