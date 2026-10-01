import os
import time
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

def get_alcor_camera_id():
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
        lines = out.split("\n")
        target = False
        for line in lines:
            if "alcor" in line.lower():
                target = True
                continue
            if target and "/dev/video" in line:
                return int(line.strip().replace("/dev/video", ""))
            if target and line.strip() == "":
                target = False
    except Exception:
        pass
    return 2

cam_id = get_alcor_camera_id()
print(f"\n[OK] Alcor UVC Kamera: /dev/video{cam_id}")

cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"HATA: /dev/video{cam_id} açılamadı!")
    exit(1)

model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"Model Devrede: {model_path}")
model = YOLO(model_path, task="detect")

# Histerezis eşikleri (Hızlı kapma + yapışkan kilit)
CONF_ACQUIRE = 0.35
CONF_HOLD    = 0.20

is_locked = False
target_pos = None
velocity = np.array([0.0, 0.0])
box_dims = (40, 40)
lost_count = 0
MAX_LOST = 5

prev_t = time.time()
fps = 15.0

def make_letterbox(img, target_size=640):
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- SIFIR GECİKME GERÇEK 15 FPS MOTOR DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    # Donanım kuyruğundaki bayat kareleri at (Lag kesici)
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 640x640 Letterbox
    lb_frame, y_offset = make_letterbox(frame, target_size=640)
    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE

    # stream=True ile tensör tahsis gecikmesini sıfırla
    results = model.predict(
        source=lb_frame,
        conf=active_conf,
        imgsz=640,
        stream=False,
        save=False,
        show=False,
        verbose=False
    )

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
        if not is_locked and area > 350 and aspect < 1.30:
            continue

        if conf > max_c:
            max_c = conf
            best_cand = (x1, y1, x2, y2, conf)

    # Doğrudan Canlı Takip & Anlık Reaksiyon
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
            new_vel = np.array([raw_x - target_pos[0], raw_y - target_pos[1]])
            vel_mag = np.linalg.norm(new_vel)
            if vel_mag > 80.0:
                new_vel = (new_vel / vel_mag) * 80.0

            velocity = 0.4 * velocity + 0.6 * new_vel
            target_pos = 0.8 * np.array([raw_x, raw_y]) + 0.2 * (target_pos + velocity)

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            velocity *= 0.75
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

        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = f"LOCKED %{int(max_c*100)}" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), box_color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (draw_x1, max(22, draw_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # Donanım seviyesinde ölçülen gerçek döngü hızı
    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Direct Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
