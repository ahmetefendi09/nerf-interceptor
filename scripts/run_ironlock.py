import os
import time
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"
cv2.setNumThreads(4)

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
print(f"[OK] Alcor Kamera Kilitlendi: /dev/video{cam_id}")

cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"HATA: /dev/video{cam_id} açılamadı!")
    exit(1)

model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"Model Yüklendi: {model_path}")
model = YOLO(model_path, task="detect")

# Eşikler
CONF_ACQUIRE = 0.40  # Yakalama güveni
CONF_HOLD    = 0.22  # Kilit koruma güveni
MAX_JUMP     = 100.0 # Maksimum izin verilen piksel sıçraması (Savrulma engeli)

is_locked = False
target_center = None
target_dims = (40, 40)
lost_count = 0
MAX_LOST = 5

prev_t = time.time()
fps = 0.0

def make_letterbox(img):
    h, w = img.shape[:2]
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    y_offset = (640 - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- KARARLI & DENGELİ TAKİP MOTORU DEVREDE ---")
print("İşlemci dengeli yükte | Çıkış: 'q'\n")

while cap.isOpened():
    t_start = time.time()

    # Donanım kuyruğundaki gecikmeyi temizle, en taze kareyi al
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    lb_frame, y_offset = make_letterbox(frame)
    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE

    # Sıralı ve kararlı çıkarım
    results = model.predict(
        source=lb_frame,
        conf=active_conf,
        imgsz=640,
        save=False,
        show=False,
        verbose=False
    )

    valid_candidates = []

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

        cand_cx = (x1 + x2) / 2.0
        cand_cy = (y1 + y2) / 2.0
        valid_candidates.append((x1, y1, x2, y2, cand_cx, cand_cy, conf, bw, bh))

    chosen = None

    if is_locked and target_center is not None:
        # Kilitliyken: Sadece mevcut hedefin 100 piksel yakınındaki adaya izin ver
        best_dist = MAX_JUMP
        for c in valid_candidates:
            dist = np.hypot(c[4] - target_center[0], c[5] - target_center[1])
            if dist < best_dist:
                best_dist = dist
                chosen = c
    else:
        # Arama modundayken: En yüksek güvenilirlikteki dartı seç
        max_c = 0.0
        for c in valid_candidates:
            if c[6] > max_c:
                max_c = c[6]
                chosen = c

    # Konum Güncelleme
    if chosen is not None:
        meas_x, meas_y = chosen[4], chosen[5]
        target_dims = (chosen[7], chosen[8])

        if not is_locked:
            target_center = np.array([meas_x, meas_y])
            is_locked = True
        else:
            # Yumuşak geçiş filtresi (Ani sıçrama ve jitter engelleme)
            target_center = 0.65 * target_center + 0.35 * np.array([meas_x, meas_y])

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_center = None

    # Çizim Katmanı
    if is_locked and target_center is not None:
        tx, ty = int(target_center[0]), int(target_center[1])
        bw, bh = target_dims
        draw_x1 = max(0, tx - bw // 2)
        draw_y1 = max(0, ty - bh // 2)
        draw_x2 = min(w, tx + bw // 2)
        draw_y2 = min(h, ty + bh // 2)

        color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = "LOCKED" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (draw_x1, max(22, draw_y1 - 8)),
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

    cv2.imshow("Stable Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
