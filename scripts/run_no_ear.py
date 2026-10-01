import os
import time
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

torch.set_num_threads(8)
cv2.setNumThreads(8)

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
print(f"[OK] Kamera: /dev/video{cam_id}")

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

print(f"Model Devrede: {model_path}")
model = YOLO(model_path, task="detect")

# Katı eşikler: Kulağı içeri almaz
CONF_ACQUIRE = 0.62   # Çok katı doğrulama (kulak skorları genelde 0.40-0.55 kalır)
CONF_HOLD    = 0.32
MAX_JUMP     = 120.0

is_locked = False
target_box = None
lost_count = 0
MAX_LOST = 3

prev_t = time.time()
fps = 0.0

def is_skin_tone(crop_bgr):
    """
    Kutunun içindeki piksellerin ten rengi oranını kontrol eder.
    Kulak veya yüz ise True döner ve hedef doğrudan elenir.
    """
    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return True
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Standart ten rengi HSV aralığı
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 175, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.count_nonzero(mask) / (crop_bgr.shape[0] * crop_bgr.shape[1])
    
    # Eğer kutunun %55'inden fazlası ten rengiyse kesinlikle kulaktır/eldir
    return skin_ratio > 0.55

print("\n--- ANTI-SKIN / ANTI-EAR DART MOTORU DEVREDE ---")
print("Ten rengi filtreleme devrede | Çıkış: 'q'\n")

while cap.isOpened():
    t_start = time.time()

    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE

    results = model.predict(
        source=frame,
        conf=active_conf,
        imgsz=320,
        max_det=5,
        agnostic_nms=True,
        save=False,
        show=False,
        verbose=False
    )

    valid_candidates = []

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        bx1 = max(0, min(w, bx1))
        by1 = max(0, min(h, by1))
        bx2 = max(0, min(w, bx2))
        by2 = max(0, min(h, by2))

        bw = bx2 - bx1
        bh = by2 - by1
        area = bw * bh

        # 1. Boyut Sınırları
        if bw < 10 or bh < 10 or bw > 140 or bh > 140 or area > 4500:
            continue

        # 2. Silindirik Geometri
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        if not is_locked and aspect < 1.70:
            continue

        # 3. KULAK VE TEN ENGELLEYİCİ (Kutu içeriğini hızlı analiz et)
        crop = frame[by1:by2, bx1:bx2]
        if not is_locked and is_skin_tone(crop):
            continue  # Kulağı veya eli doğrudan çöpe at!

        valid_candidates.append((np.array([bx1, by1, bx2, by2], dtype=np.float32), conf))

    best_match = None

    if is_locked and target_box is not None:
        curr_cx = (target_box[0] + target_box[2]) / 2.0
        curr_cy = (target_box[1] + target_box[3]) / 2.0
        best_dist = MAX_JUMP

        for b, conf in valid_candidates:
            b_cx = (b[0] + b[2]) / 2.0
            b_cy = (b[1] + b[3]) / 2.0
            dist = np.hypot(b_cx - curr_cx, b_cy - curr_cy)
            if dist < best_dist:
                best_dist = dist
                best_match = b
    else:
        max_c = 0.0
        for b, conf in valid_candidates:
            if conf > max_c:
                max_c = conf
                best_match = b

    # Kilit Yönetimi
    if best_match is not None:
        if not is_locked or target_box is None:
            target_box = best_match
            is_locked = True
        else:
            target_box = 0.90 * best_match + 0.10 * target_box
        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_box = None

    # Çizim
    if is_locked and target_box is not None:
        x1, y1, x2, y2 = map(int, target_box)
        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2

        color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status = "LOCKED" if lost_count == 0 else "HOLD"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
        cv2.putText(frame, status, (x1, max(22, y1 - 8)),
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

    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {int(dt*1000)}ms", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Anti-Ear Dart Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
