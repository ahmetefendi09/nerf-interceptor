import os
import time
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

# İşlemcinin tüm 8 izleğini serbest bırak
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

print(f"Model Devrede: {model_path} (Sıfır Kısıtlama)")
model = YOLO(model_path, task="detect")

# HIZ ODAKLI AGRESİF EŞİKLER
CONF_ACQUIRE = 0.28  # Düşük eşik: Kadraja girdiği an anında yakala
CONF_HOLD    = 0.18  # Kopmaz kilit

is_locked = False
target_box = None
lost_count = 0
MAX_LOST = 3

prev_t = time.time()
fps = 0.0

def make_letterbox(img):
    h, w = img.shape[:2]
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    y_offset = (640 - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- SIFIR GECİKME ANLIK KİLİT MOTORU DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    # Donanım gecikmesini sıfırlamak için çift boşaltma
    cap.grab()
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    lb_frame, y_offset = make_letterbox(frame)
    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE

    results = model.predict(
        source=lb_frame,
        conf=active_conf,
        imgsz=640,
        save=False,
        show=False,
        verbose=False
    )

    best_match = None
    max_c = 0.0

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        x1 = max(0, min(w, bx1))
        y1 = max(0, min(h, by1 - y_offset))
        x2 = max(0, min(w, bx2))
        y2 = max(0, min(h, by2 - y_offset))

        bw = x2 - x1
        bh = y2 - y1

        # Sadece 8 pikselden küçük ekran gürültülerini ele
        if bw < 8 or bh < 8:
            continue

        if conf > max_c:
            max_c = conf
            best_match = np.array([x1, y1, x2, y2], dtype=np.float32)

    # Kilit Yönetimi: Sıfır filtre, anlık takip
    if best_match is not None:
        # Doğrudan modelin bulduğu son konuma zıpla (gecikme sıfır)
        target_box = best_match
        is_locked = True
        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_box = None

    # Çizim & Koordinat
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

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Zero-Lag Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
