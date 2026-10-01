import os
import time
import subprocess
import threading
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

def get_alcor_camera_id():
    """v4l2-ctl çıktısından Alcor kameranın gerçek /dev/video index'ini çeker."""
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
        lines = out.split("\n")
        target_found = False
        for line in lines:
            if "alcor" in line.lower():
                target_found = True
                continue
            if target_found and "/dev/video" in line:
                dev_path = line.strip()
                cam_idx = int(dev_path.replace("/dev/video", ""))
                return cam_idx
            if target_found and line.strip() == "":
                target_found = False
    except Exception:
        pass
    return 2  # Bulunamazsa varsayılan video2

cam_id = get_alcor_camera_id()
print(f"\n[OK] Hedef Alcor Kamera Seçildi: /dev/video{cam_id}")

class AlcorCamWorker:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# En hızlı model yolunu belirle
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"Model Yüklendi: {model_path}")
model = YOLO(model_path, task="detect")

cam = AlcorCamWorker(cam_id)

latest_target = None
target_lock = threading.Lock()
ai_running = True

def make_letterbox(img, target_size=640):
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

def ai_worker_loop():
    global latest_target, ai_running
    while ai_running:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        lb_frame, y_offset = make_letterbox(frame, target_size=640)

        # save=False ve show=False ile fazladan görsel/pencere açılması engellenir
        results = model.predict(
            source=lb_frame,
            conf=0.35,
            imgsz=640,
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
            if area > 350 and aspect < 1.30:
                continue

            if conf > max_c:
                max_c = conf
                best_cand = (x1, y1, x2, y2, conf)

        with target_lock:
            latest_target = best_cand

# Yapay zeka iş parçacığını ateşle
threading.Thread(target=ai_worker_loop, daemon=True).start()

# Takip ve Hız Filtresi
target_pos = None
velocity = np.array([0.0, 0.0])
box_dims = (40, 40)
lost_count = 0
MAX_LOST = 8

prev_t = time.time()
fps = 0.0

print("\n--- ALCOR SIFIR-GECİKME NİŞANGAH BAŞLATILDI ---")
print("Çıkış için 'q' tuşuna bas.\n")

while True:
    t_start = time.time()

    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    with target_lock:
        cand = latest_target

    if cand is not None:
        x1, y1, x2, y2, conf = cand
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        box_dims = (x2 - x1, y2 - y1)

        if target_pos is None:
            target_pos = np.array([raw_x, raw_y])
            velocity = np.array([0.0, 0.0])
        else:
            new_vel = np.array([raw_x - target_pos[0], raw_y - target_pos[1]])
            vel_mag = np.linalg.norm(new_vel)
            if vel_mag > 60.0:
                new_vel = (new_vel / vel_mag) * 60.0

            velocity = 0.5 * velocity + 0.5 * new_vel
            target_pos = 0.75 * np.array([raw_x, raw_y]) + 0.25 * (target_pos + velocity)

        lost_count = 0
    else:
        if target_pos is not None:
            lost_count += 1
            velocity *= 0.8
            target_pos += velocity
            if lost_count > MAX_LOST:
                target_pos = None

    if target_pos is not None:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        bw, bh = box_dims
        draw_x1 = max(0, tx - bw // 2)
        draw_y1 = max(0, ty - bh // 2)
        draw_x2 = min(w, tx + bw // 2)
        draw_y2 = min(h, ty + bh // 2)

        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = "LOCKED" if lost_count == 0 else "HOLDING..."

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

    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.9 * fps + 0.1 * instant_fps

    cv2.putText(frame, f"EKRAN FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Alcor Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ai_running = False
cam.release()
cv2.destroyAllWindows()
