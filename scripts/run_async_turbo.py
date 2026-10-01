import os
import glob
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

# Wayland ve font çakışmasını engelle
os.environ["QT_QPA_PLATFORM"] = "xcb"

def find_camera():
    devs = sorted(glob.glob("/dev/video*"))
    for d in devs:
        try:
            num = int(d.replace("/dev/video", ""))
        except ValueError:
            continue
        c = cv2.VideoCapture(num, cv2.CAP_V4L2)
        if c.isOpened():
            ret, _ = c.read()
            c.release()
            if ret:
                return num
    return 2

cam_id = find_camera()
print(f"Aktif Kamera: /dev/video{cam_id}")

class FastCamera:
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

model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"Model Yüklendi: {model_path}")
model = YOLO(model_path, task="detect")

cam = FastCamera(cam_id)

latest_target = None
target_lock = threading.Lock()
ai_running = True

def make_letterbox(img, target_size=640):
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

def ai_inference_loop():
    global latest_target, ai_running
    while ai_running:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        lb_frame, y_offset = make_letterbox(frame, target_size=640)

        # 640x640 tam boyut çıkarım
        results = model.predict(source=lb_frame, conf=0.32, imgsz=640, verbose=False)

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

threading.Thread(target=ai_inference_loop, daemon=True).start()

# Takip Yumuşatma
target_pos = None
velocity = np.array([0.0, 0.0])
box_dims = (40, 40)
lost_count = 0
MAX_LOST = 8

prev_t = time.time()
fps = 0.0

print("\n--- ASENKRON ÇİFT MOTOR DEVREDE ---")
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

    cv2.imshow("Async Turbo Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ai_running = False
cam.release()
cv2.destroyAllWindows()
