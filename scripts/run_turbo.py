import cv2
import time
import os
import threading
import numpy as np
from ultralytics import YOLO

CAM_ID = 2

class KameraYakala:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.001)

    def oku(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def durdur(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# 1. En hızlı model formatını seç (Önce OpenVINO, yoksa ONNX, son çare PT)
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nSeçilen Turbo Motor: {model_path}")
model = YOLO(model_path, task="detect")

cam = KameraYakala(CAM_ID)

# Asenkron AI Çalışanı (Ekran FPS'ini boğmaz)
current_target = None
target_lock = threading.Lock()
ai_running = True

def ai_loop():
    global current_target, ai_running
    while ai_running:
        ret, frame = cam.oku()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        
        # 320x320 boyutunda yıldırım hızında çıkarım
        results = model.predict(source=frame, conf=0.18, imgsz=320, verbose=False)

        best_cand = None
        max_c = 0.0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            # Filtreler (yüz/gövde ve kulak eleme)
            if area > (w * h * 0.12) or bw < 6 or bh < 6:
                continue

            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
            if area > 450 and aspect < 1.30:
                continue

            if conf > max_c:
                max_c = conf
                best_cand = (x1, y1, x2, y2, conf)

        with target_lock:
            current_target = best_cand

# Arka plan AI thread'ini ateşle
threading.Thread(target=ai_loop, daemon=True).start()

prev_t = time.time()
print("Turbo Sistem Hazır. Çıkış: 'q'\n")

while True:
    ret, frame = cam.oku()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Arka plandan son hedef bilgisini al
    with target_lock:
        target = current_target

    if target is not None:
        x1, y1, x2, y2, conf = target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"DART %{int(conf*100)}", (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ARANIYOR...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cur_t = time.time()
    dt = cur_t - prev_t
    prev_t = cur_t
    fps = 1.0 / dt if dt > 0 else 0

    cv2.putText(frame, f"KAMERA FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Turbo Dart Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ai_running = False
cam.durdur()
cv2.destroyAllWindows()
