import cv2
import time
import os
import threading
import numpy as np
from ultralytics import YOLO

SRC_PANASONIC = 0
SRC_ALCOR = 2

class KameraAkisi:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.002)

    def oku(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def durdur(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

model_path = "runs/detect/dart_final/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_final/weights/best.pt"

print(f"Yüklenen Model: {model_path}")
model = YOLO(model_path)

cam_top = KameraAkisi(SRC_PANASONIC)
cam_bot = KameraAkisi(SRC_ALCOR)

# Eşiği dartı kesin yakalaması için %25'e düşürdük
CONF_THRESHOLD = 0.25

# Hedef hafızası (titremeyi ve yanıp sönmeyi bitirir)
track_top = None
track_bot = None
miss_top = 0
miss_bot = 0

prev_t = time.time()
frame_count = 0

def dart_filtrele(results, w, h):
    en_iyi_kutu = None
    max_c = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1
        b_area = bw * bh

        # 1. Ekranın %10'undan büyük devasa kutuları ele (yüz, kafa, gövde)
        if b_area > (w * h * 0.10) or bw < 8 or bh < 8:
            continue

        # 2. Geometrik Oran Kontrolü:
        # Kulak, dil ve el genelde yayvan/karemsi amorf kütlelerdir.
        # Dart ya uzundur (oran > 1.8) ya da karşıdan bakınca minik dairesel bir noktadır.
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        
        # Eğer kutu orta büyüklükteyse ve en/boy oranı 1'e çok yakınsa (kulak memesi/dil) ele
        if b_area > 1500 and aspect < 1.35:
            continue

        if conf > max_c:
            max_c = conf
            en_iyi_kutu = (x1, y1, x2, y2, conf)

    return en_iyi_kutu

print("Optimize Takip Devrede. Çıkış: 'q'")

while True:
    f_top = cam_top.oku()
    f_bot = cam_bot.oku()

    if f_top.shape[:2] != (480, 640):
        f_top = cv2.resize(f_top, (640, 480))
    if f_bot.shape[:2] != (480, 640):
        f_bot = cv2.resize(f_bot, (640, 480))

    frame_count += 1

    # CPU'yu boğmamak için kameraları sırayla modele gönderiyoruz (Interleaving)
    if frame_count % 2 == 0:
        res_top = model.predict(source=f_top, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        kutu_top = dart_filtrele(res_top, 640, 480)
        if kutu_top:
            track_top = kutu_top
            miss_top = 0
        else:
            miss_top += 1
            if miss_top > 4:
                track_top = None
    else:
        res_bot = model.predict(source=f_bot, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        kutu_bot = dart_filtrele(res_bot, 640, 480)
        if kutu_bot:
            track_bot = kutu_bot
            miss_bot = 0
        else:
            miss_bot += 1
            if miss_bot > 4:
                track_bot = None

    # Çizimler - Üst Kamera
    if track_top:
        x1, y1, x2, y2, c = track_top
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(f_top, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(f_top, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(f_top, f"Dart %{int(c*100)}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(f_top, (320, 240), (tx, ty), (255, 0, 0), 2)

    # Çizimler - Alt Kamera
    if track_bot:
        x1, y1, x2, y2, c = track_bot
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(f_bot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(f_bot, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(f_bot, f"Dart %{int(c*100)}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(f_bot, (320, 240), (tx, ty), (255, 0, 0), 2)

    # Merkez Nişangahları
    cv2.drawMarker(f_top, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.drawMarker(f_bot, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # FPS Gösterimi
    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(f_top, f"PANASONIC (UST) | FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(f_bot, f"ALCOR (ALT)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if track_top and track_bot:
        cv2.putText(f_bot, "DUAL LOCK ENGAGED", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    ekran = np.hstack((f_top, f_bot))
    cv2.imshow("Nerf Interceptor", ekran)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_top.durdur()
cam_bot.durdur()
cv2.destroyAllWindows()
