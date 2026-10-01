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

# Model seçimi
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel: {model_path}")
print("Girdi Boyutu: 640x640 (Orijinal Letterbox Formatı)")
model = YOLO(model_path, task="detect")

# Güven eşiği: Parazitleri ve alakasız nesneleri kesmek için %40
CONF_THRESHOLD = 0.40

prev_t = time.time()
fps = 0.0

def make_letterbox(img, target_size=640):
    """640x480 görüntüyü oran bozmadan 640x640 kare tuvale oturtur (80px üst/alt dolgu)."""
    h, w = img.shape[:2]
    # 640x640 siyah tuval
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # 480 yüksekliği ortala: (640 - 480) / 2 = 80 piksel ofset
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- STANDART 640x640 DOĞRU ORAN KİLİT MOTORU ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    # Bayat kareleri atlayıp tam anlık kareyi al
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. 640x640 Letterbox üretimi
    letterbox_frame, y_offset = make_letterbox(frame, target_size=640)

    # 2. Tam 640x640 boyutunda çıkarım
    results = model.predict(source=letterbox_frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

    best_target = None
    max_c = 0.0

    for box in results[0].boxes:
        # Koordinatlar 640x640 tuvaline göre gelir
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        # Koordinatları orijinal 640x480 kameraya geri haritala (y_offset çıkar)
        x1 = bx1
        x2 = bx2
        y1 = by1 - y_offset
        y2 = by2 - y_offset

        # Eğer tespit üst veya alt siyah dolguya taştıysa kırp
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # Çok devasa alanları ele
        if area > (w * h * 0.12) or bw < 8 or bh < 8:
            continue

        if conf > max_c:
            max_c = conf
            best_target = (x1, y1, x2, y2, conf)

    # Çizimler doğrudan orijinal frame üzerine yapılır
    if best_target is not None:
        x1, y1, x2, y2, conf = best_target
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
        cv2.putText(frame, "SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Exact 640 Letterbox Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
