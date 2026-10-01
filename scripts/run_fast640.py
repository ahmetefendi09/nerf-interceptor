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
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# En hızlı modeli seç (OpenVINO öncelikli)
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel: {model_path} (Tam 640 Çözünürlük)")
model = YOLO(model_path, task="detect")

# Takip ve durum değişkenleri
locked_box = None
target_center = None
prev_center = None
velocity = (0, 0)
lost_count = 0
frame_idx = 0
MAX_LOST = 8

prev_t = time.time()
fps = 15.0

print("--- TAM ÇÖZÜNÜRLÜK (640x640) AKILLI MOTOR DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    frame_idx += 1

    # Akıllı Çıkarım: Kilitliysek her 2 karede bir YOLO çalıştır,
    # aradaki kareyi hareket vektörüyle (hız tahmini) anında tamamla.
    need_yolo = (locked_box is None) or (frame_idx % 2 == 0)

    if need_yolo:
        # Tam 640x640 boyutuyla, eğitim formatına sadık çıkarım
        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

        best_cand = None
        max_c = 0.0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area > (w * h * 0.12) or bw < 8 or bh < 8:
                continue

            # Kulak filtresi
            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
            if area > 350 and aspect < 1.30:
                continue

            if conf > max_c:
                max_c = conf
                best_cand = (x1, y1, x2, y2, conf)

        if best_cand is not None:
            x1, y1, x2, y2, conf = best_cand
            new_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if target_center is not None:
                # Hız vektörünü hesapla (dx, dy)
                velocity = (new_center[0] - target_center[0], new_center[1] - target_center[1])
            
            target_center = new_center
            locked_box = best_cand
            lost_count = 0
        else:
            if locked_box is not None:
                lost_count += 1
                if lost_count > MAX_LOST:
                    locked_box = None
                    target_center = None
                    velocity = (0, 0)
    else:
        # YOLO'nun uyuduğu kareden hız vektörünü kullanarak konumu güncelle (0 ms gecikme)
        if locked_box is not None and target_center is not None:
            vx, vy = velocity
            x1, y1, x2, y2, conf = locked_box
            x1 += vx
            x2 += vx
            y1 += vy
            y2 += vy
            target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            locked_box = (x1, y1, x2, y2, conf)

    # Görselleştirme
    if locked_box is not None and target_center is not None:
        x1, y1, x2, y2, conf = locked_box
        tx, ty = target_center

        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = f"LOCKED %{int(conf*100)}" if lost_count == 0 else "TRACKING..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"DX: {err_x:+d} | DY: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TARGET SEARCHING...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    dt = time.time() - t_start
    instant_fps = 1.0 / dt if dt > 0 else 0
    fps = 0.85 * fps + 0.15 * instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("640 Full-Res Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
