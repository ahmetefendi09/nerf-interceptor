import cv2
import time
import os
from ultralytics import YOLO

CAM_ID = 2

# Kamerayı doğrudan en düşük gecikmeyle aç
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Model yükleme
model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

print(f"\nModel: {model_path}")
model = YOLO(model_path)

prev_t = time.time()
print("--- SIFIR GECIKME (ZERO-LAG) DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    # 1. BUFFER TAHLIYESI: Birikmiş bayat kareleri atlayıp tam şu anki kareyi al
    for _ in range(2):
        cap.grab()
    ret, frame = cap.retrieve()

    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 2. HIZLI INFERENCE: imgsz=320 yaparak CPU gecikmesini 4 kat düşürüyoruz
    results = model.predict(source=frame, conf=0.18, imgsz=320, verbose=False)

    best_target = None
    max_conf = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # Alan filtresi (tüm yüz/gövde elensin)
        if area > (w * h * 0.12) or bw < 6 or bh < 6:
            continue

        # Kulak filtresi: Kulak orta boy karemsidir
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        if area > 450 and aspect < 1.30:
            continue

        if conf > max_conf:
            max_conf = conf
            best_target = (x1, y1, x2, y2, conf)

    # Hedef varsa anında çiz
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
        cv2.putText(frame, f"Sapma X: {err_x:+d} | Y: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ARANIYOR...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Zero-Lag Dart Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
