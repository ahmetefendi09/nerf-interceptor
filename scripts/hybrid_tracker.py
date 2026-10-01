import cv2
import time
import os
from ultralytics import YOLO

CAM_ID = 2

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Model yükleme
model_path = "runs/detect/dart_final/weights/best.pt"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_final/weights/best.onnx"

print(f"Model yükleniyor: {model_path}")
model = YOLO(model_path)

prev_t = time.time()
print("\n--- Ultralytics ByteTrack Motoru Devrede ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # persist=True: Kareler arası hedef kimliğini (ID) ve hareket vektörünü korur
    # tracker="bytetrack.yaml": Ultralytics'in dahili yüksek hızlı takipçisi
    results = model.track(
        source=frame,
        conf=0.22,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        verbose=False
    )

    boxes = results[0].boxes
    locked_target = None
    max_c = 0.0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            bw = x2 - x1
            bh = y2 - y1

            # Aşırı büyük şekilleri ele (el, yüz, arka plan)
            if (bw * bh) > (w * h * 0.15) or bw < 8 or bh < 8:
                continue

            # Takip kimliği (Track ID) kontrolü
            track_id = int(box.id[0].item()) if box.id is not None else 0

            if conf > max_c:
                max_c = conf
                locked_target = (x1, y1, x2, y2, conf, track_id)

    # Hedef Kilit Çizimi
    if locked_target is not None:
        x1, y1, x2, y2, conf, t_id = locked_target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        # Kilit kutusu
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"LOCKED [ID:{t_id}] %{int(conf*100)}", (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        # Taret/Gimbal için merkez sapma (Error Offset)
        err_x = tx - cx
        err_y = ty - cy
        cv2.putText(frame, f"Hata X: {err_x:+d} | Y: {err_y:+d}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "HEDEF ARANIYOR...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    # Ekran merkezi nişangahı
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # FPS Hesabı
    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Nerf Interceptor - ByteTrack Kilit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
