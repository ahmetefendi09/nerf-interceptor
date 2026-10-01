import cv2
import time
from ultralytics import YOLO

CAM_ID = 2
model = YOLO("runs/detect/dart_final/weights/best.pt")

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Histerezis eşikleri
CONF_ACQUIRE = 0.30  # İlk kilitlenme için gereken güven
CONF_HOLD    = 0.15  # Kilitlendikten sonra hedefi tutma eşiği

# Takip hafızası
locked_target = None
lost_count = 0
MAX_LOST_FRAMES = 5  # Dart anlık kaybolursa 5 kare boyunca yeri koru

prev_t = time.time()
print("Yapışkan Takip (Sticky Lock) Devrede. Çıkış: 'q'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Kilitliysek düşük eşikle (%15), kilitli değilsek standart eşikle (%30) tara
    current_threshold = CONF_HOLD if locked_target is not None else CONF_ACQUIRE
    results = model.predict(source=frame, conf=current_threshold, imgsz=640, verbose=False)

    best_candidate = None
    max_c = 0.0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        bw = x2 - x1
        bh = y2 - y1

        # Aşırı büyük şekilleri ele
        if (bw * bh) > (640 * 480 * 0.15):
            continue

        if conf > max_c:
            max_c = conf
            best_candidate = (x1, y1, x2, y2, conf)

    # Durum Yönetimi (State Machine)
    if best_candidate is not None:
        # Dart bulundu -> Kilidi güncelle
        locked_target = best_candidate
        lost_count = 0
    else:
        # Dart bu karede kaçtı -> Hafızadan devam et
        if locked_target is not None:
            lost_count += 1
            if lost_count > MAX_LOST_FRAMES:
                locked_target = None  # Gerçekten sahneden çıktı, kilidi bırak

    # Çizimler
    if locked_target is not None:
        x1, y1, x2, y2, conf = locked_target
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2

        # Kayıp toleransındayken sarı, tam kilitteyken yeşil kutu
        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_text = f"LOCKED %{int(conf*100)}" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_text, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.line(frame, (320, 240), (tx, ty), (255, 0, 0), 2)

    cv2.drawMarker(frame, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cur_t = time.time()
    fps = 1.0 / (cur_t - prev_t) if (cur_t - prev_t) > 0 else 0
    prev_t = cur_t

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Dart Sticky Lock", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
