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

# En optimize modeli seç (OpenVINO -> ONNX -> PT)
model_path = "runs/detect/dart_finetuned/weights/best_openvino_model"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.onnx"
if not os.path.exists(model_path):
    model_path = "runs/detect/dart_finetuned/weights/best.pt"

if not os.path.exists(model_path):
    print(f"HATA: Model dosyasi bulunamadi! runs/detect klasorunu kontrol et.")
    exit(1)

print(f"\nModel: {model_path} (Dinamik ROI + Kalman Kararli Sürüm)")
model = YOLO(model_path, task="detect")

# Kalman Filtresi Kurulumu (x, y, dx, dy)
# Hareket tahmini (processNoiseCov) artırıldı, ölçüm gürültüsü (measurementNoiseCov) optimize edildi
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.45

is_tracking = False
lost_count = 0
MAX_LOST = 10  # 15 FPS için ~0.7 saniye kilit koruması
last_center = None
last_box_dims = (40, 40)
ROI_SIZE = 300  # Kilit sonrası taranacak pencere boyutu (biraz büyütüldü)

CONF_ACQUIRE = 0.38
CONF_HOLD    = 0.22

prev_t = time.time()
fps = 0.0

def make_letterbox(img, target_size=640):
    """640x480 görüntüyü oran bozmadan 640x640 kare tuvale oturtur."""
    h, w = img.shape[:2]
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- DINAMIK ROI & KARARLI KALMAN MOTORU DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    cap.grab() # Bayat kare tahliyesi
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Kalman Gelecek Adım Tahmini (HATA DÜZELTİLDİ)
    predicted = kalman.predict()
    # OpenCV sürümleri arasındaki çıktı farkını (scalar vs array) handle et
    try:
        pred_x = int(predicted[0])
        pred_y = int(predicted[1])
    except (TypeError, IndexError):
        pred_x = int(predicted.item(0))
        pred_y = int(predicted.item(1))

    # Kırpma (ROI) Koordinatları Belirleme
    use_roi = is_tracking and (last_center is not None)
    if use_roi:
        # Kırpma bölgesini tahmin edilen merkez etrafına kur, kadraj dışına taşma
        rx = max(0, min(w - ROI_SIZE, pred_x - ROI_SIZE // 2))
        ry = max(0, min(h - ROI_SIZE, pred_y - ROI_SIZE // 2))
        input_crop = frame[ry:ry+ROI_SIZE, rx:rx+ROI_SIZE]
        offset_x, offset_y = rx, ry
    else:
        input_crop = frame
        offset_x, offset_y = 0, 0

    # Letterbox & Çıkarım (Eğitim geometrisine uygun: Letterbox + 640x640)
    lb_img, y_off = make_letterbox(input_crop, target_size=640)
    active_conf = CONF_HOLD if is_tracking else CONF_ACQUIRE
    results = model.predict(source=lb_img, conf=active_conf, imgsz=640, verbose=False)

    best_cand = None
    max_c = 0.0

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())

        # 1. Koordinatları Letterbox'tan input_crop'a haritala
        by1 = max(0, min(input_crop.shape[0], by1 - y_off))
        by2 = max(0, min(input_crop.shape[0], by2 - y_off))

        # 2. Koordinatları input_crop'tan tam ekrana haritala
        x1 = bx1 + offset_x
        x2 = bx2 + offset_x
        y1 = by1 + offset_y
        y2 = by2 + offset_y

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # Alan sınırları (dev yüz/gövde ve minik parazit elenir)
        if area > (w * h * 0.12) or bw < 8 or bh < 8:
            continue
        
        # Geometrik filtre (Uzun ince silindir kontrolü)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
        # Kilit oturmuşsa filtreden geçsin, arama modundaysa dairesel şekilleri (kulak) ele
        if not is_tracking and area > 400 and aspect < 1.30:
            continue

        if conf > max_c:
            max_c = conf
            best_cand = (x1, y1, x2, y2, conf)

    # Durum ve Kalman Güncelleme
    if best_cand is not None:
        x1, y1, x2, y2, conf = best_cand
        meas_x = (x1 + x2) // 2
        meas_y = (y1 + y2) // 2
        last_box_dims = (x2 - x1, y2 - y1)

        if not is_tracking:
            # İlk kilitte filtre durumunu hedefin merkezine sıfırla
            kalman.statePre = np.array([[meas_x], [meas_y], [0], [0]], np.float32)
            kalman.statePost = np.array([[meas_x], [meas_y], [0], [0]], np.float32)
            is_tracking = True

        measurement = np.array([[np.float32(meas_x)], [np.float32(meas_y)]])
        corrected = kalman.correct(measurement)
        
        # Çıktı formatı farkını handle et
        try:
            target_x, target_y = int(corrected[0]), int(corrected[1])
        except (TypeError, IndexError):
            target_x, target_y = int(corrected.item(0)), int(corrected.item(1))

        last_center = (target_x, target_y)
        lost_count = 0
    else:
        if is_tracking:
            lost_count += 1
            # Model kaçırsa bile Kalman tahminiyle hedefi sürükle
            target_x, target_y = pred_x, pred_y
            last_center = (target_x, target_y)
            if lost_count > MAX_LOST:
                is_tracking = False
                last_center = None

    # Çizimler
    if is_tracking and last_center is not None:
        # Taranan ROI kutusu (Mavi - sadece debug için)
        if use_roi:
            cv2.rectangle(frame, (offset_x, offset_y), 
                          (offset_x + ROI_SIZE, offset_y + ROI_SIZE), (255, 100, 0), 1)

        bw, bh = last_box_dims
        draw_x1 = max(0, target_x - bw // 2)
        draw_y1 = max(0, target_y - bh // 2)
        draw_x2 = min(w, target_x + bw // 2)
        draw_y2 = min(h, target_y + bh // 2)

        box_color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status_txt = "LOCKED (ROI)" if lost_count == 0 else "PREDICTING..."

        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), box_color, 2)
        cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, status_txt, (draw_x1, max(22, draw_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.line(frame, (cx, cy), (target_x, target_y), (255, 0, 0), 2)

        err_x = target_x - cx
        err_y = target_y - cy
        cv2.putText(frame, f"Sapma X: {err_x:+d} | Y: {err_y:+d}", (20, 75),
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

    cv2.imshow("Nerf Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
