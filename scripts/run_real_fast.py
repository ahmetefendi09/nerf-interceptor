import os
import time
import subprocess
import cv2
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

def get_alcor_camera_id():
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
        lines = out.split("\n")
        target = False
        for line in lines:
            if "alcor" in line.lower():
                target = True
                continue
            if target and "/dev/video" in line:
                return int(line.strip().replace("/dev/video", ""))
            if target and line.strip() == "":
                target = False
    except Exception:
        pass
    return 2

cam_id = get_alcor_camera_id()
print(f"[OK] Kamera: /dev/video{cam_id}")

cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"HATA: /dev/video{cam_id} açılamadı!")
    exit(1)

model_dir = "runs/detect/dart_finetuned/weights/best_openvino_model"
xml_path = os.path.join(model_dir, "best.xml")

if not os.path.exists(xml_path):
    print(f"HATA: {xml_path} bulunamadı!")
    exit(1)

# OpenVINO doğrudan donanıma bağlanıyor (Latency modu ile anlık tepki)
core = Core()
core.set_property("CPU", {
    "INFERENCE_NUM_THREADS": "8",
    "PERFORMANCE_HINT": "LATENCY"
})

ov_model = core.read_model(model=xml_path)
compiled_model = core.compile_model(model=ov_model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Eşikler
CONF_ACQUIRE = 0.35
CONF_HOLD    = 0.20
MAX_JUMP     = 130.0

is_locked = False
target_box = None
lost_count = 0
MAX_LOST = 4

prev_t = time.time()
fps = 0.0

def make_letterbox(img):
    h, w = img.shape[:2]
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    y_offset = (640 - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

print("\n--- SAF OPENVINO HER KAREDE ANLIK KİLİT DEVREDE ---")
print("Kare atlama yok | Tam CPU çıkarımı | Çıkış: 'q'\n")

while cap.isOpened():
    t_start = time.time()

    # Donanım tamponundaki bayat kareyi tahliye et (Lag kesici)
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Hızlı Tensör Hazırlığı
    lb_frame, y_offset = make_letterbox(frame)
    blob = lb_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    # 2. HER KAREDE DOĞRUDAN ÇIKARIM
    infer_request.infer(inputs={input_layer.any_name: blob})
    output = infer_request.get_output_tensor(output_layer.index).data[0]

    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]
    confs = np.max(scores, axis=1)

    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE
    mask = confs > active_conf

    valid_boxes = boxes[mask]
    valid_confs = confs[mask]

    best_match = None

    if len(valid_confs) > 0:
        # NMS için formatlama: [x, y, w, h] -> [x1, y1, w, h]
        nms_boxes = []
        for b in valid_boxes:
            bx, by, bw, bh = b
            nms_boxes.append([int(bx - bw / 2), int(by - bh / 2), int(bw), int(bh)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, valid_confs.tolist(), active_conf, 0.40)

        if len(indices) > 0:
            candidates = []
            for i in np.asarray(indices).flatten():
                idx = int(i)
                bx, by, bw, bh = nms_boxes[idx]
                cf = float(valid_confs[idx])

                x1 = max(0, min(w, bx))
                y1 = max(0, min(h, by - y_offset))
                x2 = max(0, min(w, bx + bw))
                y2 = max(0, min(h, by + bh - y_offset))

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                candidates.append((np.array([x1, y1, x2, y2], dtype=np.float32), cf))

            if is_locked and target_box is not None:
                # Kilitliyken en yakın adayı anında seç
                curr_cx = (target_box[0] + target_box[2]) / 2.0
                curr_cy = (target_box[1] + target_box[3]) / 2.0
                best_dist = MAX_JUMP

                for b, cf in candidates:
                    b_cx = (b[0] + b[2]) / 2.0
                    b_cy = (b[1] + b[3]) / 2.0
                    dist = np.hypot(b_cx - curr_cx, b_cy - curr_cy)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = b
            else:
                # Arama modundayken en yüksek güvenli olanı seç
                max_c = 0.0
                for b, cf in candidates:
                    if cf > max_c:
                        max_c = cf
                        best_match = b

    # Anlık Kilit Güncelleme (Hantallaştıran ağır filtreler kaldırıldı)
    if best_match is not None:
        if not is_locked or target_box is None:
            target_box = best_match
            is_locked = True
        else:
            # %85 canlı ölçüm: Gecikme hissini tamamen yok eder, dartın peşinden akar
            target_box = 0.85 * best_match + 0.15 * target_box

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_box = None

    # Çizim & Hedef Koordinatları
    if is_locked and target_box is not None:
        x1, y1, x2, y2 = map(int, target_box)
        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2

        color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status = "LOCKED" if lost_count == 0 else "HOLD"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)
        cv2.putText(frame, status, (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

    cv2.imshow("Real Fast Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
