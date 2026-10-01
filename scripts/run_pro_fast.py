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

# OpenVINO C++ Çekirdeği (Tam CPU İzlekleri)
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
CONF_ACQUIRE = 0.38
CONF_HOLD    = 0.22
MAX_JUMP     = 140.0

is_locked = False
target_box = None
lost_count = 0
MAX_LOST = 4

# Bellek optimizasyonu: Her karede yeniden matris tahsis etmemek için sabit tuval
canvas = np.zeros((1, 3, 640, 640), dtype=np.float32)
y_offset = (640 - 480) // 2

prev_t = time.time()
fps = 0.0

print("\n--- OPTİMİZE EDİLMİŞ AKICI NİŞANGAH DEVREDE ---")
print("Çıkış için 'q' tuşuna bas.\n")

while cap.isOpened():
    t_start = time.time()

    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Hızlı tensör hazırlığı (Zero-Copy Buffer)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    canvas[0, :, :y_offset, :] = 0
    canvas[0, :, y_offset+h:, :] = 0
    canvas[0, :, y_offset:y_offset+h, :] = np.transpose(rgb_frame, (2, 0, 1))

    # C++ Çıkarım
    infer_request.infer(inputs={input_layer.any_name: canvas})
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
        bx = valid_boxes[:, 0]
        by = valid_boxes[:, 1]
        bw = valid_boxes[:, 2]
        bh = valid_boxes[:, 3]

        x1_arr = np.clip(bx - bw / 2, 0, w)
        y1_arr = np.clip(by - bh / 2 - y_offset, 0, h)
        w_arr = np.clip(bw, 0, w)
        h_arr = np.clip(bh, 0, h)

        nms_boxes = np.column_stack((x1_arr, y1_arr, w_arr, h_arr)).astype(int).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, valid_confs.tolist(), active_conf, 0.40)

        if len(indices) > 0:
            candidates = []
            for i in np.asarray(indices).flatten():
                idx = int(i)
                b = nms_boxes[idx]
                cf = float(valid_confs[idx])

                x1, y1, bw_box, bh_box = b
                x2 = min(w, x1 + bw_box)
                y2 = min(h, y1 + bh_box)

                if bw_box < 8 or bh_box < 8:
                    continue

                candidates.append((np.array([x1, y1, x2, y2], dtype=np.float32), cf))

            if is_locked and target_box is not None:
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
                max_c = 0.0
                for b, cf in candidates:
                    if cf > max_c:
                        max_c = cf
                        best_match = b

    # Dinamik Kilit Güncelleme
    if best_match is not None:
        if not is_locked or target_box is None:
            target_box = best_match
            is_locked = True
        else:
            delta = best_match - target_box
            speed = np.linalg.norm(delta[:2])

            # Hareket hızına göre dinamik tepki
            if speed > 20.0:
                alpha = 0.90  # Hızlı savrulmada sıfır gecikme
            elif speed > 5.0:
                alpha = 0.70  # Normal takip
            else:
                alpha = 0.35  # Titreşim önleme

            target_box = (1.0 - alpha) * target_box + alpha * best_match

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_box = None

    # Çizim ve Hedef Bilgileri
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

    cv2.imshow("Fast Interceptor Pro", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
