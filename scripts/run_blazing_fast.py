import os
import sys
import time
import subprocess
import cv2
import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat

os.environ["QT_QPA_PLATFORM"] = "xcb"

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
print(f"[OK] Kamera Belirlendi: /dev/video{cam_id}")

cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"HATA: /dev/video{cam_id} açılamadı!")
    sys.exit(1)

model_xml = "runs/detect/dart_finetuned/weights/best_openvino_model/best.xml"
print(f"[OK] Saf OpenVINO 640x640 Devrede: {model_xml}")

core = ov.Core()
core.set_property("CPU", {"INFERENCE_NUM_THREADS": 8, "PERFORMANCE_HINT": "LATENCY"})
model = core.read_model(model_xml)

# PrePostProcessor: CPU dönüşüm yükünü sıfırlar, BGR uint8 doğrudan beslenir
ppp = PrePostProcessor(model)
ppp.input().tensor() \
    .set_element_type(ov.Type.u8) \
    .set_layout(ov.Layout("NHWC")) \
    .set_color_format(ColorFormat.BGR)
ppp.input().preprocess() \
    .convert_element_type(ov.Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .scale([255.0, 255.0, 255.0])
ppp.input().model().set_layout(ov.Layout("NCHW"))
model = ppp.build()

compiled_model = core.compile_model(model, "CPU")
infer_request = compiled_model.create_infer_request()

# EŞİKLER: Kulağı dışarıda tutan ve hızlı yakalayan seviye
CONF_ACQUIRE = 0.62   # Kulağın sahte skorlarını tamamen eler
CONF_HOLD    = 0.28   # Kilitlendikten sonra dartı takip eder
MAX_JUMP     = 140.0

is_locked = False
target_box = None
lost_count = 0
MAX_LOST = 4

prev_t = time.time()
fps = 0.0

# 640x640 Letterbox tuvali (önceden tahsis edilmiş sıfır bellek yükü)
lb_canvas = np.zeros((640, 640, 3), dtype=np.uint8)
y_offset = (640 - 480) // 2

print("\n--- SAF OPENVINO 640x640 MOTORU BAŞLATILDI ---")
print("Kulak elendi | Çıkış için: 'q'\n")

while cap.isOpened():
    t_start = time.time()

    # V4L2 donanım kuyruğundaki gecikmeyi sil
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        continue

    h_orig, w_orig = frame.shape[:2]
    cx, cy = w_orig // 2, h_orig // 2

    # 1. Sıfır kopyalama Letterbox (640x480 -> 640x640)
    lb_canvas[y_offset:y_offset + h_orig, 0:w_orig] = frame

    # 2. Donanımsal Çıkarım (uint8 BGR doğrudan tensöre girer, ~10ms)
    input_tensor = ov.Tensor(lb_canvas[None, ...])
    infer_request.infer([input_tensor])
    output = infer_request.get_output_tensor(0).data  # [1, 5, 8400]

    # Transpose: [8400, 5]
    preds = output[0].T

    active_conf = CONF_HOLD if is_locked else CONF_ACQUIRE

    # Vektörize filtreleme
    scores = preds[:, 4]
    mask = scores > active_conf
    matched = preds[mask]
    matched_scores = scores[mask]

    valid_candidates = []

    if len(matched) > 0:
        bx = matched[:, 0]
        by = matched[:, 1] - y_offset
        bw = matched[:, 2]
        bh = matched[:, 3]

        x1 = np.clip(bx - bw / 2.0, 0, w_orig)
        y1 = np.clip(by - bh / 2.0, 0, h_orig)
        x2 = np.clip(bx + bw / 2.0, 0, w_orig)
        y2 = np.clip(by + bh / 2.0, 0, h_orig)

        boxes_for_nms = []
        confs_for_nms = []

        for i in range(len(matched)):
            w_b = x2[i] - x1[i]
            h_b = y2[i] - y1[i]
            sc = float(matched_scores[i])

            if w_b < 10 or h_b < 10:
                continue

            # Dart geometrisi: Boy/En oranı en az 1.45 olmalı (Kulağı ve dairesel nesneleri eler)
            aspect = max(w_b, h_b) / (min(w_b, h_b) + 1e-5)
            if not is_locked and aspect < 1.45:
                continue

            boxes_for_nms.append([int(x1[i]), int(y1[i]), int(w_b), int(h_b)])
            confs_for_nms.append(sc)

        if len(boxes_for_nms) > 0:
            indices = cv2.dnn.NMSBoxes(boxes_for_nms, confs_for_nms, active_conf, 0.35)
            if len(indices) > 0:
                for idx in indices.flatten():
                    b = boxes_for_nms[idx]
                    valid_candidates.append((
                        np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32),
                        confs_for_nms[idx]
                    ))

    best_match = None

    if is_locked and target_box is not None:
        curr_cx = (target_box[0] + target_box[2]) / 2.0
        curr_cy = (target_box[1] + target_box[3]) / 2.0
        best_dist = MAX_JUMP

        for b, conf in valid_candidates:
            b_cx = (b[0] + b[2]) / 2.0
            b_cy = (b[1] + b[3]) / 2.0
            dist = np.hypot(b_cx - curr_cx, b_cy - curr_cy)
            if dist < best_dist:
                best_dist = dist
                best_match = b
    else:
        max_c = 0.0
        for b, conf in valid_candidates:
            if conf > max_c:
                max_c = conf
                best_match = b

    # Kilit Güncelleme
    if best_match is not None:
        if not is_locked or target_box is None:
            target_box = best_match
            is_locked = True
        else:
            target_box = 0.90 * best_match + 0.10 * target_box
        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            if lost_count > MAX_LOST:
                is_locked = False
                target_box = None

    # Çizim
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

    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {int(dt*1000)}ms", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Native OpenVINO Interceptor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
