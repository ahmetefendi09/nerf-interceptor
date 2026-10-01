import cv2
import time
import os
import glob
import numpy as np

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

def get_working_camera():
    devices = sorted(glob.glob("/dev/video*"))
    for dev in devices:
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            r, _ = c.read()
            c.release()
            if r:
                return idx
    return 2

CAM_ID = get_working_camera()
print(f"Kamera Tespit Edildi: /dev/video{CAM_ID}")

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"HATA: /dev/video{CAM_ID} açılamadı! Kameranın bağlı olduğundan emin ol.")
    exit(1)

model_dir = "runs/detect/dart_finetuned/weights/best_openvino_model"
xml_path = os.path.join(model_dir, "best.xml")

if not os.path.exists(xml_path):
    print(f"HATA: {xml_path} bulunamadı!")
    exit(1)

ie = Core()
ie.set_property("CPU", {
    "INFERENCE_NUM_THREADS": "8",
    "PERFORMANCE_HINT": "LATENCY"
})

model = ie.read_model(model=xml_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print("\n--- SAF OPENVINO 8-THREAD MOTOR DEVREDE ---")
print("Girdi: 640x640 Letterbox | Çıkış: 'q'\n")

CONF_ACQUIRE = 0.40
CONF_HOLD    = 0.25

is_locked = False
target_pos = None
velocity = np.array([0.0, 0.0])
box_dims = (40, 40)
lost_count = 0
MAX_LOST = 8

prev_t = time.time()
fps = 0.0

def make_letterbox(img):
    h, w = img.shape[:2]
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    y_offset = (640 - h) // 2
    canvas[y_offset:y_offset+h, 0:w] = img
    return canvas, y_offset

while cap.isOpened():
    t_start = time.time()

    cap.grab()
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    lb_frame, y_offset = make_letterbox(frame)
    blob = lb_frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    infer_request.infer(inputs={input_layer.any_name: blob})
    output = infer_request.get_output_tensor(output_layer.index).data[0]

    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]
    confidences = np.max(scores, axis=1)

    threshold = CONF_HOLD if is_locked else CONF_ACQUIRE
    mask = confidences > threshold

    filt_boxes = boxes[mask]
    filt_confs = confidences[mask]

    best_cand = None
    if len(filt_confs) > 0:
        nms_boxes = []
        for b in filt_boxes:
            bx, by, bw, bh = b
            nms_boxes.append([int(bx - bw / 2), int(by - bh / 2), int(bw), int(bh)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, filt_confs.tolist(), threshold, 0.45)

        max_c = 0.0
        if len(indices) > 0:
            for i in np.asarray(indices).flatten():
                idx = int(i)
                bx, by, bw, bh = nms_boxes[idx]
                conf = float(filt_confs[idx])

                x1 = bx
                y1 = max(0, min(h, by - y_offset))
                x2 = bx + bw
                y2 = max(0, min(h, by + bh - y_offset))

                area = bw * bh
                if area > (w * h * 0.12) or bw < 8 or bh < 8:
                    continue

                aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                if not is_locked and area > 350 and aspect < 1.30:
                    continue

                if conf > max_c:
                    max_c = conf
                    best_cand = (x1, y1, x2, y2, conf)

    if best_cand is not None:
        x1, y1, x2, y2, conf = best_cand
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        box_dims = (x2 - x1, y2 - y1)

        if not is_locked:
            target_pos = np.array([raw_x, raw_y])
            velocity = np.array([0.0, 0.0])
            is_locked = True
        else:
            new_vel = np.array([raw_x - target_pos[0], raw_y - target_pos[1]])
            vel_mag = np.linalg.norm(new_vel)
            if vel_mag > 70.0:
                new_vel = (new_vel / vel_mag) * 70.0

            velocity = 0.5 * velocity + 0.5 * new_vel
            target_pos = 0.7 * np.array([raw_x, raw_y]) + 0.3 * (target_pos + velocity)

        lost_count = 0
    else:
        if is_locked:
            lost_count += 1
            velocity *= 0.85
            target_pos += velocity
            if lost_count > MAX_LOST:
                is_locked = False
                target_pos = None

    if is_locked and target_pos is not None:
        tx, ty = int(target_pos[0]), int(target_pos[1])
        bw, bh = box_dims
        draw_x1 = max(0, tx - bw // 2)
        draw_y1 = max(0, ty - bh // 2)
        draw_x2 = min(w, tx + bw // 2)
        draw_y2 = min(h, ty + bh // 2)

        color = (0, 255, 0) if lost_count == 0 else (0, 255, 255)
        status = "LOCKED" if lost_count == 0 else "HOLDING..."

        cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
        cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
        cv2.putText(frame, status, (draw_x1, max(22, draw_y1 - 8)),
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

    cv2.imshow("OpenVINO Native 8-Thread", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
