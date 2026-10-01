import cv2
import glob
import os
import shutil

RAW_DIR = "dataset_finetune/raw"
IMG_TRAIN = "dataset_finetune/images/train"
LBL_TRAIN = "dataset_finetune/labels/train"
IMG_VAL = "dataset_finetune/images/val"
LBL_VAL = "dataset_finetune/labels/val"

for p in [IMG_TRAIN, LBL_TRAIN, IMG_VAL, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

images = sorted(glob.glob(f"{RAW_DIR}/*.jpg"))
total = len(images)
print(f"\nEtiketlenecek {total} görsel bulundu.")
print("Kontroller: Fare ile sürükleyip kutu çiz | [Space] = Sonraki | [r] = Sıfırla | [d] = Boş Geç | [q] = Çıkış\n")

ix, iy, x2, y2 = -1, -1, -1, -1
drawing = False
current_box = None

def mouse_draw(event, x, y, flags, param):
    global ix, iy, x2, y2, drawing, current_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        xmin, xmax = min(ix, x2), max(ix, x2)
        ymin, ymax = min(iy, y2), max(iy, y2)
        if (xmax - xmin) > 5 and (ymax - ymin) > 5:
            current_box = (xmin, ymin, xmax, ymax)

cv2.namedWindow("Etiketleyici")
cv2.setMouseCallback("Etiketleyici", mouse_draw)

for idx, img_path in enumerate(images):
    current_box = None
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]

    while True:
        disp = frame.copy()
        if drawing and ix != -1:
            cv2.rectangle(disp, (ix, iy), (x2, y2), (0, 255, 255), 2)
        elif current_box is not None:
            cv2.rectangle(disp, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)

        cv2.putText(disp, f"[{idx+1}/{total}] Kutu ciz -> [Space]: Kaydet, [r]: Sil, [d]: Bos Gec",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Etiketleyici", disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (32, 13): # Space veya Enter
            # %80 train, %20 val
            target_img_dir = IMG_VAL if (idx % 5 == 0) else IMG_TRAIN
            target_lbl_dir = LBL_VAL if (idx % 5 == 0) else LBL_TRAIN

            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(target_img_dir, base_name))

            txt_name = base_name.rsplit(".", 1)[0] + ".txt"
            txt_path = os.path.join(target_lbl_dir, txt_name)

            with open(txt_path, "w") as f:
                if current_box is not None:
                    bx1, by1, bx2, by2 = current_box
                    cx = ((bx1 + bx2) / 2.0) / w
                    cy = ((by1 + by2) / 2.0) / h
                    bw = (bx2 - bx1) / float(w)
                    bh = (by2 - by1) / float(h)
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            break

        elif key == ord('r'):
            current_box = None
            ix, iy = -1, -1

        elif key == ord('d'): # Boş negatif kare
            target_img_dir = IMG_VAL if (idx % 5 == 0) else IMG_TRAIN
            target_lbl_dir = LBL_VAL if (idx % 5 == 0) else LBL_TRAIN
            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(target_img_dir, base_name))
            txt_name = base_name.rsplit(".", 1)[0] + ".txt"
            open(os.path.join(target_lbl_dir, txt_name), "w").close()
            break

        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

cv2.destroyAllWindows()
print("\nEtiketleme tamamlandı. Veri seti hazır!")
