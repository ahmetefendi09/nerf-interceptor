import os
import json

def convert_dir(img_dir, lbl_dir):
    if not os.path.exists(lbl_dir):
        return
    for f in os.listdir(lbl_dir):
        if not f.endswith(".json"):
            continue
        json_path = os.path.join(lbl_dir, f)
        txt_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")

        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            w = data.get("imageWidth")
            h = data.get("imageHeight")

            # Eğer JSON içinde w, h boşsa görselden çek
            if not w or not h:
                import cv2
                img_file = os.path.join(img_dir, os.path.splitext(f)[0] + ".jpg")
                if not os.path.exists(img_file):
                    img_file = os.path.join(img_dir, os.path.splitext(f)[0] + ".png")
                img = cv2.imread(img_file)
                if img is not None:
                    h, w = img.shape[:2]

            with open(txt_path, "w", encoding="utf-8") as tf:
                for shape in data.get("shapes", []):
                    if shape.get("shape_type") == "rectangle":
                        (x1, y1), (x2, y2) = shape["points"]
                        xmin, xmax = min(x1, x2), max(x1, x2)
                        ymin, ymax = min(y1, y2), max(y1, y2)

                        x_center = ((xmin + xmax) / 2.0) / w
                        y_center = ((ymin + ymax) / 2.0) / h
                        bw = (xmax - xmin) / w
                        bh = (ymax - ymin) / h

                        tf.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            os.remove(json_path)
        except Exception as e:
            print(f"Hata ({f}): {e}")

convert_dir("dataset/images/train", "dataset/labels/train")
convert_dir("dataset/images/val", "dataset/labels/val")
print("JSON dosyaları YOLO txt formatına dönüştürüldü!")
