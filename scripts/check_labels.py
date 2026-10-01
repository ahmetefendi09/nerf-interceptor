import cv2
import glob
import os

# Hem büyük hem küçük harf uzantıları yakala
images = sorted(glob.glob("dataset/images/train/*.[jJ][pP][gG]")) + \
         sorted(glob.glob("dataset/images/train/*.[pP][nN][gG]"))

print(f"Toplam bulunan görsel: {len(images)}")
print("Sonraki kare için herhangi bir tuşa (Space/Enter), çıkış için 'q' tuşuna bas.\n")

for i, img_path in enumerate(images[:40]):
    # Etiket dosyasını bul (dataset/images/train/... -> dataset/labels/train/...)
    label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]

    box_count = 0
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    box_count += 1
                    _, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Dart Kutu {box_count}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f"Gorsel: {os.path.basename(img_path)} ({i+1}/40) - Kutular: {box_count}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Etiket Dogrulama", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
