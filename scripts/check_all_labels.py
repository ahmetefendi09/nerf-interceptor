import cv2
import glob
import os

# Train ve val içindeki tüm görselleri topla
images = sorted(glob.glob("dataset/images/train/*.[jJ][pP][gG]")) + \
         sorted(glob.glob("dataset/images/train/*.[pP][nN][gG]")) + \
         sorted(glob.glob("dataset/images/val/*.[jJ][pP][gG]")) + \
         sorted(glob.glob("dataset/images/val/*.[pP][nN][gG]"))

total = len(images)
if total == 0:
    print("Görsel bulunamadı!")
    exit()

print(f"Toplam {total} görsel yüklendi.")
print("Kontroller:")
print("  [D] veya [Sağ Ok] / [Space] : Sonraki görsel")
print("  [A] veya [Sol Ok]          : Önceki görsel")
print("  [Q]                        : Çıkış\n")

idx = 0
while 0 <= idx < total:
    img_path = images[idx]
    label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    frame = cv2.imread(img_path)
    
    if frame is None:
        idx += 1
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
                    cv2.putText(frame, "Dart", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Bilgi çubuğu
    cv2.putText(frame, f"[{idx+1}/{total}] {os.path.basename(img_path)} | Kutular: {box_count}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Tum Veri Seti Kontrol", frame)
    key = cv2.waitKey(0) & 0xFF

    if key in (ord('d'), 83, 32):  # 'd', Sağ Ok, Space -> İleri
        idx += 1
    elif key in (ord('a'), 81):    # 'a', Sol Ok -> Geri
        idx = max(0, idx - 1)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
