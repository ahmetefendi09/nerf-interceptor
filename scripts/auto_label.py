import os
import shutil
from ultralytics import YOLO

# Eğitilen ön modeli yükle
model = YOLO("runs/detect/dart_baseline/weights/best.pt")

kaynak_klasor = "/home/ahmetbey/dart"
train_img_klasor = "dataset/images/train"
train_lbl_klasor = "dataset/labels/train"

# Zaten etiketlenmiş olan ilk 250 görseli atla
mevcut_gorseller = set(os.listdir(train_img_klasor) + os.listdir("dataset/images/val"))

tum_dosyalar = [f for f in os.listdir(kaynak_klasor) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
kalan_dosyalar = [f for f in tum_dosyalar if f not in mevcut_gorseller]

print(f"Toplam {len(kalan_dosyalar)} yeni görsel otomatik etiketlenecek...")

sayac = 0
for dosya in kalan_dosyalar:
    gorsel_yolu = os.path.join(kaynak_klasor, dosya)
    
    # Güven eşiğini 0.30 tutuyoruz ki dartı rahat yakalasın
    results = model.predict(source=gorsel_yolu, conf=0.30, imgsz=640, verbose=False)
    
    txt_adi = os.path.splitext(dosya)[0] + ".txt"
    txt_yolu = os.path.join(train_lbl_klasor, txt_adi)
    
    with open(txt_yolu, "w", encoding="utf-8") as f:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
    shutil.copy(gorsel_yolu, os.path.join(train_img_klasor, dosya))
    sayac += 1
    
    if sayac % 250 == 0:
        print(f"[{sayac}/{len(kalan_dosyalar)}] görsel işlendi...")

print(f"Bitti! Toplam {sayac} görsel otomatik etiketlendi ve veri setine eklendi.")
