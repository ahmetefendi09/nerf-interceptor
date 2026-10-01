import os
import shutil
import random

random.seed(42)

# Senin fotoğraflarının tam konumu
kaynak = "/home/ahmetbey/dart"
hedef_train = "dataset/images/train"
hedef_val = "dataset/images/val"

tum_gorseller = [f for f in os.listdir(kaynak) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(tum_gorseller)

secilenler = tum_gorseller[:250]

for d in secilenler[:200]:
    shutil.copy(os.path.join(kaynak, d), os.path.join(hedef_train, d))

for d in secilenler[200:]:
    shutil.copy(os.path.join(kaynak, d), os.path.join(hedef_val, d))

print(f"Başarılı! {len(secilenler[:200])} train ve {len(secilenler[200:])} val görseli ayrıldı.")
