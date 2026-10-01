import cv2
import os
import time

SAVE_DIR = "dataset_finetune/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = len(os.listdir(SAVE_DIR))
print("\n--- KARE TOPLAMA DEVREDE ---")
print("Dartı kameraya dik, çapraz, yakın ve uzak konumlarda tut.")
print("Kare kaydetmek için [s], çıkmak için [q] tuşuna bas.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()
    cv2.putText(disp, f"Kaydedilen: {count} / 50+", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Alcor Kare Toplama", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        count += 1
        path = f"{SAVE_DIR}/dart_live_{int(time.time()*1000)}.jpg"
        cv2.imwrite(path, frame)
        print(f"[{count}] Kaydedildi -> {path}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nToplam {count} kare toplandı.")
