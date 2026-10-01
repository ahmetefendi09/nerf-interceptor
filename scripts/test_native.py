import cv2
from ultralytics import YOLO

# Doğrudan orijinal eğitilen PyTorch ağırlığını yüklüyoruz
model = YOLO("runs/detect/dart_final/weights/best.pt")

# Alcor PC Camera
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- STANDART ULTRALYTICS PIPELINE TESTI ---")
print("Çıkış için pencere üzerindeyken 'q' tuşuna bas.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ultralytics'in kendi görselleştirme fonksiyonunu kullanıyoruz
    results = model(frame, conf=0.35, imgsz=640, verbose=False)
    
    # Modelin kendi çizdiği kutular ve etiketler
    annotated_frame = results[0].plot()

    cv2.imshow("Dart Native Test", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
