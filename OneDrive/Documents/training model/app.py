from ultralytics import YOLO
import cv2

# Gunakan .pt agar library menangani preprocessing secara otomatis
model = YOLO("retrain2.onnx") 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Gunakan .predict dengan threshold 0.4 agar deteksi lebih bersih
    results = model.predict(frame, conf=0.4, verbose=False)
    
    # Plot hasil deteksi ke frame
    annotated_frame = results[0].plot()

    cv2.imshow("Deteksi", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()