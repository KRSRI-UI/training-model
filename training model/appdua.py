from ultralytics import YOLO

model = YOLO("retrain2.onnx")

model.predict(
    source="training.mp4",
    save=True,
    show=True
)