from ultralytics import YOLO

model = YOLO("retrain2.pt")
model.export(format="onnx")