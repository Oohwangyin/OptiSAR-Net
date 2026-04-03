from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml").load('yolov8m.pt')

class_weights = [1.5, 1.7, 5.0, 2.5, 0.8, 5.0, 3.0, 1.5, 5.0, 1.0, 2.5, 5.0, 5.0, 3.0, 5.0]
model = YOLO("OptiSAR-Net.yaml")

model.train(data="DOTAv1.yaml", epochs=100, batch=16, imgsz=640, resume=True)