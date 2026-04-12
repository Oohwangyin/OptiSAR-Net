from ultralytics import YOLO

model = YOLO("yolov8-obb.yaml")

model.train(data="dota8.yaml", epochs=100, batch=32, imgsz=640,  project="runs/dota-yolo")
