from ultralytics import YOLO


model = YOLO("DAAM-TDH.yaml")
model.load("yolov8n.pt")

model.train(
    data="CORS-SAR.yaml",
    epochs=150,
    patience=40,
    batch=16,
    imgsz=640,
    project="runs/ablation/DAAM+TDH/epoch150",
)
