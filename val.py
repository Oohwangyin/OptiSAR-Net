from ultralytics import YOLO


model = YOLO("runs/ablation/DAAM+TDH/epoch150/train/weights/best.pt")

model.val(
    data="CORS-SAR-sar.yaml",
    split="test",
    batch=16,
    imgsz=640,
    project="runs/ablation/DAAM+TDH/epoch150"
)
