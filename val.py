from ultralytics import YOLO

model = YOLO('runs/plane-yolo/train/weights/best.pt')

model.val(data='CORS-SAR_sar.yaml', batch=1, split='test', project='runs/plane-yolo')