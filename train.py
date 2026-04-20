from ultralytics import YOLO

# 1. 直接加载官方原生 YOLOv10n 预训练权重
# 框架会自动根据后续传入的 yaml 数据集配置调整头部维度
model = YOLO("runs/ablation/DAAM+P2/train/weights/last.pt")

# 2. 开始跨模态双源数据的基线训练
model.train(
    resume=True
)