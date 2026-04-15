from ultralytics import YOLO

# 1. 极其关键：构建自定义模型后，强制加载官方预训练权重！
# 系统会自动匹配名字和结构相同的层（即我们的标准 Backbone），忽略不匹配的新层
model = YOLO("runs/airplane/OptisarNet-Plane/DDSV3.0_4Head/train/weights/last.pt")

# 2. 开始训练
model.train(
    resume=True                 # 指定单卡 GPU
)