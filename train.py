from ultralytics import YOLO

# 1. 必须指定为你修改后的 OptiSAR-Net.yaml
model = YOLO("OptiSAR-Net-Plane.yaml")

# 2. 修改 imgsz 建议为 800，以匹配遥感图像中飞机的多尺度特性
# 3. 确保 data 指向你包含飞机数据的 yaml 文件
model.train(
    data="CORS-SAR.yaml", # 确认这个文件包含飞机类别
    epochs=100,
    batch=16,             # 增加 Topk 和池化分支后显存占用会增加，如报错请调小 batch
    imgsz=800,
    project="runs/airplane/OptisarNet-Plane/CAAM+LKA_SPPF+VCAA",
    device=0              # 指定 GPU
)
