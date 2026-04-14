from ultralytics import YOLO

# 加载你训练好的最佳跨模态模型权重
model = YOLO('runs/airplane/OptisarNet-Plane/DFDA+DSA_SPPF+BOD/train/weights/best.pt')

# 执行测试集评估
metrics = model.val(
    data='CORS-SAR-sar.yaml',
    split='test',             # 明确指定使用 test 集
    batch=16,                 # 建议调大：batch=1 测试速度极慢，只要显存不爆，设为 8 或 16 可以大幅缩短测试时间
    imgsz=800,                # 关键：务必与你训练时设置的 imgsz 保持一致（如果在训练时设了 800）
    project="runs/airplane/OptisarNet-Plane/DFDA+DSA_SPPF+BOD"
)