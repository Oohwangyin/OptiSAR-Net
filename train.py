from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

# 训练参数配置
model.train(
    data="M4-SAR-sampled_0.1.yaml",  # 使用 10% 子集进行快速验证
    epochs=100,  # 小数据集可以减少 epoch 数
    batch=16,
    imgsz=640,
    device=0,  # 使用 GPU
    workers=8,  # 数据加载线程数
    optimizer='AdamW',  # 优化器选择
    lr0=0.0005,  # 初始学习率
    lrf=0.005,  # 最终学习率 (lr0 * lrf)
    momentum=0.937,  # SGD momentum/Adam beta1
    weight_decay=0.0005,  # 权重衰减
    warmup_epochs=3.0,  # 预热轮数
    warmup_momentum=0.8,  # 预热动量
    box=7.5,  # box loss gain
    cls=1.5,  # cls loss gain (增加到 1.5，让模型更关注分类)
    dfl=1.5,  # dfl loss gain
    hsv_h=0.015,  # HSV-Hue augmentation (fraction)
    hsv_s=0.7,  # HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # HSV-Value augmentation (fraction)
    degrees=0.0,  # rotation augmentation (degrees)
    translate=0.1,  # translation augmentation (fraction)
    scale=0.5,  # scale augmentation (fraction)
    shear=0.0,  # shear augmentation (degrees)
    perspective=0.0,  # perspective augmentation (fraction)
    flipud=0.0,  # probability of flip up-down
    fliplr=0.5,  # probability of flip left-right
    mosaic=1.0,  # probability of mosaic augmentation
    mixup=0.1,  # probability of mixup augmentation
    copy_paste=0.3,  # probability of copy-paste augmentation
    patience=50,  # 小数据集可以适当减少 patience
    save_period=10,  # Save checkpoint every x epochs
    seed=42,  # random seed
    exist_ok=False,  # overwrite existing experiment
    pretrained=True,  # use pretrained weights
    amp=True,  # automatic mixed precision training
    project='runs/detect',  # 保存项目的目录
    name='OptiSAR-Net-M4-SAR-10pct',  # 实验名称
)